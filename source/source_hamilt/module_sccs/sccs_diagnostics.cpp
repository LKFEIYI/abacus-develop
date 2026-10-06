#include "sccs_diagnostics.h"
#include "sccs_coulomb.h"
#include "sccs_response.h"
#include "sccs_pw_coulomb.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <cmath>
#include <complex>

namespace ModuleSccs
{
bool check_sccs_fixed_point(const std::vector<double>& charge,
                           const std::vector<double>& coefficient,
                           const std::vector<double>& potential,
                           const std::vector<double>& invsqrt,
                           const ModulePW::PW_Basis& basis,
                           CoulombOperator& coulomb,
                           PolarizationResult& result,
                           std::string& error)
{
    double invalid_grid = basis.nxyz > 0 ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid_grid);
    if (invalid_grid != 0.0)
    {
        error = "SCCS fixed-point diagnostic requires a positive global grid size";
        return false;
    }
    if (!validate_grid_values(charge, basis, error)
        || !validate_grid_values(coefficient, basis, error)
        || !validate_grid_values(potential, basis, error)
        || !validate_grid_values(invsqrt, basis, error)) { return false; }
    std::vector<double> right(charge.size());
    for (std::size_t i = 0; i < right.size(); ++i)
    {
        right[i] = (charge[i] - coefficient[i] * potential[i]) * invsqrt[i];
    }
    std::vector<double> image;
    if (!coulomb.apply_potential(right, image, error) || !validate_grid_values(image, basis, error))
    {
        return false;
    }
    double square = 0.0;
    double maximum = 0.0;
    for (std::size_t i = 0; i < right.size(); ++i)
    {
        const double defect = potential[i] - invsqrt[i] * image[i];
        square += defect * defect;
        const double absolute = std::abs(defect);
        if (absolute > maximum) { maximum = absolute; }
    }
    Parallel_Reduce::reduce_pool(square);
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, maximum);
    if (!std::isfinite(square) || !std::isfinite(maximum))
    {
        error = "SCCS fixed-point diagnostic is not finite";
        return false;
    }
    const double mean_square = square / basis.nxyz;
    result.fixed_point_defect_rms = std::sqrt(mean_square);
    result.fixed_point_defect_max = maximum;
    result.fixed_point_checked = true;
    return true;
}

bool continuum_polarization_charge(const std::vector<double>& charge,
                                    const SccsResponse& response,
                                    const ModulePW::PW_Basis& basis,
                                    double tpiba,
                                    std::vector<double>& density,
                                    std::string& error)
{
    if (!validate_pw_grid(basis, tpiba, error) || !validate_grid_values(charge, basis, error)
        || !validate_grid_values(response.epsilon, basis, error)
        || !validate_grid_values(response.polarization.potential, basis, error)) { return false; }
    double invalid = response.grad_log_epsilon.size() == charge.size() ? 0.0 : 1.0;
    for (double epsilon : response.epsilon)
    {
        if (epsilon <= 0.0) { invalid = 1.0; }
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS polarization diagnostic requires positive epsilon and matching gradients";
        return false;
    }
    std::vector<std::complex<double>> potential_g(basis.npw);
    basis.real2recip(response.polarization.potential.data(), potential_g.data());
    std::vector<ModuleBase::Vector3<double>> gradient(charge.size());
    XC_Functional::grad_rho(potential_g.data(), gradient.data(), &basis, tpiba);
    std::vector<double> candidate(charge.size());
    for (std::size_t i = 0; i < charge.size(); ++i)
    {
        const double contraction = response.grad_log_epsilon[i] * gradient[i];
        candidate[i] = contraction / ModuleBase::FOUR_PI + charge[i] * (1.0 / response.epsilon[i] - 1.0);
    }
    if (!validate_grid_values(candidate, basis, error)) { return false; }
    density.swap(candidate);
    return true;
}
} // namespace ModuleSccs
