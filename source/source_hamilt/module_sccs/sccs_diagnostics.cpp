#include "sccs_diagnostics.h"
#include "sccs_coulomb.h"
#include "sccs_response.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <cmath>
#include <complex>

namespace ModuleSccs
{
void check_sccs_fixed_point(const std::vector<double>& charge,
                            const std::vector<double>& coefficient,
                            const std::vector<double>& potential,
                            const std::vector<double>& invsqrt,
                            const ModulePW::PW_Basis& basis,
                            CoulombOperator& coulomb,
                            PolarizationResult& result)
{
    std::vector<double> right(charge.size());
    for (std::size_t i = 0; i < right.size(); ++i)
    {
        right[i] = (charge[i] - coefficient[i] * potential[i]) * invsqrt[i];
    }
    std::vector<double> image;
    coulomb.apply_potential(right, image);
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
    const double mean_square = square / basis.nxyz;
    result.fixed_point_defect_rms = std::sqrt(mean_square);
    result.fixed_point_defect_max = maximum;
    result.fixed_point_checked = true;
}

void continuum_polarization_charge(const std::vector<double>& charge,
                                   const SccsResponse& response,
                                   const ModulePW::PW_Basis& basis,
                                   double tpiba,
                                   std::vector<double>& density)
{
    std::vector<std::complex<double>> potential_g(basis.npw);
    basis.real2recip(response.polarization.potential.data(), potential_g.data());
    std::vector<ModuleBase::Vector3<double>> gradient(charge.size());
    XC_Functional::grad_rho(potential_g.data(), gradient.data(), &basis, tpiba);
    density.resize(charge.size());
    for (std::size_t i = 0; i < charge.size(); ++i)
    {
        const double contraction = response.grad_log_epsilon[i] * gradient[i];
        density[i] = contraction / ModuleBase::FOUR_PI + charge[i] * (1.0 / response.epsilon[i] - 1.0);
    }
}
} // namespace ModuleSccs
