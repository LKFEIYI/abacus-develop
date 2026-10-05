#include "sccs_cavity_derivatives.h"
#include "sccs_cavity.h"
#include "sccs_lowpass.h"
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
bool prepare_cavity_derivatives(const std::vector<double>& density,
                    const CavityParameters& cavity,
                    const ModulePW::PW_Basis& basis,
                    double tpiba,
                    bool open_boundary,
                    SccsResponse& response,
                    CavityDerivatives& derivatives,
                    std::string& error)
{
    if (!make_switching_filter(cavity, basis, derivatives.filter, error))
    {
        return false;
    }
    const bool lowpass = uses_switching_lowpass(cavity);
    if (lowpass && !open_boundary)
    {
        error = "SCCS lowpass requires an open-boundary Coulomb operator";
        return false;
    }
    derivatives.gradient.clear();
    std::vector<double>& coefficient = derivatives.coefficient;
    const std::size_t size = density.size();
    response.solute.resize(size);
    response.dsolute_drho.resize(size);
    response.epsilon.resize(size);
    response.depsilon_drho.resize(size);
    response.grad_log_epsilon.resize(size);
    double invalid = 0.0;
    for (std::size_t i = 0; i < size; ++i)
    {
        CavityPoint point;
        if (!evaluate_cavity(density[i], cavity, point, error))
        {
            invalid = 1.0;
            continue;
        }
        response.solute[i] = point.solute;
        response.dsolute_drho[i] = point.dsolute_drho;
        response.epsilon[i] = point.epsilon;
        response.depsilon_drho[i] = point.depsilon_drho;
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS cavity evaluation failed on a pool rank";
        return false;
    }

    if (open_boundary)
    {
        // Original PCC path: differentiate the switching boundary on the FFT
        // grid, since its chain coefficient is not converged at the cavity edge.
        std::vector<std::complex<double>> boundary_g(basis.npw);
        basis.real2recip(response.solute.data(), boundary_g.data());
        if (lowpass)
        {
            for (int ig = 0; ig < basis.npw; ++ig)
            {
                boundary_g[ig] *= derivatives.filter[ig];
            }
        }
        std::vector<ModuleBase::Vector3<double>> gradient(size);
        std::vector<double> laplacian(size);
        XC_Functional::grad_rho(boundary_g.data(), gradient.data(), &basis, tpiba);
        XC_Functional::laplacian_rho(boundary_g.data(), laplacian.data(), &basis, tpiba);
        const double log_bulk = std::log(cavity.epsilon_bulk);
        coefficient.resize(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            const double log_epsilon = log_bulk * (1.0 - response.solute[i]);
            response.epsilon[i] = std::exp(log_epsilon);
            const ModuleBase::Vector3<double>& local_gradient = gradient[i];
            const double gradient_square = local_gradient * local_gradient;
            response.grad_log_epsilon[i] = local_gradient * (-log_bulk);
            const double lap_log = -log_bulk * laplacian[i];
            coefficient[i] = response.epsilon[i]
                             * (0.5 * lap_log + 0.25 * log_bulk * log_bulk * gradient_square)
                             / ModuleBase::FOUR_PI;
        }
        if (lowpass) { derivatives.gradient.swap(gradient); }
        return validate_grid_values(coefficient, basis, error);
    }

    std::vector<std::complex<double>> density_g(basis.npw);
    basis.real2recip(density.data(), density_g.data());
    std::vector<ModuleBase::Vector3<double>> gradient(size);
    std::vector<double> laplacian(size);
    // These XC helpers use G in units of tpiba and ABACUS's normalized FFT.
    XC_Functional::grad_rho(density_g.data(), gradient.data(), &basis, tpiba);
    XC_Functional::laplacian_rho(density_g.data(), laplacian.data(), &basis, tpiba);
    const double density_ratio = cavity.density_max / cavity.density_min;
    const double width = std::log(density_ratio);
    const double log_bulk = std::log(cavity.epsilon_bulk);
    coefficient.resize(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        double second_log = 0.0;
        if (density[i] > cavity.density_min && density[i] < cavity.density_max)
        {
            const double local_ratio = cavity.density_max / density[i];
            const double x = std::log(local_ratio) / width;
            const double angle = ModuleBase::TWO_PI * x;
            second_log = log_bulk * (1.0 - std::cos(angle) + ModuleBase::TWO_PI * std::sin(angle) / width)
                         / (width * density[i] * density[i]);
        }
        const double first_log = response.depsilon_drho[i] / response.epsilon[i];
        const ModuleBase::Vector3<double>& local_gradient = gradient[i];
        const double gradient_square = local_gradient * local_gradient;
        response.grad_log_epsilon[i] = local_gradient * first_log;
        const double lap_log = first_log * laplacian[i] + second_log * gradient_square;
        coefficient[i] = response.epsilon[i] * (0.5 * lap_log + 0.25 * first_log * first_log * gradient_square)
                         / ModuleBase::FOUR_PI;
    }
    return validate_grid_values(coefficient, basis, error);
}

} // namespace ModuleSccs
