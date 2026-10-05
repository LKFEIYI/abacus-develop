#include "sccs_lowpass.h"
#include "sccs_cavity.h"
#include "sccs_cavity_derivatives.h"
#include "sccs_pw_coulomb.h"
#include "sccs_response.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"

#include <cmath>
#include <complex>
#include <utility>

namespace ModuleSccs
{
bool make_switching_filter(const CavityParameters& cavity,
                           const ModulePW::PW_Basis& basis,
                           std::vector<double>& filter,
                           std::string& error)
{
    double invalid = 0.0;
    const bool lowpass = uses_switching_lowpass(cavity);
    if (!validate_cavity_parameters(cavity, error)
        || (lowpass && (!std::isfinite(basis.ggecut) || basis.ggecut <= 0.0)))
    {
        invalid = 1.0;
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS lowpass requires valid parameters and a finite positive density cutoff on every pool rank";
        return false;
    }
    std::vector<double> candidate;
    if (lowpass)
    {
        candidate.resize(basis.npw);
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            // Environ deriv_lowpass: Gcut is the density cutoff, not ecutwfc.
            const double argument = cavity.lowpass_p1 * basis.gg[ig] / basis.ggecut - cavity.lowpass_p2;
            candidate[ig] = 0.5 * std::erfc(argument);
        }
    }
    filter = std::move(candidate);
    return true;
}

bool evaluate_lowpass_cavity_potential(const std::vector<double>& charge,
                                      const CavityParameters& cavity,
                                      const CavityDerivatives& derivatives,
                                      const ModulePW::PW_Basis& basis,
                                      double tpiba,
                                      SccsResponse& response,
                                      std::string& error)
{
    if (!validate_pw_grid(basis, tpiba, error) || !validate_grid_values(charge, basis, error)
        || !validate_grid_values(response.epsilon, basis, error)
        || !validate_grid_values(response.dsolute_drho, basis, error)
        || !validate_grid_values(response.polarization.potential, basis, error))
    {
        return false;
    }
    double invalid = 0.0;
    if (!validate_cavity_parameters(cavity, error) || !uses_switching_lowpass(cavity)
        || derivatives.filter.size() != static_cast<std::size_t>(basis.npw)
        || derivatives.gradient.size() != charge.size())
    {
        invalid = 1.0;
    }
    for (double value : derivatives.filter)
    {
        if (!std::isfinite(value) || value < 0.0 || value > 1.0) { invalid = 1.0; }
    }
    for (const auto& value : derivatives.gradient)
    {
        if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) { invalid = 1.0; }
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS discrete cavity potential requires a valid lowpass filter and switching gradient";
        return false;
    }
    const std::size_t size = charge.size();
    const double log_bulk = std::log(cavity.epsilon_bulk);
    const std::vector<double>& potential = response.polarization.potential;
    std::vector<double> weight(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        weight[i] = response.epsilon[i] * potential[i] * potential[i] / (8.0 * ModuleBase::PI);
    }
    // A = sqrt(eps) C^-1 sqrt(eps) + F is symmetric. For E=q^T A^-1 q/2,
    // dE/ds = L/2 (q v + lapl b + L div(b grad s)), b=eps v^2/(8 pi).
    // The filtered derivative transposes share one inverse FFT.
    std::vector<std::complex<double>> transpose_g(basis.npw);
    basis.real2recip(weight.data(), transpose_g.data());
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        transpose_g[ig] *= -tpiba * tpiba * basis.gg[ig] * derivatives.filter[ig];
    }
    std::vector<double> component(size);
    std::vector<std::complex<double>> component_g(basis.npw);
    for (int d = 0; d < 3; ++d)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            component[i] = weight[i] * derivatives.gradient[i][d];
        }
        basis.real2recip(component.data(), component_g.data());
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            transpose_g[ig] += log_bulk * ModuleBase::IMAG_UNIT * tpiba * basis.gcar[ig][d]
                               * component_g[ig] * derivatives.filter[ig];
        }
    }
    std::vector<double> transpose(size);
    basis.recip2real(transpose_g.data(), transpose.data());
    std::vector<double> candidate(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        const double boundary_potential = 0.5 * log_bulk * (charge[i] * potential[i] + transpose[i]);
        candidate[i] = boundary_potential * response.dsolute_drho[i];
    }
    if (!validate_grid_values(candidate, basis, error)) { return false; }
    response.cavity_potential = std::move(candidate);
    return true;
}
} // namespace ModuleSccs
