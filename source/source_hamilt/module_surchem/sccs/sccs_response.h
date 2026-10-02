#ifndef SCCS_RESPONSE_H
#define SCCS_RESPONSE_H

#include "sccs_cavity.h"
#include "sccs_poisson.h"
#include "sccs_solvent_aware.h"

#include <complex>

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSurchem
{
class ChargeReduction;
}

namespace ModuleSccs
{

// Dielectric response of one SCCS evaluation: the cavity fields and the
// sqrt-CG solution for the solute charge.
struct SccsResponse
{
    // Dielectric boundary s (one inside the solute) and ds/dn of the cavity
    // density n; epsilon = exp(ln(eps_bulk) (1 - s)). With the solvent-aware
    // filling, solute is s_sa and dsolute_drho belongs to local_solute.
    std::vector<double> solute;
    std::vector<double> dsolute_drho;
    // Solvent-aware only: the boundary s(n) before filling and the filling;
    // empty otherwise.
    std::vector<double> local_solute;
    SolventAwareBoundary filling;
    // Grid integral of s_sa - s, in bohr^3; zero without the filling.
    double filled_volume = 0.0;
    // Solvent-aware only, empty otherwise: the spectral grad n of the cavity
    // density and grad c = p * grad s with the chain grad s = s' grad n, kept
    // for solvent_aware_surface_of_density with every boundary condition.
    std::vector<ModuleBase::Vector3<double>> density_gradient;
    std::vector<ModuleBase::Vector3<double>> fraction_gradient;
    // Solvent-aware only, empty otherwise: the Fourier coefficients of the
    // cavity density on the PW_Basis G vectors, for the chain Hessian of the
    // surface.
    std::vector<std::complex<double>> density_reciprocal;
    std::vector<double> epsilon;
    std::vector<ModuleBase::Vector3<double>> grad_log_epsilon;
    PolarizationResult polarization;
    // Derivative of the reaction energy with respect to the boundary s, in
    // Ha. With the switching lowpass (PCC only) it is the exact derivative of
    // the discrete sqrt-CG energy; otherwise it is the continuum
    // L eps |grad v|^2/(8 pi) of Environ, L = ln(eps_bulk).
    std::vector<double> boundary_potential;
    // The same derivative with respect to the cavity density.
    std::vector<double> cavity_potential;
    // Unshifted sqrt-CG solution: the fixed point used for the next warm start.
    std::vector<double> restart_potential;
    // Open-boundary (PCC) solutions only: the net polarization charge seen by
    // the far field, which screens the solute to int(s)/sqrt(eps_bulk) because
    // sqrt(eps) v = C_PCC(s) with s = (q - f v)/sqrt(eps). Zero when periodic.
    double far_field_polarization_charge = 0.0;
};

// Chain a derivative with respect to the boundary to the cavity density
// (Environ calculator: sa_de_dboundary, then de_dboundary * dscaled).
// probe_kernel is the solvent-aware probe, empty without the filling.
std::vector<double> boundary_to_density_potential(const SccsResponse& response,
                                                  const std::vector<double>& probe_kernel,
                                                  const ModulePW::PW_Basis& basis,
                                                  const std::vector<double>& boundary_potential);

// Solvent-aware surface (Environ boundary_of_density with deriv_method
// 'chain' and need_hessian): grad s = s' grad n and
// H s = s' H n + s'' grad n grad n^T from the spectral density derivatives,
// passed to solvent_aware_surface for the filled boundary of response. grad n,
// grad c and the density coefficients come from response, which must be the
// solvent-aware solution for this density. With PCC the dielectric still differentiates s_sa on the FFT
// grid; the surface uses these chain derivatives as in the periodic path.
SolventAwareSurface solvent_aware_surface_of_density(const std::vector<double>& density,
                                                     const CavityParameters& cavity,
                                                     const SccsResponse& response,
                                                     const std::vector<double>& probe_kernel,
                                                     const ModulePW::PW_Basis& basis,
                                                     double tpiba,
                                                     double regularization);

// ENVIRON dielectric_of_potential polarization density,
// grad(ln eps).grad(v)/(4 pi) + q (1/eps - 1). On a finite grid it need not
// equal -laplacian(v)/(4 pi) - q; ABACUS uses it only for PCC diagnostics.
std::vector<double> continuum_polarization_charge(
    const std::vector<double>& solute_charge,
    const SccsResponse& response);

// ENVIRON sqrt-preconditioned CG for every boundary: the preconditioner uses
// coulomb (periodic or PCC-corrected). It stops on the RMS and maximum charge
// residual and warm-starts from initial_potential (previous solution, or empty)
// when that helps. probe_kernel (solvent_probe_kernel) is empty unless
// solvent_aware is enabled.
SccsResponse solve_sccs_response(
    const std::vector<double>& cavity_density,
    const std::vector<double>& solute_charge,
    const CavityParameters& cavity_parameters,
    const SolventAwareParameters& solvent_aware,
    const std::vector<double>& probe_kernel,
    const PolarizationSolverParameters& solver_parameters,
    const std::vector<double>& initial_potential,
    const ModulePW::PW_Basis& basis,
    double tpiba,
    const CoulombOperator& coulomb,
    const ModuleSurchem::ChargeReduction& reduction);

} // namespace ModuleSccs

#endif
