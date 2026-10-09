#ifndef SCCS_SOLVENT_AWARE_H
#define SCCS_SOLVENT_AWARE_H

#include "source_base/vector3.h"

#include <vector>

namespace ModuleBase
{
class Matrix3;
}
namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSccs
{
struct SolventAwareParameters;

// Fourier coefficients on the local G vectors of the probe
// erfc((|r| - R)/radial_spread), R = solvent_radius * radial_scale, cut at
// R + 5 radial_spread and normalized to unit grid integral (Environ
// solvent_probe). Environ samples the probe at the minimum-image distance,
// which truncates it in cells narrower than 2 (R + 5 radial_spread); every
// periodic image within the cutoff is summed here, which gives the same grid
// values in wider cells. lattice rows are in units of lattice_scale Bohr.
// Collective over the PW pool.
std::vector<double> solvent_probe_kernel(const ModulePW::PW_Basis& basis,
                                         const ModuleBase::Matrix3& lattice,
                                         double lattice_scale,
                                         const SolventAwareParameters& parameters);

// Periodic convolution with the probe.
std::vector<double> convolve_probe(const std::vector<double>& kernel,
                                   const ModulePW::PW_Basis& basis,
                                   const std::vector<double>& values);

// Component-wise convolution; of grad s it is grad c = p * grad s.
std::vector<ModuleBase::Vector3<double>> convolve_probe_gradient(
    const std::vector<double>& kernel,
    const ModulePW::PW_Basis& basis,
    const std::vector<ModuleBase::Vector3<double>>& gradient);

// Filled boundary s_sa = s + (1 - s) f(c) of the filled fraction c = p * s,
// f(c) = 1 - erfc((c - filling_threshold)/filling_spread)/2, with f' and f''
// (Environ solvent_aware_boundary).
struct SolventAwareBoundary
{
    std::vector<double> fraction;
    std::vector<double> filling;
    std::vector<double> dfilling;
    std::vector<double> d2filling;
    std::vector<double> boundary;
};

SolventAwareBoundary solvent_aware_boundary(const std::vector<double>& local,
                                            const std::vector<double>& kernel,
                                            const SolventAwareParameters& parameters,
                                            const ModulePW::PW_Basis& basis);

// Solvent-aware state of an SCCS cavity, empty without the filling: the
// local boundary s, its filling, the filled volume int(s_sa - s) in Bohr^3
// and the chain data of the filled surface, s'' = d2s/dn2, grad n and the
// gradient of the filled fraction, grad c = p * (s' grad n).
struct FilledCavity
{
    bool enabled() const { return !local.empty(); }

    std::vector<double> local;
    SolventAwareBoundary filling;
    double volume = 0.0;
    std::vector<double> d2solute_drho2;
    std::vector<ModuleBase::Vector3<double>> density_gradient;
    std::vector<ModuleBase::Vector3<double>> fraction_gradient;
};

// Environ solvent_aware_boundary, deriv_method 'chain': turn grad s and
// lapl s of the local boundary into those of s_sa, given the derivatives of
// the filled fraction, fraction_gradient = grad c = p * grad s and
// fraction_laplacian = lapl c = p * lapl s.
void solvent_aware_chain_derivatives(const std::vector<double>& local,
                                     const SolventAwareBoundary& filled,
                                     const std::vector<ModuleBase::Vector3<double>>& fraction_gradient,
                                     const std::vector<double>& fraction_laplacian,
                                     std::vector<ModuleBase::Vector3<double>>& gradient,
                                     std::vector<double>& laplacian);

// Surface and volume terms of the filled boundary, with a = s' grad n and
// b = p * a:
//   chain gradient   g = (1 - f) a + (1 - s) f' b
//   discrete surface S = sum_i (|g_i|_r - r), |g|_r = sqrt(|g|^2 + r^2)
//   volume           V = sum_i s_sa,i,  dV/dn = s' ((1 - f) + p * ((1 - s) f'))
// With u = g/|g|_r, A = (1 - f) u + p * ((1 - s) f' u),
// alpha = -f' u.a + (1 - s) f'' u.b and beta = -f' u.b, the exact derivative
// with respect to the cavity density at each grid point is
//   dS/dn = s' (beta + p * alpha) + s'' A.grad n - div(s' A),
// since the probe is even and the spectral divergence is minus the transpose
// of the spectral gradient. Only |u| <= 1 enters, so it stays bounded where g
// vanishes, unlike Environ's continuum curvature (g.H.g - |g|^2 tr H)/|g|^3.
// density_potential is surface_tension dS/dn + pressure dV/dn; both share one
// probe convolution. Collective over the PW pool.
struct SolventAwareNonelectrostatic
{
    std::vector<ModuleBase::Vector3<double>> gradient;
    std::vector<double> density_potential;
};

SolventAwareNonelectrostatic solvent_aware_nonelectrostatic(const FilledCavity& cavity,
                                                            const std::vector<double>& dsolute_drho,
                                                            const std::vector<double>& kernel,
                                                            const ModulePW::PW_Basis& basis,
                                                            double tpiba,
                                                            double regularization,
                                                            double surface_tension,
                                                            double pressure);

// Environ calc_solvent_aware_de_dboundary, from dE/ds_sa to dE/ds:
// (1 - f) dE/ds_sa + p * ((1 - s) f' dE/ds_sa). The probe is even, so the
// convolution is its own transpose and this is the exact adjoint.
std::vector<double> solvent_aware_adjoint(const FilledCavity& cavity,
                                          const std::vector<double>& kernel,
                                          const ModulePW::PW_Basis& basis,
                                          const std::vector<double>& boundary_potential);
} // namespace ModuleSccs

#endif
