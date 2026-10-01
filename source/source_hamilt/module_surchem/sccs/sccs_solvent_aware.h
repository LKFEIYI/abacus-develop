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

namespace ModuleSurchem
{
class ChargeReduction;
}

namespace ModuleSccs
{

struct SolventAwareParameters;

// Fourier coefficients on the local G vectors of basis of the probe
// erfc((|r| - R)/radial_spread), R = solvent_radius * radial_scale, cut at
// R + 5 radial_spread and normalized to unit grid integral, so that
// convolving a constant keeps it (Environ solvent_probe). Environ samples the
// probe with the minimum-image distance, which truncates it in cells narrower
// than 2 (R + 5 radial_spread); here every periodic image within the cutoff
// is summed, which gives the same grid values in wider cells.
// lattice_vectors (rows, in units of lattice_scale bohr) define the cell.
std::vector<double> solvent_probe_kernel(const ModulePW::PW_Basis& basis,
                                         const ModuleBase::Matrix3& lattice_vectors,
                                         double lattice_scale,
                                         const SolventAwareParameters& parameters,
                                         const ModuleSurchem::ChargeReduction& reduction);

// Periodic convolution of a grid function with the probe.
std::vector<double> convolve_probe(const std::vector<double>& kernel,
                                   const ModulePW::PW_Basis& basis,
                                   const std::vector<double>& values);

// Component-wise probe convolution of a vector field; for the local boundary
// gradient it is grad c = p * grad s of the filled fraction.
std::vector<ModuleBase::Vector3<double>> convolve_probe_gradient(
    const std::vector<double>& kernel,
    const ModulePW::PW_Basis& basis,
    const std::vector<ModuleBase::Vector3<double>>& gradient);

// Filled boundary s_sa = s + (1 - s) f(p * s), with the filling function
// f(c) = 1 - erfc((c - filling_threshold)/filling_spread)/2 and its
// derivatives at the filled fraction c = p * s (Environ solvent_aware_boundary).
struct SolventAwareBoundary
{
    // Filled fraction c = p * s (the f of Andreussi et al. 2019).
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

// Environ solvent_aware_boundary with deriv_method 'chain': turn the gradient
// and Laplacian of the local boundary s into those of s_sa, given
// fraction_gradient = grad c = p * grad s (convolve_probe_gradient of the
// incoming gradient) and with lapl c = p * lapl s.
void solvent_aware_chain_derivatives(const std::vector<double>& local,
                                     const SolventAwareBoundary& filled,
                                     const std::vector<ModuleBase::Vector3<double>>& fraction_gradient,
                                     const std::vector<double>& kernel,
                                     const ModulePW::PW_Basis& basis,
                                     std::vector<ModuleBase::Vector3<double>>& gradient,
                                     std::vector<double>& laplacian);

// Number of independent components of a symmetric 3x3 Hessian; hessian_axes
// gives the two Cartesian axes of component k in the order xx, yy, zz, xy, xz, yz.
const int hessian_component_count = 6;
void hessian_axes(int component, int& first, int& second);

// Gradient g of s_sa and the derivative of its surface int(|g|_r - r) with
// respect to s_sa, -div(g/|g|_r) = (g.H.g - |g|_r^2 tr H)/|g|_r^3, where
// |g|_r = sqrt(|g|^2 + r^2) and H is the Hessian of s_sa (Environ
// calc_dsurface_no_pre for r -> 0).
struct SolventAwareSurface
{
    std::vector<ModuleBase::Vector3<double>> gradient;
    std::vector<double> surface_derivative;
};

// Environ solvent_aware_boundary with deriv_method 'chain' and need_hessian:
// from grad s, fraction_gradient = grad c = p * grad s (as passed to
// solvent_aware_chain_derivatives) and the components of H s (local_hessian,
// hessian_axes order), with H c = p * H s and
// H s_sa = (1 - f) H s - f' (grad s grad c^T + grad c grad s^T)
//          + (1 - s) (f'' grad c grad c^T + f' H c).
// Unlike a spectral gradient of s_sa, these stay smooth when the filling
// switches within a grid spacing.
SolventAwareSurface solvent_aware_surface(const std::vector<double>& local,
                                          const SolventAwareBoundary& filled,
                                          const std::vector<double>& kernel,
                                          const ModulePW::PW_Basis& basis,
                                          const std::vector<ModuleBase::Vector3<double>>& local_gradient,
                                          const std::vector<ModuleBase::Vector3<double>>& fraction_gradient,
                                          const std::vector<std::vector<double>>& local_hessian,
                                          double regularization);

// Environ calc_solvent_aware_de_dboundary: from dE/ds_sa to dE/ds,
// (1 - f) dE/ds_sa + p * ((1 - s) f' dE/ds_sa). The probe is even, so the
// convolution is its own transpose and the result is the exact adjoint.
std::vector<double> solvent_aware_adjoint(const std::vector<double>& local,
                                          const SolventAwareBoundary& filled,
                                          const std::vector<double>& kernel,
                                          const ModulePW::PW_Basis& basis,
                                          const std::vector<double>& boundary_potential);

} // namespace ModuleSccs

#endif
