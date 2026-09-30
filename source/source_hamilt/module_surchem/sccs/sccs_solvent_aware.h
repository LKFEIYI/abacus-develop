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

// Environ solvent-aware interface (Andreussi et al., JCTC 15, 1996 (2019)):
// cavity voids too small for a solvent molecule are filled with solute.
struct SolventAwareParameters
{
    // Solvent radius in bohr; zero disables the filling.
    double solvent_radius = 0.0;
    // The probe sphere has radius solvent_radius * radial_scale.
    double radial_scale = 2.0;
    // erfc spread of the probe sphere, in bohr.
    double radial_spread = 0.5;
    // A point is filled when the solute fraction of its probe sphere passes
    // filling_threshold, over an erfc step of width filling_spread.
    double filling_threshold = 0.825;
    double filling_spread = 0.02;
};

bool uses_solvent_aware(const SolventAwareParameters& parameters);

// Environ's ranges, except that filling_threshold must also be below one:
// the solute fraction never exceeds one, so a larger threshold fills nothing.
void validate_solvent_aware_parameters(const SolventAwareParameters& parameters);

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

// Filled boundary s_sa = s + (1 - s) f(p * s), with the filling function
// f(c) = 1 - erfc((c - filling_threshold)/filling_spread)/2 and its
// derivatives at the filled fraction c = p * s (Environ solvent_aware_boundary).
struct SolventAwareBoundary
{
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
// and Laplacian of the local boundary s into those of s_sa, with
// grad c = p * grad s and lapl c = p * lapl s.
void solvent_aware_chain_derivatives(const std::vector<double>& local,
                                     const SolventAwareBoundary& filled,
                                     const std::vector<double>& kernel,
                                     const ModulePW::PW_Basis& basis,
                                     std::vector<ModuleBase::Vector3<double>>& gradient,
                                     std::vector<double>& laplacian);

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
