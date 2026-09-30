#ifndef SCCS_CAVITY_H
#define SCCS_CAVITY_H

namespace ModuleSccs
{

struct CavityParameters
{
    double density_min = 0.0;
    double density_max = 0.0;
    double epsilon_bulk = 0.0;
    // Environ deriv_lowpass_p1/p2: both positive filter the switching-function
    // derivatives by 0.5 erfc(p1 G^2/Gcut^2 - p2); both non-positive disable it.
    double lowpass_p1 = -1.0;
    double lowpass_p2 = -1.0;
};

struct CavityPoint
{
    double solute = 0.0;
    double dsolute_drho = 0.0;
    double d2solute_drho2 = 0.0;
    double epsilon = 0.0;
    double depsilon_drho = 0.0;
};

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

void validate_cavity_parameters(const CavityParameters& parameters);

bool uses_switching_lowpass(const CavityParameters& parameters);

CavityPoint evaluate_cavity(double density, const CavityParameters& parameters);

} // namespace ModuleSccs

#endif
