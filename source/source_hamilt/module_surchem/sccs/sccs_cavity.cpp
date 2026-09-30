#include "sccs_cavity.h"

#include "source_base/constants.h"

#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{

void validate_cavity_parameters(const CavityParameters& parameters)
{
    if (!std::isfinite(parameters.density_min) || !std::isfinite(parameters.density_max)
        || !std::isfinite(parameters.epsilon_bulk))
    {
        throw std::invalid_argument("SCCS cavity parameters must be finite");
    }
    if (parameters.density_min <= 0.0 || parameters.density_max <= parameters.density_min)
    {
        throw std::invalid_argument("SCCS cavity requires 0 < density_min < density_max");
    }
    if (parameters.epsilon_bulk < 1.0)
    {
        throw std::invalid_argument("SCCS bulk permittivity must be at least one");
    }
    if (!std::isfinite(parameters.lowpass_p1) || !std::isfinite(parameters.lowpass_p2)
        || (parameters.lowpass_p1 > 0.0) != (parameters.lowpass_p2 > 0.0))
    {
        throw std::invalid_argument("SCCS lowpass parameters must be finite and both positive or both non-positive");
    }
}

bool uses_solvent_aware(const SolventAwareParameters& parameters)
{
    return parameters.solvent_radius > 0.0;
}

void validate_solvent_aware_parameters(const SolventAwareParameters& parameters)
{
    if (!std::isfinite(parameters.solvent_radius) || parameters.solvent_radius < 0.0)
    {
        throw std::invalid_argument("SCCS solvent radius must be finite and non-negative");
    }
    if (!std::isfinite(parameters.radial_scale) || parameters.radial_scale < 1.0)
    {
        throw std::invalid_argument("SCCS solvent-aware radial scale must be at least one");
    }
    if (!std::isfinite(parameters.radial_spread) || parameters.radial_spread <= 0.0)
    {
        throw std::invalid_argument("SCCS solvent-aware radial spread must be positive");
    }
    if (!std::isfinite(parameters.filling_threshold) || parameters.filling_threshold <= 0.0
        || parameters.filling_threshold >= 1.0)
    {
        throw std::invalid_argument("SCCS filling threshold must lie between zero and one");
    }
    if (!std::isfinite(parameters.filling_spread) || parameters.filling_spread <= 0.0)
    {
        throw std::invalid_argument("SCCS filling spread must be positive");
    }
}

bool uses_switching_lowpass(const CavityParameters& parameters)
{
    return parameters.lowpass_p1 > 0.0 && parameters.lowpass_p2 > 0.0;
}

CavityPoint evaluate_cavity(const double density, const CavityParameters& parameters)
{
    if (!std::isfinite(density))
    {
        throw std::domain_error("SCCS cavity density must be finite");
    }

    CavityPoint result;
    // LCAO-to-grid transforms can produce negative Fourier ringing in the
    // vacuum. Every such value is below density_min and therefore belongs to
    // the constant bulk-solvent branch, whose density derivative is zero.
    if (density <= parameters.density_min)
    {
        result.epsilon = parameters.epsilon_bulk;
        return result;
    }
    if (density >= parameters.density_max)
    {
        result.solute = 1.0;
        result.epsilon = 1.0;
        return result;
    }

    const double log_width = std::log(parameters.density_max / parameters.density_min);
    const double x = std::log(parameters.density_max / density) / log_width;
    const double solvent = x - std::sin(ModuleBase::TWO_PI * x) / ModuleBase::TWO_PI;
    const double angle = ModuleBase::TWO_PI * x;
    const double dsolvent_drho = -(1.0 - std::cos(angle)) / (log_width * density);
    const double d2solvent_drho2
        = (1.0 - std::cos(angle) + ModuleBase::TWO_PI * std::sin(angle) / log_width)
          / (log_width * density * density);
    const double log_epsilon = std::log(parameters.epsilon_bulk);

    result.solute = 1.0 - solvent;
    result.dsolute_drho = -dsolvent_drho;
    result.d2solute_drho2 = -d2solvent_drho2;
    result.epsilon = std::exp(log_epsilon * solvent);
    result.depsilon_drho = result.epsilon * log_epsilon * dsolvent_drho;
    return result;
}

} // namespace ModuleSccs
