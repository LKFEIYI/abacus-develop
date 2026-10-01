#ifndef SCCS_PW_NONEL_H
#define SCCS_PW_NONEL_H

#include "sccs_charge.h"
#include "sccs_nonel.h"

namespace ModuleBase
{
template <class T> class Vector3;
}

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSccs
{

// Surface and volume of the boundary s and their energies; the potential is
// the derivative with respect to s.
NonElectrostaticResult evaluate_pw_non_electrostatic(
    const ModulePW::PW_Basis& basis,
    double tpiba,
    double volume_element,
    const NonElectrostaticParameters& parameters,
    const std::vector<double>& solute,
    const ModuleSurchem::ChargeReduction& reduction);

// Surface and volume of the boundary s from a given gradient of s and the
// surface derivative -div(grad s/|grad s|_r) (solvent_aware_surface), for the
// periodic solvent-aware path where the spectral gradient of the sharply
// filled s would ring. The potential pressure + tension * surface_derivative
// is the continuum derivative with respect to s (Environ de_dboundary).
NonElectrostaticResult evaluate_chain_non_electrostatic(
    double volume_element,
    const NonElectrostaticParameters& parameters,
    const std::vector<double>& solute,
    const std::vector<ModuleBase::Vector3<double>>& gradient,
    const std::vector<double>& surface_derivative,
    const ModuleSurchem::ChargeReduction& reduction);

} // namespace ModuleSccs

#endif
