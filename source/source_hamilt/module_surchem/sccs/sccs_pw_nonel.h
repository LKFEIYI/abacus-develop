#ifndef SCCS_PW_NONEL_H
#define SCCS_PW_NONEL_H

#include "sccs_charge.h"
#include "sccs_nonel.h"

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

} // namespace ModuleSccs

#endif
