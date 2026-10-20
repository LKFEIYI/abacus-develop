#ifndef SCCS_DIAG_H
#define SCCS_DIAG_H

#include <vector>

namespace ModulePW { class PW_Basis; }
namespace ModuleSccs
{
class CoulombOperator;
struct PolarizationResult;
struct SccsResponse;

// Extra Poisson application on the unshifted sqrt-CG solution; collective.
void check_fixed_pt(const std::vector<double>& charge,
                    const std::vector<double>& coefficient,
                    const std::vector<double>& potential,
                    const std::vector<double>& invsqrt,
                    const ModulePW::PW_Basis& basis,
                    CoulombOperator& coulomb,
                    PolarizationResult& result);

// Environ dielectric_of_potential; diagnostic density, not a new solve source.
void cont_pol_charge(const std::vector<double>& charge,
                     const SccsResponse& response,
                     const ModulePW::PW_Basis& basis,
                     double tpiba,
                     std::vector<double>& density);
} // namespace ModuleSccs
#endif
