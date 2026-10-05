#ifndef SCCS_DIAGNOSTICS_H
#define SCCS_DIAGNOSTICS_H

#include <string>
#include <vector>

namespace ModulePW { class PW_Basis; }
namespace ModuleSccs
{
class CoulombOperator;
struct PolarizationResult;
struct SccsResponse;

// Extra Poisson application on the unshifted sqrt-CG solution; collective.
bool check_sccs_fixed_point(const std::vector<double>& charge,
                           const std::vector<double>& coefficient,
                           const std::vector<double>& potential,
                           const std::vector<double>& invsqrt,
                           const ModulePW::PW_Basis& basis,
                           CoulombOperator& coulomb,
                           PolarizationResult& result,
                           std::string& error);

// ENVIRON dielectric_of_potential; diagnostic density, not a new solve source.
bool continuum_polarization_charge(const std::vector<double>& charge,
                                    const SccsResponse& response,
                                    const ModulePW::PW_Basis& basis,
                                    double tpiba,
                                    std::vector<double>& density,
                                    std::string& error);
} // namespace ModuleSccs
#endif
