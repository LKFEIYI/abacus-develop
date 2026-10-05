#ifndef SCCS_PCC_0D_ADAPTER_H
#define SCCS_PCC_0D_ADAPTER_H

#include "source_base/vector3.h"
#include <memory>
#include <string>
#include <vector>

class UnitCell;
namespace ModulePW
{
class PW_Basis;
}
namespace ModuleSccs
{
class CoulombOperator;
}
namespace unitcell
{
struct AtomData;
}
namespace elecstate
{
// Assemble an SCCS grid-charge operator using the existing cell geometry and
// mass-weighted ionic origin. Failure leaves the output unchanged.
bool make_sccs_pcc_0d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error);

// Use caller-owned Cartesian grid coordinates, e.g. from a fixed-source cache.
bool make_sccs_pcc_0d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              const std::vector<ModuleBase::Vector3<double>>& grid_positions,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error);
} // namespace elecstate

#endif
