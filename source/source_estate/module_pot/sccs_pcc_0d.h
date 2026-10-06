#ifndef SCCS_PCC_0D_ADAPTER_H
#define SCCS_PCC_0D_ADAPTER_H

#include <memory>
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
// mass-weighted ionic origin. An unsupported cell stops the run.
void make_sccs_pcc_0d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb);
} // namespace elecstate

#endif
