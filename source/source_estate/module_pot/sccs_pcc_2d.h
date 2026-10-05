#ifndef SCCS_PCC_2D_ADAPTER_H
#define SCCS_PCC_2D_ADAPTER_H

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
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error);
} // namespace elecstate

#endif
