#ifndef SCCS_PCC_2D_ADAPTER_H
#define SCCS_PCC_2D_ADAPTER_H

#include <memory>
#include <vector>

class UnitCell;
namespace ModuleBase
{
template <typename T> class Vector3;
}
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
void make_sccs_pcc_2d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               int open_axis,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb);

// Use caller-owned Cartesian grid coordinates, e.g. from a fixed-source cache.
void make_sccs_pcc_2d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               int open_axis,
                               const std::vector<ModuleBase::Vector3<double>>& grid_positions,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb);
} // namespace elecstate

#endif
