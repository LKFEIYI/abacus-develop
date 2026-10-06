#include "sccs_pcc_0d.h"

#include "source_base/tool_quit.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_pcc_0d_coulomb.h"

namespace elecstate
{
void make_sccs_pcc_0d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb)
{
    std::vector<ModuleBase::Vector3<double>> positions;
    ModulePW::grid_positions(basis, cell.latvec, cell.lat0, positions);
    make_sccs_pcc_0d_operator(cell, basis, atoms, positions, coulomb);
}

void make_sccs_pcc_0d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               const std::vector<ModuleBase::Vector3<double>>& grid_positions,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb)
{
    unitcell::OrthogonalCell geometry;
    Pcc0dParameters parameters;
    const bool orthogonal = unitcell::make_orthogonal_cell(cell.latvec, cell.lat0, 1e-10, geometry);
    const bool cubic = orthogonal && make_pcc_0d_parameters(geometry, 1e-10, parameters);
    if (!cubic) { ModuleBase::WARNING_QUIT("make_sccs_pcc_0d_operator", "PCC 0D requires an equal-edge cubic cell"); }
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> masses;
    for (const unitcell::AtomData& atom : atoms)
    {
        positions.push_back(atom.position);
        masses.push_back(atom.mass);
    }
    geometry.origin = unitcell::weighted_center(positions, masses, geometry);
    std::vector<ModuleBase::Vector3<double>> relative_positions = grid_positions;
    for (auto& position : relative_positions)
    {
        position = unitcell::relative_position(position, geometry);
    }
    coulomb.reset(new ModuleSccs::Pcc0dCoulombOperator(basis, cell.tpiba, relative_positions, parameters));
}
} // namespace elecstate
