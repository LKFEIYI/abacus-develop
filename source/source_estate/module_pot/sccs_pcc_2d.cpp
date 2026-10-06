#include "sccs_pcc_2d.h"

#include "source_base/tool_quit.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_pcc_2d_coulomb.h"

namespace elecstate
{
void make_sccs_pcc_2d_operator(const UnitCell& cell,
                               const ModulePW::PW_Basis& basis,
                               const std::vector<unitcell::AtomData>& atoms,
                               int open_axis,
                               std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb)
{
    unitcell::SlabCell geometry;
    const bool perpendicular = unitcell::make_slab_cell(cell.latvec, cell.lat0, open_axis, 1e-10, geometry);
    if (!perpendicular)
    {
        ModuleBase::WARNING_QUIT("make_sccs_pcc_2d_operator",
                                 "PCC 2D requires the open lattice vector to be perpendicular to the periodic plane");
    }
    const Pcc2dParameters parameters = make_pcc_2d_parameters(geometry);
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> masses;
    for (const unitcell::AtomData& atom : atoms)
    {
        positions.push_back(atom.position);
        masses.push_back(atom.mass);
    }
    geometry.origin = unitcell::weighted_center(positions, masses, geometry);
    std::vector<ModuleBase::Vector3<double>> relative_positions;
    ModulePW::grid_positions(basis, cell.latvec, cell.lat0, relative_positions);
    for (auto& position : relative_positions)
    {
        const double coordinate = unitcell::relative_coordinate(position, geometry);
        position = geometry.normal * coordinate;
    }
    coulomb.reset(new ModuleSccs::Pcc2dCoulombOperator(basis, cell.tpiba, relative_positions, geometry.normal,
                                                       parameters));
}
} // namespace elecstate
