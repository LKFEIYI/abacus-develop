#include "sccs_pcc_2d.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_pcc_2d_coulomb.h"

namespace elecstate
{
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error)
{
    unitcell::SlabCell geometry;
    Pcc2dParameters parameters;
    bool valid = unitcell::make_slab_cell(cell.latvec, cell.lat0, open_axis, 1e-10, geometry, error);
    if (valid) { valid = make_pcc_2d_parameters(geometry, parameters, error); }
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> masses;
    for (const unitcell::AtomData& atom : atoms)
    {
        positions.push_back(atom.position);
        masses.push_back(atom.mass);
    }
    if (valid) { valid = unitcell::weighted_center(positions, masses, geometry, geometry.origin, error); }
    std::vector<ModuleBase::Vector3<double>> relative_positions;
    if (valid) { valid = ModulePW::grid_positions(basis, cell.latvec, cell.lat0, relative_positions, error); }
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        if (error.empty()) { error = "SCCS PCC 2D geometry is invalid on another pool rank"; }
        return false;
    }
    for (auto& position : relative_positions)
    {
        const double coordinate = unitcell::relative_coordinate(position, geometry);
        position = geometry.normal * coordinate;
    }
    coulomb.reset(new ModuleSccs::Pcc2dCoulombOperator(basis, cell.tpiba, relative_positions, geometry.normal, parameters));
    return true;
}
} // namespace elecstate
