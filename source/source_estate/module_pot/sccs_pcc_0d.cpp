#include "sccs_pcc_0d.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_pcc_0d_coulomb.h"

namespace elecstate
{
bool make_sccs_pcc_0d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error)
{
    std::vector<ModuleBase::Vector3<double>> positions;
    const bool valid = ModulePW::grid_positions(basis, cell.latvec, cell.lat0, positions, error);
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        if (error.empty()) { error = "SCCS PCC grid positions are invalid on another pool rank"; }
        return false;
    }
    return make_sccs_pcc_0d_operator(cell, basis, atoms, positions, coulomb, error);
}
bool make_sccs_pcc_0d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              const std::vector<ModuleBase::Vector3<double>>& grid_positions,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error)
{
    unitcell::OrthogonalCell geometry;
    Pcc0dParameters parameters;
    bool valid = unitcell::make_orthogonal_cell(cell.latvec, cell.lat0, 1e-10, geometry, error);
    if (valid) { valid = make_pcc_0d_parameters(geometry, 1e-10, parameters, error); }
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> masses;
    for (const unitcell::AtomData& atom : atoms)
    {
        positions.push_back(atom.position);
        masses.push_back(atom.mass);
    }
    if (valid) { valid = unitcell::weighted_center(positions, masses, geometry, geometry.origin, error); }
    std::vector<ModuleBase::Vector3<double>> relative_positions = grid_positions;
    valid = valid && relative_positions.size() == static_cast<std::size_t>(basis.nrxx);
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        if (error.empty()) { error = "SCCS PCC 0D geometry is invalid on another pool rank"; }
        return false;
    }
    for (auto& position : relative_positions)
    {
        position = unitcell::relative_position(position, geometry);
    }
    coulomb.reset(new ModuleSccs::Pcc0dCoulombOperator(basis, cell.tpiba, relative_positions, parameters));
    return true;
}
} // namespace elecstate
