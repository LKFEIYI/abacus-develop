#include "sccs_pcc_2d.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_pcc_2d_coulomb.h"
#include "source_hamilt/module_sccs/sccs_parameters.h"
#include "source_hamilt/module_sccs/sccs_response.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace elecstate
{
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
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
    return make_sccs_pcc_2d_operator(cell, basis, atoms, open_axis, positions, coulomb, error);
}
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
                              const std::vector<ModuleBase::Vector3<double>>& grid_positions,
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
    std::vector<ModuleBase::Vector3<double>> relative_positions = grid_positions;
    valid = valid && relative_positions.size() == static_cast<std::size_t>(basis.nrxx);
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
bool validate_sccs_pcc_2d_screening(const ModuleSccs::SccsResponse& response,
                                   const ModuleSccs::SccsConfig& config,
                                   const ModuleSccs::PolarizationSolverParameters& solver,
                                   double electron_count,
                                   double ionic_charge,
                                   double cell_volume,
                                   std::string& warning,
                                   std::string& error)
{
    warning.clear();
    error.clear();
    const double solute_charge = ionic_charge - electron_count;
    const double expected = (1.0 / config.cavity.epsilon_bulk - 1.0) * solute_charge;
    const double system_charge = std::max(electron_count, ionic_charge);
    const double charge_scale = std::max(1.0, system_charge);
    const double relative_tolerance = 1e-4 * charge_scale;
    const double grid_tolerance = solver.tolerance_max * cell_volume;
    const double solver_tolerance = std::max(grid_tolerance, relative_tolerance);
    const double tolerance = std::max(1e-6, solver_tolerance);
    const double actual = response.far_field_polarization_charge;
    const double screening_error = actual - expected;
    if (!std::isfinite(expected) || !std::isfinite(actual) || !std::isfinite(tolerance))
    {
        error = "SCCS PCC 2D far-field screening charge or tolerance is not finite";
        return false;
    }
    if (std::abs(screening_error) > tolerance)
    {
        std::ostringstream message;
        message << "SCCS pcc_2d far-field polarization charge " << actual
                << " differs from expected " << expected << " by more than tolerance " << tolerance;
        message << ". Check convergence with respect to the charge-density cutoff.";
        warning = message.str();
    }
    return true;
}
} // namespace elecstate
