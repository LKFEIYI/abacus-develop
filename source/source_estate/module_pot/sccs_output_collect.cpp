#include "sccs_output.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/cell_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_estate/pcc_0d.h"
#include "source_estate/pcc_2d.h"
#include "source_hamilt/module_sccs/sccs_diagnostics.h"
#include "source_hamilt/module_sccs/sccs_parameters.h"
#include "source_hamilt/module_sccs/sccs_response.h"

namespace
{
bool pool_valid(bool valid, int pool_size, std::string& error)
{
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(pool_size, invalid);
    if (invalid == 0.0) { return true; }
    if (error.empty()) { error = "SCCS diagnostic geometry or moments invalid on another pool rank"; }
    return false;
}
void reduce_moments(elecstate::ChargeMoments& moments)
{
    Parallel_Reduce::reduce_pool(moments.charge);
    Parallel_Reduce::reduce_pool(moments.dipole.x);
    Parallel_Reduce::reduce_pool(moments.dipole.y);
    Parallel_Reduce::reduce_pool(moments.dipole.z);
    Parallel_Reduce::reduce_pool(moments.second_moment);
}
}
namespace elecstate
{
bool collect_sccs_output(const UnitCell& cell,
                          const ModulePW::PW_Basis& basis,
                          const std::vector<unitcell::AtomData>& atoms,
                          const std::vector<double>& ions,
                          const std::vector<double>& charge,
                          const ModuleSccs::SccsConfig& config,
                          const ModuleSccs::SccsResponse& response,
                          SccsOutput& result,
                          std::string& error)
{
    std::vector<double> polarization;
    if (!ModuleSccs::continuum_polarization_charge(charge, response, basis, cell.tpiba, polarization, error))
    {
        return false;
    }
    PccOutput candidate;
    candidate.slab = config.boundary == ModuleSccs::Boundary::Pcc2d;
    candidate.axis = config.pcc_2d_axis;
    unitcell::OrthogonalCell geometry;
    unitcell::SlabCell slab;
    Pcc0dParameters parameters;
    Pcc2dParameters slab_parameters;
    std::vector<ModuleBase::Vector3<double>> positions;
    std::vector<double> masses;
    std::vector<double> ionic_charges;
    for (const auto& atom : atoms)
    {
        positions.push_back(atom.position);
        masses.push_back(atom.mass);
        ionic_charges.push_back(atom.valence_charge);
    }
    bool valid = ions.size() == charge.size();
    if (candidate.slab)
    {
        valid = valid && unitcell::make_slab_cell(cell.latvec, cell.lat0, candidate.axis, 1e-10, slab, error);
        valid = valid && make_pcc_2d_parameters(slab, slab_parameters, error);
        valid = valid && unitcell::weighted_center(positions, masses, slab, slab.origin, error);
        candidate.coordinate = slab.origin;
        candidate.normal = slab.normal;
    }
    else
    {
        valid = valid && unitcell::make_orthogonal_cell(cell.latvec, cell.lat0, 1e-10, geometry, error);
        valid = valid && make_pcc_0d_parameters(geometry, 1e-10, parameters, error);
        valid = valid && unitcell::weighted_center(positions, masses, geometry, geometry.origin, error);
        candidate.origin = geometry.origin;
    }
    std::vector<ModuleBase::Vector3<double>> grid;
    valid = valid && ModulePW::grid_positions(basis, cell.latvec, cell.lat0, grid, error);
    if (!pool_valid(valid, basis.poolnproc, error)) { return false; }
    for (auto& position : grid)
    {
        if (candidate.slab)
        {
            const double coordinate = unitcell::relative_coordinate(position, slab);
            position = slab.normal * coordinate;
        }
        else { position = unitcell::relative_position(position, geometry); }
    }
    for (auto& position : positions)
    {
        if (candidate.slab)
        {
            const double coordinate = unitcell::relative_coordinate(position, slab);
            position = slab.normal * coordinate;
        }
        else { position = unitcell::relative_position(position, geometry); }
    }
    const double dv = basis.omega / basis.nxyz;
    std::vector<double> electrons(charge.size());
    for (std::size_t i = 0; i < charge.size(); ++i) { electrons[i] = charge[i] - ions[i]; }
    valid = charge_moments(charge.data(), grid.data(), basis.nrxx, dv, candidate.moments[0], error);
    valid = valid && charge_moments(electrons.data(), grid.data(), basis.nrxx, dv, candidate.moments[1], error);
    valid = valid && charge_moments(polarization.data(), grid.data(), basis.nrxx, dv, candidate.moments[2], error);
    ChargeMoments point_ions;
    const int atom_count = atoms.size();
    valid = valid && charge_moments(ionic_charges.data(), positions.data(), atom_count, 1.0, point_ions, error);
    if (!pool_valid(valid, basis.poolnproc, error)) { return false; }
    for (int i = 0; i < 3; ++i) { reduce_moments(candidate.moments[i]); }
    candidate.moments[1] = add_charge_moments(candidate.moments[1], point_ions);
    candidate.moments[3] = add_charge_moments(candidate.moments[0], candidate.moments[2]);
    if (candidate.slab)
    {
        candidate.smooth_energy = pcc_2d_energy(candidate.moments[0], slab_parameters);
        candidate.point_energy = pcc_2d_energy(candidate.moments[1], slab_parameters);
    }
    else
    {
        candidate.smooth_energy = pcc_0d_energy(candidate.moments[0], parameters);
        candidate.point_energy = pcc_0d_energy(candidate.moments[1], parameters);
    }
    result.pcc = candidate;
    result.expected_charge = -(1.0 - 1.0 / config.cavity.epsilon_bulk) * candidate.moments[0].charge;
    return true;
}
} // namespace elecstate
