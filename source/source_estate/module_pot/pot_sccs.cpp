#include "pot_sccs.h"
#include "sccs_pcc_0d.h"
#include "sccs_pcc_2d.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_base/timer.h"
#include "source_base/tool_quit.h"
#include "source_cell/cell_tools.h"
#include "source_hamilt/module_sccs/sccs_functional.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/sccs_ionic_force.h"
#include "source_hamilt/module_sccs/sccs_response.h"
#include "source_hamilt/module_sccs/sccs_pw_coulomb.h"
#include "source_io/module_parameter/input_parameter.h"

#include <cmath>
#include <utility>
#include <ostream>

namespace
{
void require_valid_on_pool(bool valid, const std::string& error)
{
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_pool(invalid);
    if (invalid != 0.0)
    {
        const std::string message = error.empty() ? "Invalid SCCS input on another pool rank" : error;
        ModuleBase::WARNING_QUIT("PotSccs", message);
    }
}
}

namespace elecstate
{
bool make_sccs_config_from_input(const Input_para& input,
                                 ModuleSccs::SccsConfig& config,
                                 ModuleSccs::PolarizationSolverParameters& solver,
                                 std::string& error)
{
    if (input.sccs_debug < 0 || input.sccs_debug > 2)
    {
        error = "sccs_debug must be 0, 1, or 2";
        return false;
    }
    ModuleSccs::Preset preset;
    if (!ModuleSccs::parse_preset(input.sccs_preset, preset, error)) { return false; }
    ModuleSccs::SccsConfig candidate;
    if (preset == ModuleSccs::Preset::Custom)
    {
        candidate.cavity.density_min = input.sccs_rho_min;
        candidate.cavity.density_max = input.sccs_rho_max;
        candidate.cavity.epsilon_bulk = input.sccs_epsilon;
        candidate.surface_tension = ModuleSccs::dyn_per_cm_to_hartree_per_bohr2(input.sccs_gamma);
        candidate.pressure = ModuleSccs::gpa_to_hartree_per_bohr3(input.sccs_pressure);
    }
    else if (!ModuleSccs::make_sccs_config(preset, candidate, error)) { return false; }
    if (input.sccs_solvent_mode != "electronic" && input.sccs_solvent_mode != "full")
    {
        error = "sccs_solvent_mode must be electronic or full";
        return false;
    }
    candidate.start_drho = input.sccs_start_drho;
    candidate.start_nmax = input.sccs_start_nmax;
    const bool delayed = candidate.start_drho > 0.0;
    if (delayed && (candidate.start_drho <= input.scf_thr || candidate.start_nmax >= input.scf_nmax))
    {
        error = "Delayed SCCS requires sccs_start_drho > scf_thr and sccs_start_nmax < scf_nmax";
        return false;
    }
    candidate.core_electrons = input.sccs_solvent_mode == "full";
    candidate.core_spreads = input.sccs_corespread;
    candidate.surface_regularization = input.sccs_surface_eta;
    candidate.cavity.lowpass_p1 = input.sccs_lowpass_p1;
    candidate.cavity.lowpass_p2 = input.sccs_lowpass_p2;
    if (input.sccs_maxiter <= 0 || !std::isfinite(input.sccs_tol_rms) || input.sccs_tol_rms <= 0.0
        || !std::isfinite(input.sccs_tol_max) || input.sccs_tol_max <= 0.0)
    {
        error = "SCCS requires a positive iteration limit and finite positive residual tolerances";
        return false;
    }
    if (input.assume_isolated == "pcc_0d") { candidate.boundary = ModuleSccs::Boundary::Pcc0d; }
    else if (input.assume_isolated == "pcc_2d")
    {
        candidate.boundary = ModuleSccs::Boundary::Pcc2d;
        candidate.pcc_2d_axis = input.pcc_2d_axis;
    }
    else if (input.assume_isolated != "none")
    {
        error = "SCCS supports assume_isolated none, pcc_0d or pcc_2d";
        return false;
    }
    if (!ModuleSccs::validate_config(candidate, error)) { return false; }
    config = candidate;
    solver.check_fixed_point = input.sccs_debug >= 2;
    solver.max_iterations = input.sccs_maxiter;
    solver.tolerance_rms = input.sccs_tol_rms;
    solver.tolerance_max = input.sccs_tol_max;
    return true;
}

PotSccs::PotSccs(const ModulePW::PW_Basis* basis,
                 const ModuleSccs::SccsConfig& config,
                 const ModuleSccs::PolarizationSolverParameters& solver)
    : config_(config), solver_(solver)
{
    this->rho_basis_ = basis;
    this->dynamic_mode = true;
    sccs_active_ = config_.start_drho <= 0.0;
}

bool PotSccs::update_scf_state(int electronic_iteration, double density_residual)
{
    if (sccs_active_) { return false; }
    if (density_residual > config_.start_drho && electronic_iteration < config_.start_nmax)
    {
        return false;
    }
    sccs_active_ = true;
    restart_potential_.clear();
    return true;
}

void PotSccs::cal_v_eff(const Charge* charge, const UnitCell* cell, ModuleBase::matrix& potential)
{
    ModuleBase::timer::start("PotSccs", "cal_v_eff");
    const bool storage_valid = charge != nullptr && cell != nullptr && this->rho_basis_ != nullptr;
    require_valid_on_pool(storage_valid, "SCCS requires charge, cell and PW basis storage");
    const ModulePW::PW_Basis& basis = *this->rho_basis_;
    const bool grid_valid = (charge->nspin == 1 || charge->nspin == 2)
                            && potential.nr == charge->nspin && potential.nc == basis.nrxx
                            && charge->rho != nullptr && cell->atoms != nullptr && cell->ntype > 0
                            && std::isfinite(cell->lat0) && cell->lat0 > 0.0;
    require_valid_on_pool(grid_valid, "SCCS requires initialized atom/density storage and an nspin=1/2 potential");
    bool density_valid = true;
    for (int spin = 0; spin < charge->nspin; ++spin)
    {
        if (basis.nrxx > 0 && charge->rho[spin] == nullptr) { density_valid = false; }
    }
    require_valid_on_pool(density_valid, "SCCS charge density is not available");
    if (!sccs_active_)
    {
        electrostatic_rydberg_ = 0.0;
        non_electrostatic_rydberg_ = 0.0;
        electrostatic_potential_.assign(basis.nrxx, 0.0);
        cavity_potential_.assign(basis.nrxx, 0.0);
        ModuleBase::timer::end("PotSccs", "cal_v_eff");
        return;
    }
    const std::vector<unitcell::AtomData> atoms = unitcell::get_atom_data(cell->atoms, cell->ntype, cell->lat0);
    const int atom_count = atoms.size();
    const bool count_valid = atom_count == cell->nat;
    require_valid_on_pool(count_valid, "SCCS atom count does not match UnitCell");
    if (config_.core_electrons)
    {
        const bool widths_valid = config_.core_spreads.size() == 1 || config_.core_spreads.size() == atoms.size();
        require_valid_on_pool(widths_valid, "sccs_corespread must contain one value or exactly nat values");
    }
    std::vector<double> ions;
    std::string error;
    const bool ions_valid = ModuleSccs::gaussian_ionic_density(atoms, basis, cell->tpiba,
                                                              ModuleSccs::gaussian_ion_spread, ions, error);
    require_valid_on_pool(ions_valid, error);
    std::vector<double> density(basis.nrxx, 0.0);
    std::vector<double> solute_charge(basis.nrxx);
    double ionic_sum = 0.0;
    double net_charge = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        for (int spin = 0; spin < charge->nspin; ++spin) { density[ir] += charge->rho[spin][ir]; }
        solute_charge[ir] = ions[ir] - density[ir];
        ionic_sum += ions[ir];
        net_charge += solute_charge[ir];
    }
    Parallel_Reduce::reduce_pool(ionic_sum);
    Parallel_Reduce::reduce_pool(net_charge);
    const double dv = basis.omega / basis.nxyz;
    ionic_sum *= dv;
    net_charge *= dv;
    double expected_ionic_charge = 0.0;
    for (const unitcell::AtomData& atom : atoms) { expected_ionic_charge += atom.valence_charge; }
    const double normalization_error = ionic_sum - expected_ionic_charge;
    const bool normalization_valid = std::isfinite(normalization_error) && std::abs(normalization_error) < 1e-6;
    require_valid_on_pool(normalization_valid, "SCCS Gaussian ionic charge normalization failed");
    const bool neutral = std::isfinite(net_charge) && std::abs(net_charge) < 1e-6;
    if (config_.boundary == ModuleSccs::Boundary::Periodic)
    {
        require_valid_on_pool(neutral, "Periodic SCCS currently requires a neutral cell");
    }
    std::vector<ModuleBase::Vector3<double>> grid_positions;
    if (config_.boundary != ModuleSccs::Boundary::Periodic)
    {
        const bool positions_valid = ModulePW::grid_positions(basis, cell->latvec, cell->lat0, grid_positions, error);
        require_valid_on_pool(positions_valid, error);
    }
    std::unique_ptr<ModuleSccs::CoulombOperator> coulomb;
    if (config_.boundary == ModuleSccs::Boundary::Pcc0d)
    {
        const bool geometry_valid = make_sccs_pcc_0d_operator(*cell, basis, atoms, grid_positions, coulomb, error);
        require_valid_on_pool(geometry_valid, error);
    }
    else if (config_.boundary == ModuleSccs::Boundary::Pcc2d)
    {
        const bool geometry_valid = make_sccs_pcc_2d_operator(*cell, basis, atoms, config_.pcc_2d_axis,
                                                              grid_positions, coulomb, error);
        require_valid_on_pool(geometry_valid, error);
    }
    else
    {
        coulomb.reset(new ModuleSccs::PeriodicCoulombOperator(basis, cell->tpiba));
    }
    if (config_.core_electrons)
    {
        std::vector<double> core_density;
        const bool core_valid = ModuleSccs::gaussian_core_density(atoms, basis, cell->tpiba,
                                                                 config_.core_spreads, core_density, error);
        require_valid_on_pool(core_valid, error);
        for (int ir = 0; ir < basis.nrxx; ++ir) { density[ir] += core_density[ir]; }
    }
    ModuleSccs::SccsResponse response;
    const bool response_valid = ModuleSccs::solve_sccs_response(density, solute_charge, config_.cavity,
                                                               solver_, restart_potential_, basis, cell->tpiba,
                                                               *coulomb, response, error);
    require_valid_on_pool(response_valid, error);
    output_.transforms = coulomb->transform_counts();
    if (config_.boundary == ModuleSccs::Boundary::Pcc2d)
    {
        const double electron_count = ionic_sum - net_charge;
        const bool screening_valid = validate_sccs_pcc_2d_screening(response, config_, solver_, electron_count,
                                                                    ionic_sum, basis.omega, error);
        require_valid_on_pool(screening_valid, error);
    }
    ModuleSccs::FunctionalResult functional;
    const bool functional_valid = ModuleSccs::evaluate_functional(solute_charge, response, config_, basis,
                                                                  cell->tpiba, *coulomb, functional, error);
    require_valid_on_pool(functional_valid, error);
    output_.valid = true;
    output_.iterations = response.polarization.iterations;
    output_.warm_started = response.polarization.warm_started;
    output_.residual_rms = response.polarization.residual_rms;
    output_.residual_max = response.polarization.residual_max;
    output_.fixed_point_checked = response.polarization.fixed_point_checked;
    output_.fixed_point_defect_rms = response.polarization.fixed_point_defect_rms;
    output_.fixed_point_defect_max = response.polarization.fixed_point_defect_max;
    output_.reaction_energy = functional.reaction_energy;
    output_.volume = functional.volume;
    output_.surface = functional.surface;
    output_.far_field_charge = response.far_field_polarization_charge;
    if (solver_.check_fixed_point && config_.boundary != ModuleSccs::Boundary::Periodic)
    {
        const bool diagnostics_valid = collect_sccs_output(*cell, basis, atoms, grid_positions, ions, solute_charge,
                                                           config_, response, output_, error);
        require_valid_on_pool(diagnostics_valid, error);
    }
    electrostatic_rydberg_ = 2.0 * functional.reaction_energy;
    non_electrostatic_rydberg_ = 2.0 * (functional.surface_energy + functional.volume_energy);
    electrostatic_potential_.resize(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        electrostatic_potential_[ir] = -2.0 * functional.reaction_potential[ir];
        const double value = 2.0 * functional.electron_potential[ir];
        for (int spin = 0; spin < charge->nspin; ++spin) { potential(spin, ir) += value; }
    }
    cavity_potential_ = std::move(functional.cavity_potential);
    restart_potential_ = std::move(response.restart_potential);
    ModuleBase::timer::end("PotSccs", "cal_v_eff");
}

void PotSccs::add_solvation_force(const UnitCell& cell, ModuleBase::matrix& force) const
{
    ModuleBase::timer::start("PotSccs", "add_solvation_force");
    if (!sccs_active_)
    {
        ModuleBase::timer::end("PotSccs", "add_solvation_force");
        return;
    }
    const bool shape_valid = force.nr == cell.nat && force.nc == 3
                             && this->rho_basis_ != nullptr && cell.atoms != nullptr && cell.ntype > 0;
    require_valid_on_pool(shape_valid, "SCCS force requires initialized cell and atom-major storage");
    const ModulePW::PW_Basis& basis = *this->rho_basis_;
    const bool result_valid = electrostatic_potential_.size() == static_cast<std::size_t>(basis.nrxx);
    require_valid_on_pool(result_valid, "SCCS force requires a completed response on the current grid");
    const std::vector<unitcell::AtomData> atoms = unitcell::get_atom_data(cell.atoms, cell.ntype, cell.lat0);
    const bool count_valid = atoms.size() == static_cast<std::size_t>(cell.nat);
    require_valid_on_pool(count_valid, "SCCS force atom count does not match UnitCell");
    std::vector<double> reaction(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        reaction[ir] = -0.5 * electrostatic_potential_[ir];
    }
    std::vector<ModuleBase::Vector3<double>> ionic_force;
    std::string error;
    const bool valid = ModuleSccs::gaussian_ionic_force(atoms, reaction, basis, cell.tpiba,
                                                       ModuleSccs::gaussian_ion_spread, ionic_force, error);
    require_valid_on_pool(valid, error);
    if (config_.core_electrons)
    {
        std::vector<ModuleBase::Vector3<double>> core_force;
        const bool core_valid = ModuleSccs::gaussian_core_force(atoms, cavity_potential_, basis, cell.tpiba,
                                                               config_.core_spreads, core_force, error);
        require_valid_on_pool(core_valid, error);
        for (int ia = 0; ia < cell.nat; ++ia) { ionic_force[ia] += core_force[ia]; }
    }
    for (int ia = 0; ia < cell.nat; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            force(ia, axis) += 2.0 * ionic_force[ia][axis];
        }
    }
    ModuleBase::timer::end("PotSccs", "add_solvation_force");
}

int PotSccs::correction_output_priority() const
{
    if (!sccs_active_) { return 1; }
    return output_.valid ? 3 : 0;
}
void PotSccs::write_correction_iteration(std::ostream& output, int level, double residual, double pcc_energy) const
{
    if (level == 0) { return; }
    if (!sccs_active_)
    {
        output << " SCCS_DEFERRED DRHO " << residual << " START_DRHO " << config_.start_drho
               << " START_NMAX " << config_.start_nmax << '\n';
        return;
    }
    const double energy = get_energy();
    const bool pcc = config_.boundary != ModuleSccs::Boundary::Periodic;
    write_sccs_output(output, output_, level, energy, pcc, pcc_energy);
}
void PotSccs::write_correction_final(std::ostream& output, int level, double pcc_energy) const
{
    if (level < 2 || !sccs_active_) { return; }
    const bool slab = config_.boundary == ModuleSccs::Boundary::Pcc2d;
    write_sccs_final_output(output, output_, slab, pcc_energy);
}

double PotSccs::get_energy() const
{
    return electrostatic_rydberg_ + non_electrostatic_rydberg_;
}

void PotSccs::get_solvation_energy(double& electrostatic, double& non_electrostatic) const
{
    electrostatic = electrostatic_rydberg_;
    non_electrostatic = non_electrostatic_rydberg_;
}

const std::vector<double>* PotSccs::solvent_electrostatic_potential() const
{
    return &electrostatic_potential_;
}
} // namespace elecstate
