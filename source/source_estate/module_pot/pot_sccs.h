#ifndef POT_SCCS_H
#define POT_SCCS_H

#include "pot_base.h"
#include "sccs_output.h"
#include "sccs_fixed_sources.h"
#include "source_hamilt/module_sccs/sccs_parameters.h"

#include <iosfwd>
#include <string>

struct Input_para;
namespace elecstate
{
// Map INPUT values, already validated by the INPUT reader, onto the solver.
void make_sccs_config_from_input(const Input_para& input,
                                 ModuleSccs::SccsConfig& config,
                                 ModuleSccs::PolarizationSolverParameters& solver);

// Structure-dependent checks that the INPUT reader cannot make: a
// sccs_corespread count other than one or nat stops the run. Returns a warning
// (empty if none) when a single width meets an unknown pseudopotential
// element, whose automatic core exclusion is then not possible.
std::string check_sccs_structure(const ModuleSccs::SccsConfig& config, const UnitCell& cell);

class PotSccs : public PotBase
{
public:
    // resume_active keeps an activation reached in an earlier ionic step.
    PotSccs(const ModulePW::PW_Basis* basis,
            const ModuleSccs::SccsConfig& config,
            const ModuleSccs::PolarizationSolverParameters& solver,
            bool resume_active);
    void cal_v_eff(const Charge* charge, const UnitCell* cell, ModuleBase::matrix& potential) override;
    double get_energy() const override;
    void add_solvation_force(const UnitCell& cell, ModuleBase::matrix& force) const override;
    void get_solvation_energy(double& electrostatic, double& non_electrostatic) const override;
    const std::vector<double>* solvent_electrostatic_potential() const override;

    // Delayed start: called once per electronic iteration with the
    // pool-consistent density residual. Returns true only in the iteration
    // that activates SCCS; activation is irreversible.
    bool update_activation(int electronic_iteration, double density_residual);
    bool is_active() const { return sccs_active_; }
    bool has_output() const { return output_.valid; }
    void write_iteration_output(std::ostream& output, int level, double residual, double pcc_energy) const;
    void write_final_output(std::ostream& output, double pcc_energy) const;

private:
    bool sccs_active_ = false;
    const ModuleSccs::SccsConfig config_;
    const ModuleSccs::PolarizationSolverParameters solver_;
    SccsOutput output_;
    SccsFixedSources fixed_sources_;
    double electrostatic_rydberg_ = 0.0;
    double non_electrostatic_rydberg_ = 0.0;
    std::vector<double> electrostatic_potential_;
    std::vector<double> restart_potential_;
    std::vector<double> cavity_potential_;
};
} // namespace elecstate
#endif
