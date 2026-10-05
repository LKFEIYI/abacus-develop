#ifndef POT_SCCS_H
#define POT_SCCS_H

#include "pot_base.h"
#include "sccs_output.h"
#include "sccs_fixed_sources.h"
#include "source_hamilt/module_sccs/sccs_parameters.h"

struct Input_para;
namespace elecstate
{
bool make_sccs_config_from_input(const Input_para& input,
                                 ModuleSccs::SccsConfig& config,
                                 ModuleSccs::PolarizationSolverParameters& solver,
                                 std::string& error);

class PotSccs : public PotBase
{
public:
    PotSccs(const ModulePW::PW_Basis* basis,
            const ModuleSccs::SccsConfig& config,
            const ModuleSccs::PolarizationSolverParameters& solver);
    void cal_v_eff(const Charge* charge, const UnitCell* cell, ModuleBase::matrix& potential) override;
    double get_energy() const override;
    bool update_scf_state(int electronic_iteration, double density_residual) override;
    int correction_output_priority() const override;
    void write_correction_iteration(std::ostream& output, int level, double residual, double pcc_energy) const override;
    void write_correction_final(std::ostream& output, int level, double pcc_energy) const override;
    bool sccs_is_active() const { return sccs_active_; }
    void add_solvation_force(const UnitCell& cell, ModuleBase::matrix& force) const override;
    void get_solvation_energy(double& electrostatic, double& non_electrostatic) const override;
    const std::vector<double>* solvent_electrostatic_potential() const override;

private:
    // Initialized in the constructor; update_scf_state is the sole transition
    // owner. There is no external setter and activation is irreversible.
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
