#include "potential_new.h"
#include "pot_pcc.h"
#include "pot_sccs.h"

#include <ostream>

namespace elecstate
{
bool Potential::update_model(int electronic_iteration, double density_residual, std::ostream* log)
{
    PotSccs* sccs = this->sccs_component();
    if (sccs == nullptr)
    {
        return false;
    }
    const bool activated = sccs->try_activate(electronic_iteration, density_residual);
    if (activated && log != nullptr)
    {
        *log << " SCCS activated at electronic iteration " << electronic_iteration
             << ", DRHO = " << density_residual << std::endl;
    }
    return activated;
}

bool Potential::model_pending() const
{
    const PotSccs* sccs = this->sccs_component();
    return sccs != nullptr && !sccs->is_active();
}

void Potential::write_corr_iter(std::ostream& output, int level, double residual) const
{
    if (level == 0)
    {
        return;
    }
    const PotSccs* sccs = this->sccs_component();
    const PotPcc* pcc = this->pcc_component();
    const double pcc_energy = this->pcc_energy_rydberg();
    // Active SCCS reports the coupled PCC energy too; a deferred SCCS reports
    // its start condition only when no PCC result is available.
    const bool sccs_active = sccs != nullptr && sccs->is_active();
    const bool sccs_reports = sccs_active && sccs->has_output();
    const bool pcc_reports = pcc != nullptr && pcc->has_result();
    if (sccs_reports)
    {
        sccs->write_iter(output, level, residual, pcc_energy);
    }
    else if (pcc_reports)
    {
        pcc->write_iter(output, level);
    }
    else if (sccs != nullptr && !sccs_active)
    {
        sccs->write_iter(output, level, residual, pcc_energy);
    }
}

void Potential::write_corr_fin(std::ostream& output, int level) const
{
    if (level < 2)
    {
        return;
    }
    const PotSccs* sccs = this->sccs_component();
    if (sccs == nullptr)
    {
        return;
    }
    const double pcc_energy = this->pcc_energy_rydberg();
    sccs->write_final(output, pcc_energy);
}

const PotSccs* Potential::sccs_component() const
{
    for (const PotBase* component : this->components)
    {
        const PotSccs* sccs = dynamic_cast<const PotSccs*>(component);
        if (sccs != nullptr)
        {
            return sccs;
        }
    }
    return nullptr;
}

PotSccs* Potential::sccs_component()
{
    for (PotBase* component : this->components)
    {
        PotSccs* sccs = dynamic_cast<PotSccs*>(component);
        if (sccs != nullptr)
        {
            return sccs;
        }
    }
    return nullptr;
}

SccsResume Potential::get_resume() const
{
    const PotSccs* sccs = this->sccs_component();
    if (sccs == nullptr)
    {
        return SccsResume();
    }
    return sccs->resume_state();
}
} // namespace elecstate
