#ifndef SCCS_PCC_COULOMB_H
#define SCCS_PCC_COULOMB_H

#include "sccs_charge.h"
#include "sccs_pcc.h"
#include "sccs_poisson.h"
#include "sccs_pw_coulomb.h"

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSccs
{

class PccCoulombOperator : public CoulombOperator
{
  public:
    PccCoulombOperator(const ModulePW::PW_Basis& basis,
                       double tpiba,
                       const std::vector<ModuleBase::Vector3<double>>& positions,
                       double volume_element,
                       const ModuleBase::Vector3<double>& origin,
                       const PccParameters& parameters,
                       const ChargeReduction& reduction);

    void apply(const std::vector<double>& charge, ElectrostaticField& field) const override;

  private:
    PeriodicCoulombOperator periodic_;
    const std::vector<ModuleBase::Vector3<double>>& positions_;
    double volume_element_ = 0.0;
    ModuleBase::Vector3<double> origin_;
    PccParameters parameters_;
    const ChargeReduction& reduction_;
};

} // namespace ModuleSccs

#endif
