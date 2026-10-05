#ifndef POTBASE_H
#define POTBASE_H

#include <iosfwd>

#include "source_base/complexmatrix.h"
#include "source_base/matrix.h"
#include "source_cell/unitcell.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_estate/module_charge/charge.h"

namespace elecstate
{
/** This class is the base class of Potential module
 1. Main class Potential is derived from it.
 2. components of potentials on real space grids can derived from it and will be registered into Potential.
    a. cal_fixed_v() is a virtual function, it can be override to contribute potentials which do not change with Charge object.
    b. cal_v_eff() is a virtual function, it can be override to contribute potentials which change with Charge object.
    c. fixed_mode should be set "true" if you want Potential class call cal_fixed_v()
    d. dynamic_mode should be set "true" if you want Potential class call cal_v_eff()
    e. rho_basis_ is needed to provide number of real space grids(nrxx) and number of spin(nspin) and FFT(real<->recip) interface
*/
class PotBase
{
  public:
    PotBase(){}
    virtual ~PotBase(){}

    virtual void cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff){}

    virtual void cal_fixed_v(double* vl_pseudo){}

    virtual double get_energy() const { return 0.0; }

    // Called with the synchronized output-density residual before SCF mixing.
    // True means the potential changed its model: restart mixing and require
    // another electronic iteration before accepting convergence.
    virtual bool update_scf_state(int electronic_iteration, double density_residual) { return false; }

    // Preserve component-owned SCF workflow state when ionic steps rebuild
    // the Hamiltonian. Numerical results and grid storage are not inherited.
    virtual void inherit_scf_state(const PotBase& previous) {}

    // Prefer active solvation (3), otherwise PCC (2), then deferred solvation (1).
    virtual int correction_output_priority() const { return 0; }
    virtual void write_correction_iteration(std::ostream&, int, double, double) const {}
    virtual void write_correction_final(std::ostream&, int, double) const {}

    // Solvation contributions in Ry, added once after the band-energy potential subtraction.
    virtual void get_solvation_energy(double& electrostatic, double& non_electrostatic) const
    {
        electrostatic = 0.0;
        non_electrostatic = 0.0;
    }
    // Explicit ionic solvation force in Ry/Bohr, added to existing contributions.
    virtual void add_solvation_force(const UnitCell&, ModuleBase::matrix&) const {}
    // Electronic electrostatic correction only; excludes cavity/surface derivatives.
    virtual const std::vector<double>* solvent_electrostatic_potential() const { return nullptr; }
    
    bool fixed_mode = 0;
    bool dynamic_mode = 0;

  protected:
    const ModulePW::PW_Basis* rho_basis_ = nullptr;
    const ModulePW::PW_Basis* rho_basis_smooth_ = nullptr;
};

} // namespace elecstate

#endif