#ifndef SURCHEM_H
#define SURCHEM_H

#include "source_base/atom_in.h"
#include "source_base/global_function.h"
#include "source_base/global_variable.h"
#include "source_base/matrix.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_pw/module_pwdft/parallel_grid.h"
#include "source_pw/module_pwdft/structure_factor.h"

class surchem
{
  public:
    surchem();
    ~surchem();

    double* TOTN_real;
    double* delta_phi;
    double* epspot;
    ModuleBase::matrix Vcav;
    ModuleBase::matrix Vel;
    double qs;

    static double Acav;
    static double Ael;

    // get atom info
    atom_in GetAtom;

    // allocate memory and deallocate them
    void allocate(const int& nrxx, const int& nspin);

    void clear();

    void cal_epsilon(const ModulePW::PW_Basis* rho_basis, const double* PS_TOTN_real, double* epsilon, double* epsilon0);

    void cal_pseudo(const UnitCell& cell,
                    const Parallel_Grid& pgrid,
                    const ModulePW::PW_Basis* rho_basis,
                    const std::complex<double>* Porter_g,
                    std::complex<double>* PS_TOTN,
                    Structure_Factor* sf);

    void gauss_charge(const UnitCell& cell,
                      const Parallel_Grid& pgrid,
                      const ModulePW::PW_Basis* rho_basis,
                      std::complex<double>* N,
                      Structure_Factor* sf);

    void cal_totn(const UnitCell& cell,
                  const ModulePW::PW_Basis* rho_basis,
                  const std::complex<double>* Porter_g,
                  std::complex<double>* N,
                  std::complex<double>* TOTN,
                  const double* vlocal);

    void createcavity(const UnitCell& ucell,
                      const ModulePW::PW_Basis* rho_basis,
                      const std::complex<double>* PS_TOTN,
                      double* vwork);

    ModuleBase::matrix cal_vcav(const UnitCell& ucell,
                                const ModulePW::PW_Basis* rho_basis,
                                std::complex<double>* PS_TOTN,
                                int nspin);

    ModuleBase::matrix cal_vel(const UnitCell& cell,
                               const ModulePW::PW_Basis* rho_basis,
                               std::complex<double>* TOTN,
                               std::complex<double>* PS_TOTN,
                               int nspin);

    double cal_Ael(const UnitCell& cell,
                   const int& nrxx,  // num. of real space grids on current core
                   const int& nxyz); // total num. of real space grids

    double cal_Acav(const UnitCell& cell,
                    const int& nxyz); // total num. of real space grids

    void cal_Acomp(const UnitCell& cell,
                   const ModulePW::PW_Basis* rho_basis,
                   const double* const* const rho,
                   std::vector<double>& res);

    void minimize_cg(const UnitCell& ucell,
                     const ModulePW::PW_Basis* rho_basis,
                     double* d_eps,
                     const double* kappa2_factor, // NEW: 4pi * d(rho_ion)/d(phi)
                     const std::complex<double>* tot_N,
                     std::complex<double>* phi,
                     int& ncgsol);

    void Leps2(const UnitCell& ucell,
               const ModulePW::PW_Basis* rho_basis,
               std::complex<double>* phi,
               double* epsilon,
               const double* kappa2_factor, // NEW
               std::complex<double>* gradphi_x,
               std::complex<double>* gradphi_y,
               std::complex<double>* gradphi_z,
               std::complex<double>* phi_work,
               std::complex<double>* lp);
               
      void test_smpbe_driver(const UnitCell& cell, const ModulePW::PW_Basis* rho_basis);

// 新增：VASPsol++ 物理量计算辅助函数
    void cal_smpbe_physics(const int nrxx,
                           const double* phi_R,
                           const double* eps_R,
                           double* rho_ion_out,
                           double* kappa2_factor_out);

    ModuleBase::matrix v_correction(const UnitCell& cell,
                                    const Parallel_Grid& pgrid,
                                    const ModulePW::PW_Basis* rho_basis,
                                    const int& nspin,
                                    const double* const* const rho,
                                    const double* vlocal,
                                    Structure_Factor* sf);

    void test_V_to_N(ModuleBase::matrix& v,
                     const UnitCell& cell,
                     const ModulePW::PW_Basis* rho_basis,
                     const double* const* const rho);

    void cal_force_sol(const UnitCell& cell,
                       const ModulePW::PW_Basis* rho_basis,
                       const ModuleBase::matrix& vloc,
                       ModuleBase::matrix& forcesol);

    void force_cor_one(const UnitCell& cell,
                       const ModulePW::PW_Basis* rho_basis,
                       const ModuleBase::matrix& vloc,
                       ModuleBase::matrix& forcesol);

    void force_cor_two(const UnitCell& cell, const ModulePW::PW_Basis* rho_basis, ModuleBase::matrix& forcesol);

    void get_totn_reci(const UnitCell& cell, const ModulePW::PW_Basis* rho_basis, std::complex<double>* totn_reci);

    void induced_charge(const UnitCell& cell, const ModulePW::PW_Basis* rho_basis, double* induced_rho) const;

  private:
};

#endif
