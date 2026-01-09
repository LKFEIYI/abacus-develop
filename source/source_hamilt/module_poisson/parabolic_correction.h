#ifndef PARABOLIC_CORRECTION_H
#define PARABOLIC_CORRECTION_H

#include "source_base/matrix.h"
#include <vector>
#include "source_cell/unitcell.h"

class UnitCell;
namespace ModulePW { class PW_Basis; }

class ParabolicCorrection
{
public:
    ParabolicCorrection();
    ~ParabolicCorrection();

    // 修改返回值 void -> double，用于返回离子修正能
    double apply_correction(const UnitCell& cell, 
                            const ModulePW::PW_Basis* rho_basis, 
                            double* v_hartree, 
                            const double* const* rho_elec, // 修正类型
                            int nspin,                     // 新增参数
                            //double nelec,
                            int dir);

    // 力修正需要单独调用（在力计算模块）
    void calc_force_correction(const UnitCell& cell, 
                               ModuleBase::matrix& force, 
                               int dir,
                               double net_charge,
                               double total_dipole,
                               double vacuum_center,
                               double area);

    // 公开这个辅助函数，方便外部获取参数传给 calc_force_correction
    double find_vacuum_center(const UnitCell& cell, int dir);
    double calc_net_charge(const UnitCell& cell, double nelec);
    
    // 获取偶极矩的接口，方便力修正使用
    double get_last_total_dipole() const { return last_total_dipole_; }
    double calc_total_quadrupole(const UnitCell& cell, 
                                 const ModulePW::PW_Basis* rho_basis,
                                 const double* const* rho_elec, 
                                 int nspin,
                                 int dir, 
                                 double center);

private:
    double calc_total_dipole(const UnitCell& cell, 
                             const ModulePW::PW_Basis* rho_basis,
                             const double* const* rho_elec, 
                             int nspin,
                             int dir,
                             double center);

    double calc_ion_dipole(const UnitCell& cell, int dir, double center);

    double calc_elec_dipole(const ModulePW::PW_Basis* rho_basis, 
                            const double* const* rho_elec, 
                            int nspin,
                            int dir, 
                            double center,
                            double omega);

    double calc_energy_correction(const UnitCell& cell, 
                                  int dir,
                                  double net_charge,
                                  double total_dipole,
                                  double vacuum_center,
                                  double v_const);

    // 缓存一些中间变量供力修正使用，避免重复计算
    double last_total_dipole_ = 0.0;
    double calc_ion_quadrupole(const UnitCell& cell, int dir, double center);
    double calc_elec_quadrupole(const ModulePW::PW_Basis* rho_basis, 
                                const double* const* rho_elec, 
                                int nspin, 
                                int dir, 
                                double center, 
                                double omega);
};

#endif