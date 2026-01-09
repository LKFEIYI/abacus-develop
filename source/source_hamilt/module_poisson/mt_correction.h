#ifndef MT_CORRECTION_H
#define MT_CORRECTION_H

#include <complex>
#include <vector>

// 前置声明
namespace ModulePW { class PW_Basis; }
class UnitCell;

class MTCorrection
{
public:
    MTCorrection();
    ~MTCorrection();

    /**
     * @brief 执行 Martyna-Tuckerman 修正
     * * 物理原理：
     * 1. 构建总电荷密度 rho_tot(G) = rho_ion(G) - rho_elec(G)
     * 2. 计算筛选后的修正势 V_corr(G) = V_screen(G) * rho_tot(G)
     * 3. 计算修正能量 E_corr = 0.5 * sum( V_corr * rho_tot* )
     * 4. 将 V_corr(r) 叠加回实空间势场
     * * @param cell 晶胞信息
     * @param rho_basis 平面波基组
     * @param v_hartree [输入/输出] 实空间势场，修正值将累加到此
     * @param rho_elec 实空间电子电荷密度指针
     * @param nspin 自旋
     * @param dir 修正方向 (0=x, 1=y, 2=z)
     * @return double 修正能量 (Rydberg)
     */
    double apply_correction(const UnitCell& cell, 
                            ModulePW::PW_Basis* rho_basis, 
                            double* v_hartree, 
                            const double* const* rho_elec, 
                            int nspin,                     
                            int dir);

private:
    // 内部计算离子结构因子
    void cal_structure_factor(const UnitCell& cell, 
                              ModulePW::PW_Basis* rho_basis, 
                              std::vector<std::complex<double>>& ion_rho_g);
};

#endif // MT_CORRECTION_H