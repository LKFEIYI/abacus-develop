#include "mt_correction.h"
#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_io/module_parameter/parameter.h"
#include "source_hamilt/module_poisson/mt_poisson.h" // 你的筛选函数

#include <cmath>
#include <iostream>
#include <algorithm>

MTCorrection::MTCorrection() {}
MTCorrection::~MTCorrection() {}

double MTCorrection::apply_correction(const UnitCell& cell, 
                                      ModulePW::PW_Basis* rho_basis, 
                                      double* v_hartree, 
                                      const double* const* rho_elec, 
                                      int nspin,                     
                                      int dir)
{
    // =============================================================
    // 1. 准备电子密度 rho_elec(G)
    // =============================================================
    int ng = rho_basis->npw;
    int nrxx = rho_basis->nrxx;
    
    // 使用 vector 管理内存，避免手动 new/delete
    std::vector<std::complex<double>> rho_tot_g(rho_basis->nmaxgr, {0.0, 0.0});
    
    int nspin_eff = (nspin == 2) ? 2 : 1;
    
    // 将实空间电子密度 rho(r) 放入 buffer
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int ir = 0; ir < nrxx; ir++) {
        double val = 0.0;
        for (int is = 0; is < nspin_eff; is++) {
            val += rho_elec[is][ir];
        }
        rho_tot_g[ir] = std::complex<double>(val, 0.0);
    }

    // FFT: R -> G
    rho_basis->real2recip(rho_tot_g.data(), rho_tot_g.data());

    // =============================================================
    // 2. 计算离子结构因子 rho_ion(G)
    // =============================================================
    std::vector<std::complex<double>> rho_ion_g(ng);
    this->cal_structure_factor(cell, rho_basis, rho_ion_g);

    // =============================================================
    // 3. 计算修正势 V_corr(G) 和 修正能量 E_corr
    // =============================================================
    std::vector<std::complex<double>> v_corr_g(ng);
    double e_corr = 0.0;
    
    // 几何参数
    double L = 0.0;
    if (dir == 0) L = cell.a1.norm() * cell.lat0;
    else if (dir == 1) L = cell.a2.norm() * cell.lat0;
    else if (dir == 2) L = cell.a3.norm() * cell.lat0;

    double tpiba = cell.tpiba;
    double tpiba2 = cell.tpiba2;
    int ig0 = rho_basis->ig_gge0;

    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:e_corr)
    #endif
    for (int ig = 0; ig < ng; ig++)
    {
        if (ig == ig0) {
            v_corr_g[ig] = {0.0, 0.0};
            // 关键：G=0 处设为 0，防止常数偏移
            continue;
        }

        // A. 构建总电荷密度 rho_tot(G) = rho_ion(G) - rho_elec(G)
        // 电子带负电，所以总电荷 = Z_ion - N_elec
        std::complex<double> total_charge_g = rho_ion_g[ig] - rho_tot_g[ig];

        // B. 获取筛选函数 V_screen(G)
        // 这是 MT 修正的核心：V_screen 仅包含要去处的镜像作用项
        // 单位通常对应 4pi/G^2 的量级
        double g2 = tpiba2 * rho_basis->gg[ig];
        ModuleBase::Vector3<double> g_vec = rho_basis->gcar[ig] * tpiba;
        
        // 调用你的 MTPoisson 模块
        // 注意：get_screen_val_g0 已经在 mt_poisson.cpp 里设为 0 了，这里再保险一次
        double screen_val = MTPoisson::get_screen_val(g2, g_vec, L, 0.0, "2D", dir);

        // C. 计算修正势 V_corr(G) = Screen(G) * rho_tot(G)
        // 乘以 e2 (Hartree -> Rydberg)
        double kernel = ModuleBase::e2 * screen_val; 
        
        v_corr_g[ig] = kernel * total_charge_g;

        // D. 计算修正能量 E = 0.5 * sum( V_corr(G) * conj(rho_tot(G)) )
        // 这包含了 (ion-ion) + (elec-elec) + 2*(ion-elec) 所有交叉项的修正
        double rho_mod2 = std::norm(total_charge_g); // |rho_tot|^2
        e_corr += 0.5 * kernel * rho_mod2;
    }

    // 规约能量 (MPI reduce) 并乘以体积因子
    Parallel_Reduce::reduce_pool(e_corr);
    e_corr *= cell.omega; // 归一化

    // =============================================================
    // 4. 将修正势变换回实空间 V_corr(r) 并叠加
    // =============================================================
    // 使用 rho_tot_g 作为 buffer 进行 IFFT (G -> R)
    // 结果现在代表 V_corr(r)
    rho_basis->recip2real(v_corr_g.data(), rho_tot_g.data()); 

    // 叠加到输出势场 v_hartree
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int ir = 0; ir < nrxx; ir++) {
        v_hartree[ir] += rho_tot_g[ir].real();
    }

    return e_corr;
}

// 辅助函数：计算离子结构因子 (直接复用 Structure_Factor 的逻辑)
void MTCorrection::cal_structure_factor(const UnitCell& cell, 
                                        ModulePW::PW_Basis* rho_basis, 
                                        std::vector<std::complex<double>>& ion_rho_g)
{
    std::fill(ion_rho_g.begin(), ion_rho_g.end(), std::complex<double>(0.0, 0.0));

    // 虚数单位 * 2pi (用于指数计算，参考 structure_factor.cpp)
    const std::complex<double> ci_tpi = ModuleBase::NEG_IMAG_UNIT * ModuleBase::TWO_PI;

    int ig0 = rho_basis->ig_gge0;

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if (ig == ig0) continue; 

        std::complex<double> sum_struc_fac = {0.0, 0.0};
        
        // 获取当前 G 向量 (Cartesian, internal units)
        const ModuleBase::Vector3<double>& gvec = rho_basis->gcar[ig];

        // 遍历所有原子类型
        for (int it = 0; it < cell.ntype; it++)
        {
            double zv = cell.atoms[it].ncpp.zv; // 离子电荷
            const int na = cell.atoms[it].na;
            const ModuleBase::Vector3<double>* tau = cell.atoms[it].tau.data();

            std::complex<double> s_type = {0.0, 0.0};

            // 计算该类型原子的结构因子: sum(exp(-i * 2pi * G * tau))
            // 这个公式与 structure_factor.cpp 中完全一致
            for(int ia = 0; ia < na; ++ia) {
                // gvec * tau[ia] 是点积
                s_type += exp( ci_tpi * (gvec * tau[ia]) );
            }
            
            // 累加到总离子电荷密度: Z * S(G)
            sum_struc_fac += zv * s_type;
        }
        
        // 注意：泊松方程 rho_ion(G) = (1/Omega) * sum(...)
        // 但这里我们只存储 sum(...) 部分，系数 1/Omega 隐含在最后的能量积分公式中
        // 或者在这个阶段就除以 Omega? 
        // 按照 H_Ewald_pw 的逻辑，ewaldg 最终乘了 (4pi/omega)，说明中间变量不需要除。
        // 我们上面的 apply_correction 逻辑中，e_corr *= cell.omega，这和 Ewald 是一致的。
        ion_rho_g[ig] = sum_struc_fac;
    }
}