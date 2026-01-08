#include "parabolic_correction.h"
#include "source_base/constants.h" // ModuleBase::e2, PI, FOUR_PI
#include "source_base/parallel_reduce.h" // Parallel_Reduce
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

#include <cmath>
#include <iostream>
#include <algorithm> // for std::sort

// ---------------------------------------------------------
// 1. 自动寻找真空层中心 (逻辑完全借鉴 Efield::autoset)
// ---------------------------------------------------------
double ParabolicCorrection::find_vacuum_center(const UnitCell& cell, int dir)
{
    std::vector<double> pos;
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia)
        {
            double p = cell.atoms[it].taud[ia][dir];
            p -= std::floor(p); // 归一化到 [0, 1)
            pos.push_back(p);
        }
    }

    std::sort(pos.begin(), pos.end());

    double max_gap = 0.0;
    double center = 0.5;

    // 内部间隙
    for (size_t i = 1; i < pos.size(); i++)
    {
        double gap = pos[i] - pos[i - 1];
        if (gap > max_gap)
        {
            max_gap = gap;
            center = (pos[i] + pos[i - 1]) / 2.0;
        }
    }

    // 跨边界间隙
    if (!pos.empty()) {
        double tail_gap = pos[0] + 1.0 - pos.back();
        if (tail_gap > max_gap)
        {
            max_gap = tail_gap;
            center = (pos[0] + pos.back() + 1.0) / 2.0;
            if (center >= 1.0) center -= 1.0;
        }
    }

    return center;
}

ParabolicCorrection::ParabolicCorrection() {}
ParabolicCorrection::~ParabolicCorrection() {}

// ---------------------------------------------------------
// 2. 应用校正的主函数
// ---------------------------------------------------------
void ParabolicCorrection::apply_correction(const UnitCell& cell, 
                                           const ModulePW::PW_Basis* rho_basis, 
                                           double* v_hartree, 
                                           const double* const* rho_elec, // 【修正1】指针的指针
                                           int nspin,                     // 【修正2】添加 nspin 参数
                                           double nelec,
                                           int dir)
{
    // 基础几何参数
    double omega = cell.omega;       
    double lat_vec = 0.0;            
    double area = 0.0;               

    if (dir == 0) {
        lat_vec = cell.a1.norm() * cell.lat0;
    } else if (dir == 1) {
        lat_vec = cell.a2.norm() * cell.lat0;
    } else { // dir == 2
        lat_vec = cell.a3.norm() * cell.lat0;
    }
    area = omega / lat_vec;

    // 1. 自动寻找真空层中心
    double vacuum_center = find_vacuum_center(cell, dir);

    // 2. 定义 Slab 的几何中心 (真空中心的对面)
    double slab_center = vacuum_center + 0.5;
    if(slab_center >= 1.0) slab_center -= 1.0;

    // 3. 计算净电荷 (Q_ion - Q_elec)
    double net_charge = calc_net_charge(cell, nelec); 
    
    // 4. 计算总偶极矩 (传递 slab_center 作为参考原点)
    double total_dipole = calc_total_dipole(cell, rho_basis, rho_elec, nspin, dir, slab_center);

    // 5. 构造 1D 修正势
    // 系数 factor = 4pi / Area * e^2
    double factor = (ModuleBase::FOUR_PI / area) * ModuleBase::e2;

    int nrxx = rho_basis->nrxx; 

    for (int ir = 0; ir < nrxx; ++ir)
    {
        int i = ir / (rho_basis->ny * rho_basis->nplane);
        int j = ir / rho_basis->nplane - i * rho_basis->ny;
        int k = ir % rho_basis->nplane + rho_basis->startz_current;

        double coord_frac = 0.0;
        if (dir == 0) coord_frac = (double)i / rho_basis->nx;
        else if (dir == 1) coord_frac = (double)j / rho_basis->ny;
        else coord_frac = (double)k / rho_basis->nz;

        // 计算到 Slab 中心的物理距离 (考虑 PBC)
        double dist_frac = coord_frac - slab_center;
        if (dist_frac > 0.5) dist_frac -= 1.0;
        if (dist_frac < -0.5) dist_frac += 1.0;

        double dist_bohr = dist_frac * lat_vec;

        // 核心抛物线修正公式
        double v_corr = factor * ( -0.5 * net_charge * dist_bohr * dist_bohr 
                                   + total_dipole * dist_bohr );

        v_hartree[ir] += v_corr;
    }
}

double ParabolicCorrection::calc_net_charge(const UnitCell& cell, double nelec)
{
    double ion_charge = 0.0;
    for(int it=0; it<cell.ntype; ++it) {
        ion_charge += cell.atoms[it].na * cell.atoms[it].ncpp.zv;
    }
    return ion_charge - nelec; 
}

double ParabolicCorrection::calc_total_dipole(const UnitCell& cell, 
                                              const ModulePW::PW_Basis* rho_basis,
                                              const double* const* rho_elec, // 【修正】类型匹配
                                              int nspin,                     // 【修正】传递 nspin
                                              int dir,
                                              double center)
{
    double lat_vec = 0.0;
    if(dir==0) lat_vec = cell.a1.norm() * cell.lat0;
    else if(dir==1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;

    double d_ion = calc_ion_dipole(cell, dir, center) * lat_vec;
    double d_elec = calc_elec_dipole(rho_basis, rho_elec, nspin, dir, center, cell.omega) * lat_vec;

    return d_ion - d_elec;
}

double ParabolicCorrection::calc_ion_dipole(const UnitCell& cell, int dir, double center)
{
    double d = 0.0;
    for(int it=0; it<cell.ntype; ++it) {
        for(int ia=0; ia<cell.atoms[it].na; ++ia) {
            double pos = cell.atoms[it].taud[ia][dir];
            double dist = pos - center;
            if(dist > 0.5) dist -= 1.0;
            if(dist < -0.5) dist += 1.0;

            d += cell.atoms[it].ncpp.zv * dist;
        }
    }
    return d;
}

// ---------------------------------------------------------
// 3. 计算电子偶极矩 (包含多自旋求和)
// ---------------------------------------------------------
double ParabolicCorrection::calc_elec_dipole(const ModulePW::PW_Basis* rho_basis, 
                                             const double* const* rho_elec, // 【修正】指针的指针
                                             int nspin,                     // 【修正】接收 nspin
                                             int dir, 
                                             double center,
                                             double omega)
{
    double d = 0.0;
    int nrxx = rho_basis->nrxx;

    // 【修正】Runtime 下，rho[0]=Up, rho[1]=Down，需要求和
    int n_components = (nspin == 2) ? 2 : 1;

    for (int ir = 0; ir < nrxx; ++ir)
    {
        int i = ir / (rho_basis->ny * rho_basis->nplane);
        int j = ir / rho_basis->nplane - i * rho_basis->ny;
        int k = ir % rho_basis->nplane + rho_basis->startz_current;

        double pos = 0.0;
        if(dir==0) pos = (double)i / rho_basis->nx;
        else if(dir==1) pos = (double)j / rho_basis->ny;
        else pos = (double)k / rho_basis->nz;

        double dist = pos - center;
        if(dist > 0.5) dist -= 1.0;
        if(dist < -0.5) dist += 1.0;

        double rho_val = 0.0;
        for(int is=0; is<n_components; ++is) {
            rho_val += rho_elec[is][ir]; // 【修正】正确访问二维数组
        }

        d += rho_val * dist;
    } // 【修正】补回了丢失的括号

    Parallel_Reduce::reduce_pool(d);

    d *= (omega / rho_basis->nxyz); 

    return d;
}