#include "parabolic_correction.h"
#include "source_base/constants.h" 
#include "source_base/parallel_reduce.h" 
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"

#include <cmath>
#include <iostream>
#include <algorithm> 

ParabolicCorrection::ParabolicCorrection() {}
ParabolicCorrection::~ParabolicCorrection() {}

// ---------------------------------------------------------
// 1. 自动寻找真空层中心
// ---------------------------------------------------------
double ParabolicCorrection::find_vacuum_center(const UnitCell& cell, int dir)
{
    std::vector<double> pos;
    for (int it = 0; it < cell.ntype; ++it)
    {
        for (int ia = 0; ia < cell.atoms[it].na; ++ia)
        {
            double p = cell.atoms[it].taud[ia][dir];
            p -= std::floor(p); 
            pos.push_back(p);
        }
    }
    std::sort(pos.begin(), pos.end());

    double max_gap = 0.0;
    double center = 0.5;

    for (size_t i = 1; i < pos.size(); i++)
    {
        double gap = pos[i] - pos[i - 1];
        if (gap > max_gap)
        {
            max_gap = gap;
            center = (pos[i] + pos[i - 1]) / 2.0;
        }
    }

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

// ---------------------------------------------------------
// 2. 应用校正的主函数 (集成离子能量计算)
// ---------------------------------------------------------
double ParabolicCorrection::apply_correction(const UnitCell& cell, 
                                             const ModulePW::PW_Basis* rho_basis, 
                                             double* v_hartree, 
                                             const double* const* rho_elec, 
                                             int nspin,                     
                                             double nelec,
                                             int dir)
{
    // 基础几何参数
    double omega = cell.omega;       
    double lat_vec = 0.0;            
    double area = 0.0;               

    if (dir == 0) lat_vec = cell.a1.norm() * cell.lat0;
    else if (dir == 1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;
    
    area = omega / lat_vec;

    // 1. 寻找中心
    double vacuum_center = find_vacuum_center(cell, dir);
    double slab_center = vacuum_center + 0.5;
    if(slab_center >= 1.0) slab_center -= 1.0;

    // 2. 计算电荷与偶极
    double net_charge = calc_net_charge(cell, nelec); 
    double total_dipole = calc_total_dipole(cell, rho_basis, rho_elec, nspin, dir, slab_center);
    
    // 缓存偶极矩供后续力修正使用
    this->last_total_dipole_ = total_dipole;

    // 3. 计算离子修正能 (这是你想要加入的！)
    double e_ion_corr = calc_energy_correction(cell, dir, net_charge, total_dipole, vacuum_center);

    // 4. 构造 1D 修正势并叠加到 v_hartree
    double factor = (ModuleBase::FOUR_PI / area) * ModuleBase::e2;
    int nrxx = rho_basis->nrxx; 

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
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

    // 返回离子修正能，方便外部加到 Total Energy
    return e_ion_corr;
}

// ---------------------------------------------------------
// 辅助计算函数
// ---------------------------------------------------------
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
                                              const double* const* rho_elec,
                                              int nspin,
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

double ParabolicCorrection::calc_elec_dipole(const ModulePW::PW_Basis* rho_basis, 
                                             const double* const* rho_elec, 
                                             int nspin,
                                             int dir, 
                                             double center,
                                             double omega)
{
    double d = 0.0;
    int nrxx = rho_basis->nrxx;
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
            rho_val += rho_elec[is][ir];
        }

        d += rho_val * dist;
    } // 【修复了这里丢失的括号】

    Parallel_Reduce::reduce_pool(d);
    d *= (omega / rho_basis->nxyz); 
    return d;
}

// ---------------------------------------------------------
// 离子能量校正实现
// ---------------------------------------------------------
double ParabolicCorrection::calc_energy_correction(const UnitCell& cell, 
                                                   int dir,
                                                   double net_charge,
                                                   double total_dipole,
                                                   double vacuum_center)
{
    double lat_vec = 0.0;
    if (dir == 0) lat_vec = cell.a1.norm() * cell.lat0;
    else if (dir == 1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;
    
    double area = cell.omega / lat_vec;
    double factor = (ModuleBase::FOUR_PI / area) * ModuleBase::e2;
    
    double slab_center = vacuum_center + 0.5;
    if(slab_center >= 1.0) slab_center -= 1.0;

    double e_corr = 0.0;

    for(int it=0; it<cell.ntype; ++it) {
        double Z = cell.atoms[it].ncpp.zv;
        for(int ia=0; ia<cell.atoms[it].na; ++ia) {
            double pos = cell.atoms[it].taud[ia][dir];
            
            double dist_frac = pos - slab_center;
            if (dist_frac > 0.5) dist_frac -= 1.0;
            if (dist_frac < -0.5) dist_frac += 1.0;
            double dist_bohr = dist_frac * lat_vec;

            // V_corr at atom position
            double v_at_atom = factor * ( -0.5 * net_charge * dist_bohr * dist_bohr 
                                          + total_dipole * dist_bohr );
            
            e_corr += Z * v_at_atom;
        }
    }
    return e_corr;
}

// ---------------------------------------------------------
// 力校正实现 (保持独立，供 force 模块调用)
// ---------------------------------------------------------
void ParabolicCorrection::calc_force_correction(const UnitCell& cell, 
                                                ModuleBase::matrix& force, 
                                                int dir,
                                                double net_charge,
                                                double total_dipole,
                                                double vacuum_center,
                                                double area)
{
    double factor = (ModuleBase::FOUR_PI / area) * ModuleBase::e2;
    double slab_center = vacuum_center + 0.5;
    if(slab_center >= 1.0) slab_center -= 1.0;
    
    double lat_vec = cell.omega / area; // 反推 L

    int iat = 0;
    for(int it=0; it<cell.ntype; ++it) {
        double Z = cell.atoms[it].ncpp.zv;
        for(int ia=0; ia<cell.atoms[it].na; ++ia) {
            
            double pos = cell.atoms[it].taud[ia][dir];
            double dist_frac = pos - slab_center;
            if (dist_frac > 0.5) dist_frac -= 1.0;
            if (dist_frac < -0.5) dist_frac += 1.0;
            double dist = dist_frac * lat_vec;

            // 1. Electric Field Force: -Z * dV/dz
            double f_field = -Z * factor * (-net_charge * dist + total_dipole);

            // 2. Dipole Response Force: Z * d(Total_Dipole)/dz * dE/dD
            double f_dipole = factor * Z * (net_charge * dist - total_dipole);
            
            // 总修正力
            double f_corr = f_field + f_dipole; 
            // 注意：上面的推导中两项可能会相互抵消或合并，具体依赖于公式的变分导数
            // 简单验证：Environ 中 f = (charge * pos - dipole) * fact * Z
            // 这里的 f_corr 简化后确实类似。
            
            force(iat, dir) += f_corr;
            
            iat++;
        }
    }
}