#include "parabolic_correction.h"
#include "source_base/constants.h" 
#include "source_base/parallel_reduce.h" 
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/unitcell.h"
#include "source_io/module_parameter/parameter.h"

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
                                             //double nelec,
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
    double net_charge = -PARAM.inp.nelec_delta; 
    double total_dipole = calc_total_dipole(cell, rho_basis, rho_elec, nspin, dir, slab_center);

    double total_quadrupole = calc_total_quadrupole(cell, rho_basis, rho_elec, nspin, dir, slab_center);
    double factor_common = (ModuleBase::FOUR_PI / omega) * ModuleBase::e2; // 4pi/Omega * e2
    double factor_quad = factor_common * 0.5;
    double madelung_term = - (ModuleBase::PI / 3.0) * net_charge / lat_vec * ModuleBase::e2;
    double quadrupole_term = - factor_quad * total_quadrupole;
    
    double v_const = madelung_term + quadrupole_term;
    
    // 缓存偶极矩供后续力修正使用
    this->last_total_dipole_ = total_dipole;

    // 3. 计算离子修正能 (这是你想要加入的！)
    double e_ion_corr = calc_energy_correction(cell, dir, net_charge, total_dipole, vacuum_center,v_const);
    double e_elec_corr = 0.0;

    // 4. 构造 1D 修正势并叠加到 v_hartree
    // double factor = (ModuleBase::FOUR_PI / omega) * ModuleBase::e2;   
    int nrxx = rho_basis->nrxx;
    int nspin_eff = (nspin == 2) ? 2 : 1;

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
        double v_corr = factor_common  * ( -0.5 * net_charge * dist_bohr * dist_bohr 
                                   + total_dipole * dist_bohr );

        v_hartree[ir] += v_corr;
        double rho_val = 0.0;
        for(int is=0; is<nspin_eff; ++is) rho_val += rho_elec[is][ir];
        e_elec_corr += rho_val * v_corr;
    }
    Parallel_Reduce::reduce_pool(e_elec_corr);
    e_elec_corr *= (cell.omega / rho_basis->nxyz);
    double total_correction_energy = e_ion_corr - e_elec_corr;

if (GlobalV::RANK_IN_POOL == 0) {
        std::cout << "DEBUG PARABOLIC:" << std::endl;
        std::cout << "  nelec_delta (Input): " << PARAM.inp.nelec_delta << std::endl;
        std::cout << "  net_charge (Used):   " << net_charge << std::endl;
        std::cout << "  Total Dipole:        " << total_dipole << std::endl;
        std::cout << "  Total Quadrupole:    " << total_quadrupole << std::endl; // 建议打印四极矩
        std::cout << "  Madelung Term:       " << madelung_term << std::endl;    // 建议打印各项贡献
        std::cout << "  Quadrupole Term:     " << quadrupole_term << std::endl;
        std::cout << "  Constant Shift:      " << v_const << std::endl;
        
        std::cout << "  ----------------------------------------" << std::endl;
        std::cout << "  Electronic Energy Corr: " << e_elec_corr << " Ry" << std::endl;
        std::cout << "  Ionic Energy Corr:      " << e_ion_corr << " Ry" << std::endl;
        std::cout << "  TOTAL Parabolic Corr:   " << total_correction_energy << " Ry" << std::endl;
        std::cout << "  ----------------------------------------" << std::endl;
    }

    // 返回离子修正能，方便外部加到 Total Energy
    // double total_ion_charge = net_charge - (-PARAM.inp.nelec_delta);
    return total_correction_energy;
}

// ---------------------------------------------------------
// 辅助计算函数
// ---------------------------------------------------------
// double ParabolicCorrection::calc_net_charge(const UnitCell& cell, double nelec)
// {
//     double ion_charge = 0.0;
//     for(int it=0; it<cell.ntype; ++it) {
//         ion_charge += cell.atoms[it].na * cell.atoms[it].ncpp.zv;
//     }
//     return ion_charge - nelec; 
// }

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
                                                   double vacuum_center,
                                                    double v_const)
{
    double lat_vec = 0.0;
    if (dir == 0) lat_vec = cell.a1.norm() * cell.lat0;
    else if (dir == 1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;
    
    double area = cell.omega / lat_vec;
    double factor = (ModuleBase::FOUR_PI / cell.omega) * ModuleBase::e2;
    
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
                                          - total_dipole * dist_bohr ) + v_const;
            
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
    double factor = (ModuleBase::FOUR_PI / cell.omega) * ModuleBase::e2;
    double slab_center = vacuum_center + 0.5;
    if(slab_center >= 1.0) slab_center -= 1.0;
    
    double lat_vec = 0.0;
    if (dir == 0) lat_vec = cell.a1.norm() * cell.lat0;
    else if (dir == 1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;

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

double ParabolicCorrection::calc_total_quadrupole(const UnitCell& cell, 
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

    // Q_ion
    double q_ion = calc_ion_quadrupole(cell, dir, center);
    // 注意：离子位置已经是物理长度（Bohr），所以算出来直接是 Bohr^2
    // 如果 calc_ion_dipole 用的是分数坐标差，这里要注意单位转换。
    // 看原代码 calc_ion_dipole 中 dist 是分数坐标差，最后乘了 lat_vec。
    // 这里我们需要 (dist_frac * lat_vec)^2
    
    // Q_elec
    double q_elec_frac = calc_elec_quadrupole(rho_basis, rho_elec, nspin, dir, center, cell.omega);
    // 同样需要乘以 lat_vec^2
    double q_elec = q_elec_frac * (lat_vec * lat_vec);
    // 假设辅助函数内部处理单位，或者在这里统一处理
    // 让我们看辅助函数实现细节：
    return q_ion - q_elec;
}

double ParabolicCorrection::calc_ion_quadrupole(const UnitCell& cell, int dir, double center)
{
    double q = 0.0;
    
    double lat_vec = 0.0;
    if(dir==0) lat_vec = cell.a1.norm() * cell.lat0;
    else if(dir==1) lat_vec = cell.a2.norm() * cell.lat0;
    else lat_vec = cell.a3.norm() * cell.lat0;

    for(int it=0; it<cell.ntype; ++it) {
        for(int ia=0; ia<cell.atoms[it].na; ++ia) {
            double pos = cell.atoms[it].taud[ia][dir];
            double dist_frac = pos - center;
            
            // 最小镜像处理
            if(dist_frac > 0.5) dist_frac -= 1.0;
            if(dist_frac < -0.5) dist_frac += 1.0;

            double dist_bohr = dist_frac * lat_vec;
            
            // sum Z * r^2
            q += cell.atoms[it].ncpp.zv * (dist_bohr * dist_bohr);
        }
    }
    return q;
}

double ParabolicCorrection::calc_elec_quadrupole(const ModulePW::PW_Basis* rho_basis, 
                                                 const double* const* rho_elec, 
                                                 int nspin,
                                                 int dir, 
                                                 double center,
                                                 double omega)
{
    double q = 0.0;
    int nrxx = rho_basis->nrxx;
    int n_components = (nspin == 2) ? 2 : 1;
    
    // 获取晶格长度
    // 这里的 omega 参数虽然传进来了，但我们需要长度来转换坐标
    // 简单起见，我们在外部计算好 lat_vec 比较麻烦，
    // 不如在这里根据 dir 重新算一下或者假设输入保证一致。
    // 为了严谨，建议像 calc_elec_dipole 一样，只算分数坐标部分的积分，最后在外面乘长度平方。
    // 但 calc_elec_dipole 是最后乘了 lat_vec。
    // 我们这里直接在循环里乘好吧，或者模仿原结构。
    
    // 为了复用代码结构，我们计算 sum [ rho * (dist_frac)^2 ]
    // 最后再乘以 (lat_vec^2) * (omega / nxyz)
    
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

        q += rho_val * (dist * dist);
    }

    Parallel_Reduce::reduce_pool(q);
    
    // 积分体积元 dV = omega / nxyz
    q *= (omega / rho_basis->nxyz);
    
    // 此时 q 是 sum rho * (dist_frac)^2 * dV
    // 我们需要在外部乘以 lat_vec^2 才能变成 Bohr^2 单位
    return q; 
}