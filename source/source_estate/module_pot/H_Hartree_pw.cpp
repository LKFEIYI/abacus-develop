#include "H_Hartree_pw.h"

#include "source_io/module_parameter/parameter.h"
#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_poisson/mt_poisson.h"
#include "source_hamilt/module_poisson/parabolic_correction.h"

namespace elecstate
{

double H_Hartree_pw::hartree_energy = 0.0;

//--------------------------------------------------------------------
// Transform charge density to hartree potential.
//--------------------------------------------------------------------
ModuleBase::matrix H_Hartree_pw::v_hartree(const UnitCell &cell,
                                           ModulePW::PW_Basis *rho_basis,
                                           const int &nspin,
                                           const double *const *const rho)
{
    ModuleBase::TITLE("H_Hartree_pw", "v_hartree");
    ModuleBase::timer::tick("H_Hartree_pw", "v_hartree");

    //  Hartree potential VH(r) from n(r)
    std::vector<std::complex<double>> Porter(rho_basis->nmaxgr);
    const int nspin0 = (nspin == 2) ? 2 : 1;
    for (int is = 0; is < nspin0; is++)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 256)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
            Porter[ir] += std::complex<double>(rho[is][ir], 0.0);
    }
    //=============================
    //  bring rho (aux) to G space
    //=============================
    rho_basis->real2recip(Porter.data(), Porter.data());

    //=======================================================
    // calculate hartree potential in G-space (NB: V(G=0)=0 )
    //=======================================================

    double ehart = 0.0;

    std::vector<std::complex<double>> vh_g(rho_basis->npw);
    const int ig0 = rho_basis->ig_gge0;

    bool use_mt = (PARAM.inp.dim_corr == "mt");
    bool use_parabolic = (PARAM.inp.dim_corr == "parabolic");
    double tpiba = cell.tpiba;
    double tpiba2 = cell.tpiba2;
    int dir = PARAM.inp.dim_corr_dir;; // 0=x, 1=y, 2=z
    double L = 0.0;
    if (dir == 0) L = cell.a1.norm() * cell.lat0; // X方向, 注意乘以 lat0 (Bohr)
    else if (dir == 1) L = cell.a2.norm() * cell.lat0; // Y方向
    else if (dir == 2) L = cell.a3.norm() * cell.lat0; // Z方向

#ifdef _OPENMP
#pragma omp parallel for reduction(+:ehart)
#endif
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
            
	    double g2 = tpiba2 * rho_basis->gg[ig];

        if (ig == ig0) 
        {
            vh_g[ig] = std::complex<double>(0.0, 0.0);
            continue; // skip G=0
        }
        double fac = ModuleBase::e2 * ModuleBase::FOUR_PI / g2;
       
        ehart += (conj(Porter[ig]) * Porter[ig]).real() * fac;
        vh_g[ig] = fac * Porter[ig];
        
    }

    Parallel_Reduce::reduce_pool(ehart);
    ehart *= 0.5 * cell.omega;
    // std::cout << " ehart=" << ehart << std::endl;
    H_Hartree_pw::hartree_energy = ehart;

    //==========================================
    // transform hartree potential to real space
    //==========================================
    rho_basis->recip2real(vh_g.data(), Porter.data());

    //==========================================
    // Add hartree potential to the xc potential
    //==========================================
    ModuleBase::matrix v(nspin, rho_basis->nrxx);
    if (nspin == 4)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 512)
#endif
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
            v(0, ir) = Porter[ir].real();
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static, 512)
#endif
        for (int is = 0; is < nspin; is++)
            for (int ir = 0; ir < rho_basis->nrxx; ir++)
                v(is, ir) = Porter[ir].real();
    }

if (PARAM.inp.dim_corr == "mt") 
{
    MTCorrection mt;
    std::vector<double> v_corr(rho_basis->nrxx, 0.0);
    
    // 计算修正
    double e_mt = mt.apply_correction(cell, rho_basis, v_corr.data(), rho, nspin, dir);
    
    // 叠加势场 (Total Spin)
    for (int ir = 0; ir < rho_basis->nrxx; ir++) {
        v(0, ir) += v_corr[ir];
    }
    // 如果有自旋，Spin down 也要加上同样的静电势修正
    if (nspin == 2) {
        for (int ir = 0; ir < rho_basis->nrxx; ir++) {
            v(1, ir) += v_corr[ir];
        }
    }
    
    // 修正能量
    H_Hartree_pw::hartree_energy += e_mt;
    
    if (GlobalV::RANK_IN_POOL == 0) {
        std::cout << " MT Correction Applied. Energy correction = " << e_mt << " Ry" << std::endl;
    }
}

if (use_parabolic) 
    {
        ParabolicCorrection pc;
        
        // 1. 调用 apply_correction (同时获取离子修正能)
        double e_ion = pc.apply_correction(
            cell,
            rho_basis,
            &v(0, 0),       // 势场指针
            rho,            // 密度指针 (const double* const*)
            nspin,          // 自旋
            //GlobalV::nelec,
            dir
        );

        // 2. 将离子修正能加到 Hartree 能量中
        // 这样总能量就正确了
        H_Hartree_pw::hartree_energy += e_ion;
        // 3. (可选) 日志输出
        if (GlobalV::RANK_IN_POOL == 0) {
            std::cout << " Parabolic Correction Applied. E_ion_corr = " << e_ion << " Ry" << std::endl;
        }   

        // 4. 处理 nspin=2 的第二列势场 (能量只需加一次，所以这里不加)
        if (nspin == 2) {
             pc.apply_correction(
                cell, rho_basis, &v(1, 0), 
                rho, nspin, dir
            );
        }
    }

    ModuleBase::timer::tick("H_Hartree_pw", "v_hartree");
    return v;
} // end subroutine v_h

PotHartree::PotHartree(const ModulePW::PW_Basis* rho_basis_in)
{
    this->rho_basis_ = rho_basis_in;
    this->dynamic_mode = true;
    this->fixed_mode = false;
}

void PotHartree::cal_v_eff(const Charge*const chg, const UnitCell*const ucell, ModuleBase::matrix& v_eff)
{
    v_eff += H_Hartree_pw::v_hartree(*ucell, const_cast<ModulePW::PW_Basis*>(this->rho_basis_), v_eff.nr, chg->rho);
    return;
}

} // namespace elecstate
