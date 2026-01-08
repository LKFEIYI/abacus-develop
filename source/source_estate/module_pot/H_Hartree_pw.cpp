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
        if (use_mt)
        {
            ModuleBase::Vector3<double> g_vec = rho_basis->gcar[ig] * tpiba;
            
            double screen_val = MTPoisson::get_screen_val(g2, g_vec, L, 0, PARAM.inp.mt_type, dir); //alpha is not used now

            fac += ModuleBase::e2 * screen_val;
        }
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
    if (do_parabolic) 
if (use_parabolic) 
    {
        ParabolicCorrection pc;
        // int dir = PARAM.inp.dim_corr_dir;

        // 1. 修正 Spin 1 (Up) 或 Total
        // 注意：这里传入完整的 rho 和 nspin，模块内部会自动处理密度求和
        pc.apply_correction(
            GlobalC::unitcell,
            rho_basis,
            &v(0, 0),       // 修改第一列势场
            rho,            // 传入完整的 rho 指针 (const double* const*)
            nspin,          // 传入 nspin
            GlobalV::nelec,
            dir
        );

        // 2. 如果是 Spin 2 (Down)，必须施加同样的修正
        // 因为 Hartree 势对 Up 和 Down 电子是一样的
        if (nspin == 2) {
             pc.apply_correction(
                GlobalC::unitcell,
                rho_basis,
                &v(1, 0),   // 修改第二列势场
                rho,        // 依然传入同样的 rho，计算出的偶极矩是一样的
                nspin,
                GlobalV::nelec,
                dir
            );
        }
    }
        // 对于 nspin=4，通常只处理 v(0, :)，因为它是 H_scalar
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
