#include "surchem.h"
#include "source_base/global_function.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include <iostream>

// 注意：minimize_cg.cpp 只包含求解器逻辑

void surchem::minimize_cg(const UnitCell& ucell,
                          const ModulePW::PW_Basis* rho_basis,
                          double* d_eps,
                          const double* kappa2_factor,
                          const std::complex<double>* tot_N,
                          std::complex<double>* phi,
                          int& ncgsol)
{

    std::complex<double>* resid = this->cg_resid.data();
    std::complex<double>* z = this->cg_z.data();
    std::complex<double>* lp = this->cg_lp.data();
    std::complex<double>* gsqu = this->cg_gsqu.data();
    std::complex<double>* d = this->cg_d.data();

    ModuleBase::GlobalFunc::ZEROS(phi, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(resid, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(z, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(lp, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(d, rho_basis->npw);
    // parameters of CG method
    double alpha = 0;
    double beta = 0;
    double rinvLr = 0;
    double r2 = 0;





    int count = 0;
    
    // Check salt
    bool has_salt = (kappa2_factor != nullptr);
    double avg_kappa2 = 0.0;
    if (has_salt)
    {
        for (int i = 0; i < rho_basis->nrxx; ++i) avg_kappa2 += kappa2_factor[i];
        Parallel_Reduce::reduce_pool(avg_kappa2);
        avg_kappa2 /= rho_basis->nrxx;
    }
    
    // calculate precondition vector GSQU
    const int ig0 = rho_basis->ig_gge0;
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        double gg = rho_basis->gg[ig];
        double denom = gg * ucell.tpiba2 + avg_kappa2; 
        if (denom < 1e-9) denom = 1e-9; 
        
        gsqu[ig] = std::complex<double>(1.0 / denom, 0.0);
    }

    // init guess for phi
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if(!has_salt && ig == ig0) continue;
        phi[ig] = tot_N[ig] * gsqu[ig];
    }

    // call leps
    Leps2(ucell, rho_basis, phi, d_eps, kappa2_factor, lp);

    // residue
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if (!has_salt && ig == ig0) continue;
        resid[ig] = lp[ig] + tot_N[ig];
    }

    // z = M^-1 r
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if(!has_salt && ig == ig0) continue;
        z[ig] = gsqu[ig] * resid[ig];
    }

    // calculate r*z
    rinvLr = ModuleBase::GlobalFunc::ddot_real(rho_basis->npw, resid, z);
    r2 = ModuleBase::GlobalFunc::ddot_real(rho_basis->npw, resid, resid);

    // copy d = z
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if(!has_salt && ig == ig0) continue; // 修正语法错误 ifif -> if
        d[ig] = z[ig];
    }

    // CG Loop
    while (count < 20000 && sqrt(r2) > 1e-5 && sqrt(rinvLr) > 1e-10)
    {
        if (sqrt(r2) > 1e6)
        {
            std::cout << "CG ERROR!!! Diverged." << std::endl;
            break;
        }

        Leps2(ucell, rho_basis, d, d_eps, kappa2_factor, lp);

        // calculate alpha
        std::complex<double> d_dot_lp = 0.0;
        for (int ig = 0; ig < rho_basis->npw; ig++)
        {
             if (!has_salt && ig == ig0) continue;
             d_dot_lp += std::conj(d[ig]) * lp[ig];
        }
        double denom_alpha = d_dot_lp.real();
        Parallel_Reduce::reduce_pool(denom_alpha);
        
        // protect division by zero
        if(std::abs(denom_alpha) < 1e-20) break;

        alpha = -rinvLr / denom_alpha;
        
        // update phi & resid
        for (int ig = 0; ig < rho_basis->npw; ig++)
        {
            if(!has_salt && ig == ig0) continue;
            phi[ig] += alpha * d[ig];
            resid[ig] += alpha * lp[ig];
        }

        // precond
        for (int ig = 0; ig < rho_basis->npw; ig++)
        {
            if(!has_salt && ig == ig0) continue;
            z[ig] = gsqu[ig] * resid[ig];
        }

        // calculate beta
        double rinvLr_old = rinvLr;
        rinvLr = ModuleBase::GlobalFunc::ddot_real(rho_basis->npw, resid, z);
        beta = rinvLr / rinvLr_old;
        
        // update d
        for (int ig = 0; ig < rho_basis->npw; ig++)
        {
            if(!has_salt && ig == ig0) continue;
            d[ig] = beta * d[ig] + z[ig];
        }
        
        // check convergence
        r2 = ModuleBase::GlobalFunc::ddot_real(rho_basis->npw, resid, resid);

        count++;
    } 

    ncgsol = count;

    // cleanup
}

void surchem::Leps2(const UnitCell& ucell,
                    const ModulePW::PW_Basis* rho_basis,
                    std::complex<double>* phi,
                    double* epsilon,            
                    const double* kappa2_factor,
                    std::complex<double>* lp)
{
    ModuleBase::Vector3<double> *grad_phi = this->le_grad_phi.data();
    std::complex<double> *grad_grad_phi_G = this->le_grad_grad_phi_G.data();
    ModuleBase::Vector3<double> *tmp_vector3 = this->le_tmp_vector3.data();
    double *lp_real_ptr = this->le_lp_real.data();
    double *aux_real_ptr = this->le_aux_real.data();


    #pragma omp parallel for schedule(static)
    for(int i=0; i<rho_basis->nrxx; ++i) {
        lp_real_ptr[i] = 0.0;
    }

    XC_Functional::grad_rho(phi, grad_phi, rho_basis, ucell.tpiba);

    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        grad_phi[ir].x *= epsilon[ir];
        grad_phi[ir].y *= epsilon[ir];
        grad_phi[ir].z *= epsilon[ir];
    }
    





    // Helper lambda to calculate div component
    auto calc_div_component = [&](double ModuleBase::Vector3<double>::* component) {

        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < rho_basis->nrxx; ir++) {
            aux_real_ptr[ir] = grad_phi[ir].*component;
        }
        rho_basis->real2recip(aux_real_ptr, grad_grad_phi_G);
        XC_Functional::grad_rho(grad_grad_phi_G, tmp_vector3, rho_basis, ucell.tpiba);

        #pragma omp parallel for schedule(static)
        for (int ir = 0; ir < rho_basis->nrxx; ir++) {
            // 所有线程写入同一个 lp_real 数组，但写入索引 ir 不同，因此无需 atomic，绝对安全。
            lp_real_ptr[ir] += tmp_vector3[ir].*component;
        }
    };

    // Calculate div(eps grad phi)
    calc_div_component(&ModuleBase::Vector3<double>::x);
    calc_div_component(&ModuleBase::Vector3<double>::y);
    calc_div_component(&ModuleBase::Vector3<double>::z);

    // === VASPsol++ Contribution: - epsilon * kappa^2 * phi ===
    if (kappa2_factor != nullptr)
    {
        // 复用 aux_real_ptr 来存储 phi_real，节省内存
        // 注意：这里覆盖了 aux_real_ptr 的旧数据，这是安全的，因为上面已经用完了
        rho_basis->recip2real(phi, aux_real_ptr);
        
        #pragma omp parallel for schedule(static)
        for(int ir = 0; ir < rho_basis->nrxx; ir++)
        {
            lp_real_ptr[ir] -= epsilon[ir] * kappa2_factor[ir] * aux_real_ptr[ir];
        }
    }
    // =========================================================

    rho_basis->real2recip(lp_real_ptr, lp);


}