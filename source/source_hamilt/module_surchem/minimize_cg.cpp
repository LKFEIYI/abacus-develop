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
    // parameters of CG method
    double alpha = 0;
    double beta = 0;
    double rinvLr = 0;
    double r2 = 0;

    ModuleBase::GlobalFunc::ZEROS(phi, rho_basis->npw);

    // malloc vectors in G space
    std::complex<double> *resid = new std::complex<double>[rho_basis->npw];
    std::complex<double> *z = new std::complex<double>[rho_basis->npw];
    std::complex<double> *lp = new std::complex<double>[rho_basis->npw];
    std::complex<double> *gsqu = new std::complex<double>[rho_basis->npw];
    std::complex<double> *d = new std::complex<double>[rho_basis->npw];

    std::complex<double> *gradphi_x = new std::complex<double>[rho_basis->npw];
    std::complex<double> *gradphi_y = new std::complex<double>[rho_basis->npw];
    std::complex<double> *gradphi_z = new std::complex<double>[rho_basis->npw];
    std::complex<double> *phi_work = new std::complex<double>[rho_basis->npw];

    ModuleBase::GlobalFunc::ZEROS(resid, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(z, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(lp, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(gsqu, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(d, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(gradphi_x, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(gradphi_y, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(gradphi_z, rho_basis->npw);
    ModuleBase::GlobalFunc::ZEROS(phi_work, rho_basis->npw);

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
        // [Image of Modified Preconditioner]
        // 加上 avg_kappa2 使得 G=0 时分母不为0
        double denom = gg * ucell.tpiba2 + avg_kappa2; 
        if (denom < 1e-9) denom = 1e-9; 
        
        gsqu[ig].real(1.0 / denom);
        gsqu[ig].imag(0);
    }

    // init guess for phi
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        if(!has_salt && ig == ig0) continue;
        phi[ig] = tot_N[ig] * gsqu[ig];
    }

    // call leps
    Leps2(ucell, rho_basis, phi, d_eps, kappa2_factor, 
          gradphi_x, gradphi_y, gradphi_z, phi_work, lp);

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

        Leps2(ucell, rho_basis, d, d_eps, kappa2_factor, 
              gradphi_x, gradphi_y, gradphi_z, phi_work, lp);

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
    delete[] resid;
    delete[] z;
    delete[] lp;
    delete[] gsqu;
    delete[] d;
    delete[] gradphi_x;
    delete[] gradphi_y;
    delete[] gradphi_z;
    delete[] phi_work;
}

void surchem::Leps2(const UnitCell& ucell,
                    const ModulePW::PW_Basis* rho_basis,
                    std::complex<double>* phi,
                    double* epsilon,            
                    const double* kappa2_factor,
                    std::complex<double>* gradphi_x, 
                    std::complex<double>* gradphi_y,
                    std::complex<double>* gradphi_z,
                    std::complex<double>* phi_work,
                    std::complex<double>* lp)
{
    ModuleBase::Vector3<double> *grad_phi = new ModuleBase::Vector3<double>[rho_basis->nrxx];

    XC_Functional::grad_rho(phi, grad_phi, rho_basis, ucell.tpiba);
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        grad_phi[ir].x *= epsilon[ir];
        grad_phi[ir].y *= epsilon[ir];
        grad_phi[ir].z *= epsilon[ir];
    }
    
    std::vector<double> lp_real(rho_basis->nrxx, 0.0);
    ModuleBase::GlobalFunc::ZEROS(lp, rho_basis->npw);

    std::vector<double> grad_grad_phi(rho_basis->nrxx, 0.0);
    std::complex<double> *grad_grad_phi_G = new std::complex<double>[rho_basis->npw];
    ModuleBase::Vector3<double> *tmp_vector3 = new ModuleBase::Vector3<double>[rho_basis->nrxx];

    // Helper lambda to calculate div component
    auto calc_div_component = [&](double ModuleBase::Vector3<double>::* component) {
        ModuleBase::GlobalFunc::ZEROS(grad_grad_phi_G, rho_basis->npw);
        ModuleBase::GlobalFunc::ZEROS(tmp_vector3, rho_basis->nrxx);
        for (int ir = 0; ir < rho_basis->nrxx; ir++) {
            grad_grad_phi[ir] = grad_phi[ir].*component;
        }
        rho_basis->real2recip(grad_grad_phi.data(), grad_grad_phi_G);
        XC_Functional::grad_rho(grad_grad_phi_G, tmp_vector3, rho_basis, ucell.tpiba);
        for (int ir = 0; ir < rho_basis->nrxx; ir++) {
            lp_real[ir] += tmp_vector3[ir].*component;
        }
    };

    // Calculate div(eps grad phi)
    calc_div_component(&ModuleBase::Vector3<double>::x);
    calc_div_component(&ModuleBase::Vector3<double>::y);
    calc_div_component(&ModuleBase::Vector3<double>::z);

    // === VASPsol++ Contribution: - epsilon * kappa^2 * phi ===
    if (kappa2_factor != nullptr)
    {
        double* phi_real = new double[rho_basis->nrxx];
        rho_basis->recip2real(phi, phi_real);
        #pragma omp parallel for schedule(static)
        for(int ir = 0; ir < rho_basis->nrxx; ir++)
        {
            // L = div(eps grad) - eps * kappa^2
            // 注意: kappa2_factor 是 4pi * d(rho)/d(phi)
            // 根据 Helmholtz 算符定义，这里应减去线性
            lp_real[ir] -= epsilon[ir] * kappa2_factor[ir] * phi_real[ir];
        }
        delete[] phi_real;
    }
    // =========================================================

    rho_basis->real2recip(lp_real.data(), lp);

    delete[] grad_phi;
    delete[] grad_grad_phi_G;
    delete[] tmp_vector3;
}