#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include "surchem.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include "source_base/global_function.h"

// --- 物理常数 ---
const double KB_au = 3.1668114e-6;      // Boltzmann constant in Hartree/K
const double Ang2Bohr = 1.0 / 0.52917721; // Angstrom to Bohr conversion

// =============================================================================
// Helper Function 1: Calculate SMPBE Physics (Ionic Charge & Kappa^2)
// 用于 imp_sol = 2 & 3
// =============================================================================
void surchem::cal_smpbe_physics(const int nrxx,
                                const double* phi_R,
                                const double* eps_R,
                                double* rho_ion_out,
                                double* kappa2_factor_out)
{
    // 1. 获取参数
    double T = (PARAM.inp.sol_temp < 1.0) ? 298.15 : PARAM.inp.sol_temp;
    double beta = 1.0 / (KB_au * T);
    double c_bulk_M = PARAM.inp.c_molar;
    double c_bulk_au = c_bulk_M * 6.022e-4 * pow(0.52917721, 3);
    double z = (PARAM.inp.z_ion == 0) ? 1.0 : PARAM.inp.z_ion;
    double a_ion = (PARAM.inp.ion_size < 0.1) ? 3.0 : PARAM.inp.ion_size;
    
    // Packing fraction theta = c_bulk / c_max
    double theta = c_bulk_au * pow(a_ion * Ang2Bohr, 3); 
    #pragma omp parallel for schedule(static)
    for (int ir = 0; ir < nrxx; ir++)
    {
        // 在介电常数接近 1 的区域（真空/板层内部），强制无离子
        if (eps_R[ir] < 1.5) {
            rho_ion_out[ir] = 0.0;
            kappa2_factor_out[ir] = 0.0;
            continue;
        }

        double u = z * beta * phi_R[ir];
        
        // 限制 u 的范围防止溢出 (Though ABACUS uses double, safety first)
        u = std::max(-20.0, std::min(u, 20.0));

        double exp_u = exp(u);
        double exp_neg_u = 1.0 / exp_u;
        double sinh_u = 0.5 * (exp_u - exp_neg_u);
        double cosh_u = 0.5 * (exp_u + exp_neg_u);
        
        // SMPBE Formula: Lattice-Gas Model
        double denom = 1.0 + theta * (cosh_u - 1.0);
        
        // rho_ion (e/Bohr^3)
        // 符号: 正电势吸引负离子(如果z>0)，rho应为负。
        // 假设 z 是离子的绝对电荷数，这里我们需要处理正负离子平衡
        // 标准 SMPBE: rho = -2zc * sinh(u) / denom (对于 1:1 电解质)
        // 这里的公式系数对应 z:z 电解质
        rho_ion_out[ir] = -z * c_bulk_au * sinh_u / denom;

        // kappa2_factor = 4pi * |d(rho)/d(phi)|
        // d(rho)/d(u) formula derivation
        double deriv_term = (cosh_u + theta * (1.0 - cosh_u)) / (denom * denom);
        
        // kappa^2 系数 (用于 Helmholz 方程左端项)
        kappa2_factor_out[ir] = 4.0 * ModuleBase::PI * (z * beta * z * c_bulk_au * deriv_term);
    }
}

// =============================================================================
// Helper Function 2: Calculate Dielectric Saturation
// 用于 imp_sol = 3
// =============================================================================
void cal_dielectric_saturation(const int nrxx,
                               const ModuleBase::Vector3<double>* grad_phi,
                               const double* shape_func,
                               double* epsilon_out)
{
    double eps_inf = (PARAM.inp.epsilon_inf < 1.0) ? 1.78 : PARAM.inp.epsilon_inf;
    double eps_bulk = PARAM.inp.eb_k;
    double T = (PARAM.inp.sol_temp < 1.0) ? 298.15 : PARAM.inp.sol_temp;
    double beta = 1.0 / (KB_au * T);
    
    // Debye -> Atomic Units conversion (1 Debye approx 0.39343 au)
    double p_mol_au = (PARAM.inp.p_mol < 0.01) ? 1.85 * 0.39343 : PARAM.inp.p_mol * 0.39343;
    // Mol density: 1/Ang^3 -> 1/Bohr^3
    double n_mol_au = (PARAM.inp.n_mol < 1e-4) ? 0.0333 * pow(0.52917721, 3) : PARAM.inp.n_mol;
    #pragma omp parallel for schedule(static)
    for(int i=0; i<nrxx; ++i) {
        if(shape_func[i] < 1e-6) {
            epsilon_out[i] = 1.0;
            continue;
        }

        double E_val = sqrt(grad_phi[i].x*grad_phi[i].x + 
                            grad_phi[i].y*grad_phi[i].y + 
                            grad_phi[i].z*grad_phi[i].z);

        // Langevin function L(x) = coth(x) - 1/x
        double x = beta * p_mol_au * E_val;
        double langevin = 0.0;
        
        if(x < 1e-4) {
            langevin = x / 3.0; 
        } else {
            // 优化尝试：利用 libm::exp 替换 tanh
            // tanh(x) = 1 - 2 / (exp(2x) + 1)
            double exp_2x = exp(2.0 * x);
            double tanh_x = 1.0 - 2.0 / (exp_2x + 1.0);
            
            langevin = (1.0 / tanh_x) - (1.0 / x);
        }
        
        double term_dipole = 0.0;
        // Bootstrapping: when E->0, should recover eps_bulk
        // Use Kirchhoff's formula for P vs E
        if(E_val > 1e-10) {
            term_dipole = (4.0 * ModuleBase::PI * n_mol_au * p_mol_au / E_val) * langevin;
        } else {
            term_dipole = (eps_bulk - eps_inf);
        }
        
        double eps_pure = eps_inf + term_dipole;
        
        // Mix with shape function: eps(r) = 1 + (eps_sat - 1) * S(r)
        epsilon_out[i] = 1.0 + (eps_pure - 1.0) * shape_func[i];
    }
}

// =============================================================================
// Helper Function 3: Shape Gradient (Standard VASPsol)
// =============================================================================
void shape_gradn(const double* PS_TOTN_real, const ModulePW::PW_Basis* rho_basis, double* eprime)
{
    double epr_c = 1.0 / sqrt(ModuleBase::TWO_PI) / PARAM.inp.sigma_k;
    double epr_z = 0;
    double min = 1e-10;
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        // Gaussian error function derivative chain rule
        epr_z = log(std::max(PS_TOTN_real[ir], min) / PARAM.inp.nc_k) / sqrt(2) / PARAM.inp.sigma_k;
        eprime[ir] = epr_c * exp(-pow(epr_z, 2)) / std::max(PS_TOTN_real[ir], min);
  
    }
}

// =============================================================================
// Helper Function 4: Epsilon Potential (Non-electrostatic term)
// =============================================================================
void eps_pot(const double* PS_TOTN_real,
             const double& tpiba,
             const std::complex<double>* phi,
             const ModulePW::PW_Basis* rho_basis,
             const double* epsilon_for_calc,
             double* vwork)
{
    double *eprime = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(eprime, rho_basis->nrxx);

    shape_gradn(PS_TOTN_real, rho_basis, eprime);

    // Note: V_epsilon depends on d(eps)/d(n).
    // Even if imp_sol=3 (saturated), the boundary position is mainly determined by electron density.
    // So we use (eps_bulk - 1) * d(shape)/d(n) as the primary force.
    // A rigorous derivation for saturated epsilon is much more complex.
    // Approximation: Use the fixed bulk epsilon parameter for the boundary force magnitude.
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        eprime[ir] = eprime[ir] * (PARAM.inp.eb_k - 1.0);
    }

    ModuleBase::Vector3<double> *nabla_phi = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    double *phisq = new double[rho_basis->nrxx];

    // nabla phi
    XC_Functional::grad_rho(phi, nabla_phi, rho_basis, tpiba);

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        phisq[ir] = pow(nabla_phi[ir].x, 2) + pow(nabla_phi[ir].y, 2) + pow(nabla_phi[ir].z, 2);
        vwork[ir] = eprime[ir] * phisq[ir] / (8.0 * ModuleBase::PI);
    }

    delete[] eprime;
    delete[] nabla_phi;
    delete[] phisq;
}

// =============================================================================
// Main Function: cal_vel
// =============================================================================
ModuleBase::matrix surchem::cal_vel(const UnitCell& cell,
                                    const ModulePW::PW_Basis* rho_basis,
                                    std::complex<double>* TOTN,
                                    std::complex<double>* PS_TOTN,
                                    int nspin)
{
    ModuleBase::TITLE("surchem", "cal_vel");
    ModuleBase::timer::tick("surchem", "cal_vel");

    // Optional: Run diagnostic test once
    /*
    static bool test_done = false;
    if(!test_done && ModuleBase::GlobalFunc::MY_RANK==0) {
        test_smpbe_driver(cell, rho_basis);
        test_done = true;
    }
    */

    // 1. Prepare Data
    rho_basis->recip2real(TOTN, TOTN_real);
    const double switch_threshold = 0.01; 
    
    // 获取用户设定的求解模式 (2 或 3)
    int target_imp_sol = PARAM.inp.imp_sol; // 或者 INPUT.imp_sol
    int actual_run_mode = target_imp_sol;   // 实际运行的模式

    // 计算局部的 DRHO
    double local_drho = 0.0;
    
    // 检查历史密度是否存在且大小匹配
    if(this->rho_history.size() == rho_basis->npw)
    {
        for(int i=0; i<rho_basis->npw; ++i)
        {
            // 简单的误差度量：所有 G 分量的模之和 (或者你可以做 FFT 后在实空间积分)
            // 这里为了快，直接在倒空间估算
            local_drho += std::abs(PS_TOTN[i] - this->rho_history[i]);
        }
        // 归一化 (可选，视 ps_totn 的量级而定，通常 ps_totn 是 1/Omega 量级)
        // 也可以简单地看绝对值变化
    }
    else
    {
        // 如果是第一步 (没有历史)，强制认为误差很大，或者直接跑线性
        local_drho = 100.0; 
        this->rho_history.resize(rho_basis->npw);
    }

    // 保存当前密度到历史 (供下一步用)
    for(int i=0; i<rho_basis->npw; ++i) {
        this->rho_history[i] = PS_TOTN[i];
    }

    // [决策时刻]
    if (local_drho > switch_threshold)
    {
        // 误差太大，降级为线性模型 (跑得快，稳)
        if (GlobalV::MY_RANK == 0 && target_imp_sol >= 2) {
            std::cout << " [SURCHEM] Large DRHO (" << local_drho 
                      << " > " << switch_threshold 
                      << "), downgrading to Linear Model (imp_sol=1)." << std::endl;
        }
        actual_run_mode = 1; 
    }
    else
    {
        // 误差小，开启完全非线性迭代
        actual_run_mode = target_imp_sol;
    }
    // B_elec = -4pi * rho_elec(G)
    std::complex<double> *B_elec = new std::complex<double>[rho_basis->npw];
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        B_elec[ig] = -4.0 * ModuleBase::PI * TOTN[ig];
    }

    double *PS_TOTN_real = new double[rho_basis->nrxx];
    rho_basis->recip2real(PS_TOTN, PS_TOTN_real);

    // 2. Build initial epsilon (based on shape)
    double *epsilon = new double[rho_basis->nrxx];
    double *epsilon0 = new double[rho_basis->nrxx];
    cal_epsilon(rho_basis, PS_TOTN_real, epsilon, epsilon0);

    // Store shape function reference for imp_sol=3
    // shape = (eps - 1) / (eb - 1)
    double *shape_ref = new double[rho_basis->nrxx];
    for(int i=0; i<rho_basis->nrxx; ++i) 
        shape_ref[i] = (epsilon[i] - 1.0) / (std::max(PARAM.inp.eb_k - 1.0, 1e-10));

    // 3. Setup Variables for Solver
    std::complex<double> *Sol_phi = new std::complex<double>[rho_basis->npw];
    std::complex<double> *Sol_phi0 = new std::complex<double>[rho_basis->npw];
    if (this->phi_history.size() == rho_basis->npw) {
        // 有缓存：拷贝作为初猜
        for(int i=0; i<rho_basis->npw; ++i) {
            Sol_phi[i] = this->phi_history[i];
        }
    } else {
        // 无缓存或大小不匹配：重置为0
        ModuleBase::GlobalFunc::ZEROS(Sol_phi, rho_basis->npw);
        // 调整大小以备后用
        this->phi_history.resize(rho_basis->npw, std::complex<double>(0,0));
    }


    double* rho_ion_R = new double[rho_basis->nrxx];
    double* kappa2_R = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(rho_ion_R, rho_basis->nrxx);
    ModuleBase::GlobalFunc::ZEROS(kappa2_R, rho_basis->nrxx);

    int ncgsol = 0;
    
    // Mode Determination
    // int mode = PARAM.inp.imp_sol; 
    int mode = actual_run_mode;
    bool is_nonlinear = (mode >= 2);       // imp_sol = 2 or 3
    bool use_dielectric_sat = (mode == 3); // imp_sol = 3 only
    double sol_thr_val = (PARAM.inp.sol_thr > 1e-12) ? PARAM.inp.sol_thr : 1.0e-5;



    // =========================================================================
    // Nonlinear Loop (VASPsol++) OR Linear Solver (VASPsol)
    // =========================================================================
    if (is_nonlinear)
    {
        // Prep temp arrays for loop
        double* phi_R_tmp = new double[rho_basis->nrxx];
        std::complex<double> *B_total = new std::complex<double>[rho_basis->npw];
        std::complex<double> *phi_new = new std::complex<double>[rho_basis->npw];
        
        // Newton-Raphson Loop
        int max_iter = 100;
        for(int iter = 0; iter < max_iter; ++iter)
        {
            // A. Get current Real-space Potential
            rho_basis->recip2real(Sol_phi, phi_R_tmp);

            // B. Update Physics: rho_ion & kappa^2 (Mode 2 & 3)
            cal_smpbe_physics(rho_basis->nrxx, phi_R_tmp, epsilon, rho_ion_R, kappa2_R);

            // C. Update Physics: Epsilon Saturation (Mode 3 Only)
            if(use_dielectric_sat)
            {
                ModuleBase::Vector3<double> *grad_phi_R = new ModuleBase::Vector3<double>[rho_basis->nrxx];
                XC_Functional::grad_rho(Sol_phi, grad_phi_R, rho_basis, cell.tpiba);
                
                cal_dielectric_saturation(rho_basis->nrxx, grad_phi_R, shape_ref, epsilon);
                
                delete[] grad_phi_R;
            }

            // D. Construct Linearized Source Term
            // Linearized Eq: (L - eps*k^2) * dphi = Residual
            // Equivalent to solving: (L - eps*k^2) * phi_new = B_elec + B_ion(phi_old) - eps*k^2*phi_old
            // RHS in Real Space:
            #pragma omp parallel for schedule(static)
            for(int i=0; i<rho_basis->nrxx; ++i) {
                phi_R_tmp[i] = -4.0 * ModuleBase::PI * rho_ion_R[i] 
                               - epsilon[i] * kappa2_R[i] * phi_R_tmp[i];
            }
            
            // Transform RHS to G space & Add Electron Source
            rho_basis->real2recip(phi_R_tmp, B_total);
            for(int ig=0; ig<rho_basis->npw; ++ig) B_total[ig] += B_elec[ig];

            // E. Solve Linearized Equation
            // Note: minimize_cg must accept kappa2_factor
            minimize_cg(cell, rho_basis, epsilon, kappa2_R, B_total, phi_new, ncgsol);

            // F. Mixing & Convergence Check
            double alpha = 0.6; // Mixing parameter
            double diff = 0.0;
            for(int ig=0; ig<rho_basis->npw; ++ig) {
                diff += std::abs(phi_new[ig] - Sol_phi[ig]);
                Sol_phi[ig] = alpha * phi_new[ig] + (1.0 - alpha) * Sol_phi[ig];
            }

            Parallel_Reduce::reduce_pool(diff);

           if (GlobalV::MY_RANK == 0) {
		      std::cout << "ITER " << iter << " Imp_Sol=" << mode 
			                   << " Diff=" << diff << " Ael=" << this->Ael << std::endl;
	   } 
            if(diff < sol_thr_val) break; 
        }

        delete[] phi_R_tmp;
        delete[] B_total;
        delete[] phi_new;
    }
    else // Mode 1: Linear
    {
        // Linear solve with kappa2 = nullptr
        minimize_cg(cell, rho_basis, epsilon, nullptr, B_elec, Sol_phi, ncgsol);
    }

    // =========================================================================
    // Post-Processing
    // =========================================================================

    // 1. Calculate Vacuum Reference (phi0)
    minimize_cg(cell, rho_basis, epsilon0, nullptr, B_elec, Sol_phi0, ncgsol);

    // 2. Prepare Real-space potentials
    double *phi_R_sol = new double[rho_basis->nrxx];
    double *phi_R_vac = new double[rho_basis->nrxx];
    rho_basis->recip2real(Sol_phi, phi_R_sol);
    rho_basis->recip2real(Sol_phi0, phi_R_vac);

    // Ensure rho_ion is consistent with final potential (for nonlinear modes)
    if(is_nonlinear) {
        double* k_dummy = new double[rho_basis->nrxx];
        cal_smpbe_physics(rho_basis->nrxx, phi_R_sol, epsilon, rho_ion_R, k_dummy);
        delete[] k_dummy;
    }

    if(this->phi_history.size() != rho_basis->npw) {
         this->phi_history.resize(rho_basis->npw);
    }
    // 保存当前收敛的解，供下一步 SCF 使用
    for(int i=0; i<rho_basis->npw; ++i) {
        this->phi_history[i] = Sol_phi[i];
    }

    // 3. Calculate Vel and Ael
    double *tmp_Vel = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(tmp_Vel, rho_basis->nrxx);
    
    // Ensure surchem class has members: delta_phi, epspot
    // If not, use local pointers:
    // double* delta_phi = new double[rho_basis->nrxx];
    // double* epspot = new double[rho_basis->nrxx];
    // Here we assume class members exist as per original code context.

    this->Ael = 0.0;
    for (int i = 0; i < rho_basis->nrxx; i++)
    {
        // Reaction Field Potential
        this->delta_phi[i] = phi_R_sol[i] - phi_R_vac[i];
        
        // Add to Effective Potential
        tmp_Vel[i] += this->delta_phi[i];

        // Energy Correction (Ael)
        // Ael = 0.5 * Integral [ (rho_elec + rho_ion) * phi_sol - rho_elec * phi_vac ]
        // Note: TOTN_real sign depends on ABACUS definition (usually rho_valence)
        double term_elec = TOTN_real[i] * this->delta_phi[i];
        double term_ion  = rho_ion_R[i] * phi_R_sol[i]; 
        
        this->Ael -= (term_elec + term_ion);
    }
    Parallel_Reduce::reduce_pool(this->Ael);
    this->Ael *= cell.omega / rho_basis->nxyz; // Linear Response Factor

    // 4. Calculate Non-electrostatic Potential (eps_pot)
    // NOTE: For imp_sol=3, we use the shape-based epsilon gradient for the force term
    // to maintain stability, even though epsilon itself is saturated.
    eps_pot(PS_TOTN_real, cell.tpiba, Sol_phi, rho_basis, epsilon, this->epspot);

    for (int i = 0; i < rho_basis->nrxx; i++)
    {
        tmp_Vel[i] += this->epspot[i];
    }

    // 5. Assign to Output Matrix Vel
    ModuleBase::GlobalFunc::ZEROS(Vel.c, nspin * rho_basis->nrxx);
    if (nspin == 4)
    {
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
            Vel(0, ir) += tmp_Vel[ir];
    }
    else
    {
        for (int is = 0; is < nspin; is++)
            for (int ir = 0; ir < rho_basis->nrxx; ir++)
                Vel(is, ir) += tmp_Vel[ir];
    }

    // =========================================================================
    // Cleanup
    // =========================================================================
    delete[] rho_ion_R;
    delete[] kappa2_R;
    delete[] shape_ref;
    
    delete[] PS_TOTN_real;
    delete[] B_elec;
    delete[] epsilon;
    delete[] epsilon0;
    
    delete[] Sol_phi;
    delete[] Sol_phi0;
    delete[] phi_R_sol;
    delete[] phi_R_vac;
    delete[] tmp_Vel;

    ModuleBase::timer::tick("surchem", "cal_vel");
    return Vel;
}
