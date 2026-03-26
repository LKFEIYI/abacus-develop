#include "source_base/timer.h"
#include "source_hamilt/module_xc/xc_functional.h"
#include "source_io/module_parameter/parameter.h"
#include "surchem.h"

void shape_gradn(const double* PS_TOTN_real, const ModulePW::PW_Basis* rho_basis, double* eprime)
{

    double epr_c = 1.0 / sqrt(ModuleBase::TWO_PI) / PARAM.inp.sigma_k;
    double epr_z = 0;
    double min = 1e-10;
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        epr_z = log(std::max(PS_TOTN_real[ir], min) / PARAM.inp.nc_k) / sqrt(2) / PARAM.inp.sigma_k;
        eprime[ir] = epr_c * exp(-pow(epr_z, 2)) / std::max(PS_TOTN_real[ir], min);
    }
}

void eps_pot(const double* PS_TOTN_real,
             const double& tpiba,
             const std::complex<double>* phi,
             const ModulePW::PW_Basis* rho_basis,
             double* d_eps,
             double* vwork)
{
    double *eprime = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(eprime, rho_basis->nrxx);

    shape_gradn(PS_TOTN_real, rho_basis, eprime);

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        eprime[ir] = eprime[ir] * (PARAM.inp.eb_k - 1);
    }

    ModuleBase::Vector3<double> *nabla_phi = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    double *phisq = new double[rho_basis->nrxx];

    // nabla phi
    XC_Functional::grad_rho(phi, nabla_phi, rho_basis, tpiba);

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        phisq[ir] = pow(nabla_phi[ir].x, 2) + pow(nabla_phi[ir].y, 2) + pow(nabla_phi[ir].z, 2);
    }

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        vwork[ir] = eprime[ir] * phisq[ir] / (8 * ModuleBase::PI);
    }

    delete[] eprime;
    delete[] nabla_phi;
    delete[] phisq;
}

//The interface is changed to use an explicit output parameter to
//clarify lifetime management and avoid hidden allocations.
void surchem::cal_vel(const UnitCell& cell,
                      const ModulePW::PW_Basis* rho_basis,
                      std::complex<double>* TOTN,
                      std::complex<double>* PS_TOTN,
                      int nspin,
                      ModuleBase::matrix& v)
{
    ModuleBase::TITLE("surchem", "cal_vel");
    ModuleBase::timer::tick("surchem", "cal_vel");

    rho_basis->recip2real(TOTN, TOTN_real);

    // -4pi * TOTN(G)
    std::complex<double> *B = new std::complex<double>[rho_basis->npw];
    for (int ig = 0; ig < rho_basis->npw; ig++)
    {
        B[ig] = -4.0 * ModuleBase::PI * TOTN[ig];
    }

    // Build a nrxx vector to DO FFT .
    double *PS_TOTN_real = new double[rho_basis->nrxx];
    rho_basis->recip2real(PS_TOTN, PS_TOTN_real);

    // build epsilon in real space (nrxx)
    double *epsilon = new double[rho_basis->nrxx];
    double *epsilon0 = new double[rho_basis->nrxx];
     cal_epsilon(rho_basis, PS_TOTN_real, epsilon, epsilon0);

    double* chi_tensor[3][3];
    double* chi_tensor0[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            chi_tensor[i][j] = new double[rho_basis->nrxx];
            chi_tensor0[i][j] = new double[rho_basis->nrxx];
        
        // 初始化全 0
            ModuleBase::GlobalFunc::ZEROS(chi_tensor[i][j], rho_basis->nrxx);
            ModuleBase::GlobalFunc::ZEROS(chi_tensor0[i][j], rho_basis->nrxx);
        }
    }
    for (int ir = 0; ir < rho_basis->nrxx; ir++) {
    chi_tensor[0][0][ir] = epsilon[ir];
    chi_tensor[1][1][ir] = epsilon[ir];
    chi_tensor[2][2][ir] = epsilon[ir];

    chi_tensor0[0][0][ir] = epsilon0[ir];
    chi_tensor0[1][1][ir] = epsilon0[ir];
    chi_tensor0[2][2][ir] = epsilon0[ir];
    }

   

    std::complex<double> *Sol_phi = new std::complex<double>[rho_basis->npw];
    std::complex<double> *Sol_phi0 = new std::complex<double>[rho_basis->npw];
    int ncgsol = 0;

    double *tmp_Vel = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(tmp_Vel, rho_basis->nrxx);

    // Calculate Sol_phi with epsilon.
    ncgsol = 0;
    minimize_cg(cell, rho_basis, chi_tensor, B, Sol_phi, ncgsol);

    ncgsol = 0;
    // Calculate Sol_phi0 with epsilon0.
    minimize_cg(cell, rho_basis, chi_tensor0, B, Sol_phi0, ncgsol);

    double *phi_tilda_R = new double[rho_basis->nrxx];
    double *phi_tilda_R0 = new double[rho_basis->nrxx];

    rho_basis->recip2real(Sol_phi, phi_tilda_R);
    rho_basis->recip2real(Sol_phi0, phi_tilda_R0);

    // the 1st item of tmp_Vel
    for (int i = 0; i < rho_basis->nrxx; i++)
    {
        delta_phi[i] = phi_tilda_R[i] - phi_tilda_R0[i];
        tmp_Vel[i] += delta_phi[i];
    }

    // calculate Ael
    this->Ael = 0.0;
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        this->Ael -= TOTN_real[ir] * delta_phi[ir];
    }
    Parallel_Reduce::reduce_pool(this->Ael);
    this->Ael *= cell.omega / rho_basis->nxyz;

    // the 2nd item of tmp_Vel
    eps_pot(PS_TOTN_real, cell.tpiba, Sol_phi, rho_basis, epsilon, epspot);

    for (int i = 0; i < rho_basis->nrxx; i++)
    {
        tmp_Vel[i] += epspot[i];
    }

    // ModuleBase::matrix v(nspin, rho_basis->nrxx);
    ModuleBase::GlobalFunc::ZEROS(Vel.c, nspin * rho_basis->nrxx);

    if (nspin == 4)
    {
        for (int ir = 0; ir < rho_basis->nrxx; ir++)
        {
            Vel(0, ir) += tmp_Vel[ir];
            v(0, ir) += Vel(0, ir);
        }
    }
    else
    {
        for (int is = 0; is < nspin; is++)
        {
            for (int ir = 0; ir < rho_basis->nrxx; ir++)
            {
                Vel(is, ir) += tmp_Vel[ir];
                v(is, ir) += Vel(is, ir);
            }
        }
    }

    delete[] PS_TOTN_real;
    delete[] Sol_phi;
    delete[] Sol_phi0;
    delete[] B;
    delete[] epsilon;
    delete[] epsilon0;
    delete[] tmp_Vel;
    delete[] phi_tilda_R;
    delete[] phi_tilda_R0;

    for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
        delete[] chi_tensor[i][j];
        delete[] chi_tensor0[i][j];
    }
}

    ModuleBase::timer::tick("surchem", "cal_vel");
    return;
}

void surchem::update_nlpb_and_chi(const UnitCell& ucell,
                                  const ModulePW::PW_Basis* rho_basis,
                                  std::complex<double>* phi,
                                  const std::complex<double>* B,
                                  const double* epsilon_linear, // 传入第一步算好的线性 epsilon 用于提取空腔形状
                                  double* chi[3][3],
                                  std::complex<double>* rhs,
                                  double& rms)
{
    // =====================================================================
    // 1. 物理常数准备 (以常温纯水为例，后续建议移入 PARAM.inp 读取)
    // =====================================================================
    double SolTemp = 298.15;
    double invBETA = 8.617333262e-5 * SolTemp; // kb * T (eV)
    double eb_k = PARAM.inp.eb_k;              // 体相介电常数 (水: 78.4)
    double epsilon_inf = 1.78;                 // 光学介电常数
    double n_mol = 0.0335;                     // 溶剂摩尔密度 (1/A^3)
    double p_mol = 0.50;                       // 溶剂偶极矩 (e*A)

    double PBETA = p_mol / invBETA;
    double alpha0_rot = (1.0 / (4.0 * ModuleBase::PI)) * invBETA * pow(PBETA, 2) / 3.0; 
    double alpha_pol = alpha0_rot / (eb_k - epsilon_inf) * (epsilon_inf - 1.0);
    double invalpha_sic = ((eb_k - epsilon_inf) / alpha0_rot - n_mol) / (eb_k - 1.0);

    // =====================================================================
    // 2. 计算宏观电场 E = -grad(phi)
    // =====================================================================
    ModuleBase::Vector3<double> *E_field = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    XC_Functional::grad_rho(phi, E_field, rho_basis, ucell.tpiba);
    for (int ir = 0; ir < rho_basis->nrxx; ir++) {
        E_field[ir].x = -E_field[ir].x;
        E_field[ir].y = -E_field[ir].y;
        E_field[ir].z = -E_field[ir].z;
    }

    // =====================================================================
    // 3. 计算非线性介电张量 chi_tensor (核心物理)
    // =====================================================================
    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        double Ex = E_field[ir].x;
        double Ey = E_field[ir].y;
        double Ez = E_field[ir].z;
        double E_norm = sqrt(Ex*Ex + Ey*Ey + Ez*Ez + 1e-20); // 1e-20 防止除零

        // 渐进策略：暂时令局部场 E_loc = 宏观场 E_norm
        // -----------------------------------------------------------
        // 准备计算局域场 E_loc = f_loc * E_norm
        // -----------------------------------------------------------
        double f_loc = 1.0; 
        
        // f_loc 的理论下界和上界
        double f_min = 1.0 / (1.0 - alpha_pol * invalpha_sic);
        double f_max = 1.0 / (1.0 - (alpha_pol + alpha0_rot) * invalpha_sic);
        
        // 只有当宏观电场足够大时，才需要求解非线性的 f_loc
        if (E_norm * PBETA > 1e-4) {
            double low = f_min;
            double high = f_max * (1.0 + 1e-6); // 略微放大上界确保包围根
            
            // 定义一个 Lambda 表达式计算超越方程的残差 F(f_loc)
            auto calc_F = [&](double f) {
                double x_loc = f * PBETA * E_norm;
                double g_rot = 1.0;
                if (x_loc > 2e-4) {
                    g_rot = 3.0 * (x_loc - tanh(x_loc)) / (x_loc * x_loc * tanh(x_loc));
                }
                return 1.0 / (1.0 - (g_rot * alpha0_rot + alpha_pol) * invalpha_sic) - f;
            };

            double F_low = calc_F(low);
            
            // 快速二分法求根 (通常 15 步左右即可收敛到双精度极限)
            for (int iter_f = 0; iter_f < 30; iter_f++) {
                double mid = 0.5 * (low + high);
                double F_mid = calc_F(mid);
                
                if (std::abs(F_mid) < 1e-7) {
                    f_loc = mid;
                    break;
                }
                if (F_low * F_mid < 0) {
                    high = mid;
                } else {
                    low = mid;
                    F_low = F_mid;
                }
            }
            f_loc = 0.5 * (low + high);
        } else {
            f_loc = f_max; // 弱场下直接等于线性极化极限
        }

        // -----------------------------------------------------------
        // 更新 X (使用真实的局域场)
        // -----------------------------------------------------------
        double X = f_loc * PBETA * E_norm;
        
        double chi_par = 0.0;
        double chi_perp = 0.0;

        // 计算 Langevin 函数的导数响应
        if (X < 2e-4) {
            // 弱场极限 (线性区)，使用泰勒展开防止数值灾难
            chi_par = 1.0;
            chi_perp = 1.0;
        } else if (X > 100.0) {
            // 强场极限 (完全饱和区)
            chi_par = 3.0 * (1.0 / (X * X));
            chi_perp = 3.0 * (1.0 - 1.0 / X) / X;
        } else {
            // 正常非线性区
            double sinh_X = sinh(X);
            double tanh_X = tanh(X);
            chi_par = 3.0 * (1.0 / (X * X) - 1.0 / (sinh_X * sinh_X));
            chi_perp = 3.0 * (1.0 / tanh_X - 1.0 / X) / X;
        }

        // 组合电子极化和取向极化
        chi_par = alpha_pol + alpha0_rot * chi_par;
        chi_perp = alpha_pol + alpha0_rot * chi_perp;

        // 【极其重要】：从 epsilon_linear 中反推提取空腔形状 S_diel (0到1之间)
        // 只有在溶剂存在的区域，才会有非线性极化
        double S_diel_val = (epsilon_linear[ir] - 1.0) / (eb_k - 1.0);
        // 确保 S_diel_val 在 [0, 1] 之间，防止数值舍入误差
        S_diel_val = std::max(0.0, std::min(1.0, S_diel_val)); 

        chi_par = n_mol * S_diel_val / (1.0 / chi_par - invalpha_sic);
        chi_perp = n_mol * S_diel_val / (1.0 / chi_perp - invalpha_sic);

        // 组装 3x3 笛卡尔响应张量
        double inv_E2 = 1.0 / (E_norm * E_norm);
        double delta_chi = chi_par - chi_perp;

        // 计算张量元素并加上真空介电常数 1.0
        chi[0][0][ir] = delta_chi * (Ex * Ex * inv_E2) + chi_perp + 1.0;
        chi[0][1][ir] = delta_chi * (Ex * Ey * inv_E2);
        chi[0][2][ir] = delta_chi * (Ex * Ez * inv_E2);
        
        chi[1][0][ir] = chi[0][1][ir];
        chi[1][1][ir] = delta_chi * (Ey * Ey * inv_E2) + chi_perp + 1.0;
        chi[1][2][ir] = delta_chi * (Ey * Ez * inv_E2);
        
        chi[2][0][ir] = chi[0][2][ir];
        chi[2][1][ir] = chi[1][2][ir];
        chi[2][2][ir] = delta_chi * (Ez * Ez * inv_E2) + chi_perp + 1.0;
    }

    // =====================================================================
    // 4. 计算非线性残差: rhs = L(phi) + B
    // =====================================================================
    std::complex<double> *lp = new std::complex<double>[rho_basis->npw];
    std::complex<double> *gradphi_G_work = new std::complex<double>[rho_basis->npw];
    ModuleBase::Vector3<double> *aux_grad_phi = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    double *aux_grad_grad_phi_real = new double[rho_basis->nrxx];
    
    ModuleBase::GlobalFunc::ZEROS(lp, rho_basis->npw);

    // 调用我们在第一步改造好的 3x3 张量算子
    Leps2(ucell, rho_basis, phi, chi, gradphi_G_work, lp, aux_grad_phi, aux_grad_grad_phi_real);

    double r2 = 0;
    const int ig0 = rho_basis->ig_gge0;
    for (int ig = 0; ig < rho_basis->npw; ig++) {
        if(ig == ig0) {
            rhs[ig] = 0;
            continue;
        }
        rhs[ig] = lp[ig] + B[ig]; // RHS = \nabla(\chi \nabla \phi) - 4\pi\rho
        r2 += rhs[ig].real()*rhs[ig].real() + rhs[ig].imag()*rhs[ig].imag();
    }
    Parallel_Reduce::reduce_pool(r2);
    rms = sqrt(r2);

    // 释放内存
    delete[] E_field;
    delete[] lp; delete[] gradphi_G_work; delete[] aux_grad_phi; delete[] aux_grad_grad_phi_real;
}

void surchem::cal_vel_nlpcm(const UnitCell& cell,
                            const ModulePW::PW_Basis* rho_basis,
                            std::complex<double>* TOTN,
                            std::complex<double>* PS_TOTN,
                            int nspin,
                            ModuleBase::matrix& v)
{
    ModuleBase::TITLE("surchem", "cal_vel_nlpcm");
    ModuleBase::timer::tick("surchem", "cal_vel_nlpcm");

    // ===== 准备工作 (和原来一样) =====
    rho_basis->recip2real(TOTN, TOTN_real);
    std::complex<double> *B = new std::complex<double>[rho_basis->npw];
    for (int ig = 0; ig < rho_basis->npw; ig++) { B[ig] = -4.0 * ModuleBase::PI * TOTN[ig]; }

    double *PS_TOTN_real = new double[rho_basis->nrxx];
    rho_basis->recip2real(PS_TOTN, PS_TOTN_real);

    double *epsilon = new double[rho_basis->nrxx];
    double *epsilon0 = new double[rho_basis->nrxx];
    cal_epsilon(rho_basis, PS_TOTN_real, epsilon, epsilon0);

    double* chi_tensor[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            chi_tensor[i][j] = new double[rho_basis->nrxx];
        }
    }

    std::complex<double> *Sol_phi = new std::complex<double>[rho_basis->npw];
    std::complex<double> *dphi = new std::complex<double>[rho_basis->npw];
    std::complex<double> *rhs = new std::complex<double>[rho_basis->npw];
    ModuleBase::GlobalFunc::ZEROS(Sol_phi, rho_basis->npw); // 电势初猜为 0

    // ===============================================
    // 核心骨架：非线性牛顿迭代外层循环 (求解 Sol_phi)
    // ===============================================
    int iter = 0;
    double rms = 1.0;
    double tol = 1e-5; // 外层收敛阈值

    std::cout << "Starting NLPCM Newton Iterations..." << std::endl;
    while (iter < 20 && rms > tol)
    {
        // 步骤 A: 喂入假物理，计算当前张量和残差 rhs
        update_nlpb_and_chi(cell, rho_basis, Sol_phi, B, epsilon, chi_tensor, rhs, rms);
        std::cout << "  Iter " << iter << ", RMS = " << rms << std::endl;

        if (rms <= tol) break;
        double inner_tol = std::max(rms * 0.1, 1e-7);

        // 步骤 B: 调用内层线性求解器解 dphi
        int ncgsol_step = 0;
        minimize_cg_linear(cell, rho_basis, chi_tensor, rhs, dphi, ncgsol_step, inner_tol);
        std::cout << "    Inner CG steps: " << ncgsol_step << " (tol: " << inner_tol << ")" << std::endl;

        // 步骤 C: 更新电势 Phi_new = Phi_old + dphi
        for (int ig = 0; ig < rho_basis->npw; ig++) {
            Sol_phi[ig] += dphi[ig];
        }
        iter++;
    }

    // ===============================================
    // (真空相 Sol_phi0 这里用原来的老线性方法求解即可，因为真空永远是线性的)
    // ===============================================
    std::complex<double> *Sol_phi0 = new std::complex<double>[rho_basis->npw];
    double* chi_tensor0[3][3];
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) { chi_tensor0[i][j] = new double[rho_basis->nrxx]; ModuleBase::GlobalFunc::ZEROS(chi_tensor0[i][j], rho_basis->nrxx); }
    for (int ir = 0; ir < rho_basis->nrxx; ir++) { chi_tensor0[0][0][ir] = epsilon0[ir]; chi_tensor0[1][1][ir] = epsilon0[ir]; chi_tensor0[2][2][ir] = epsilon0[ir]; }
    
    int ncgsol0 = 0;
    minimize_cg(cell, rho_basis, chi_tensor0, B, Sol_phi0, ncgsol0);

    // ===============================================
    // 计算能量与势 (和原逻辑一致)
    // ===============================================
    double *phi_tilda_R = new double[rho_basis->nrxx];
    double *phi_tilda_R0 = new double[rho_basis->nrxx];
    rho_basis->recip2real(Sol_phi, phi_tilda_R);
    rho_basis->recip2real(Sol_phi0, phi_tilda_R0);

    double *tmp_Vel = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(tmp_Vel, rho_basis->nrxx);
    
    // =====================================================================
    // 终极步骤：计算非线性溶剂化自由能 Ael (替换原有的 Ael 计算逻辑)
    // =====================================================================

    // 1. 获取溶剂相电场 E 和 真空相电场 E0
    ModuleBase::Vector3<double> *E_field = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    ModuleBase::Vector3<double> *E0_field = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    
    XC_Functional::grad_rho(Sol_phi, E_field, rho_basis, cell.tpiba);
    XC_Functional::grad_rho(Sol_phi0, E0_field, rho_basis, cell.tpiba);

    // 常数准备 (保持与 update_nlpb_and_chi 中一致)
    double SolTemp = 298.15;
    double invBETA = 8.617333262e-5 * SolTemp; 
    double eb_k = PARAM.inp.eb_k;              
    double epsilon_inf = 1.78;                 
    double n_mol = 0.0335;                     
    double p_mol = 0.50;                       
    double PBETA = p_mol / invBETA;
    double alpha0_rot = (1.0 / (4.0 * ModuleBase::PI)) * invBETA * pow(PBETA, 2) / 3.0; 
    double alpha_pol = alpha0_rot / (eb_k - epsilon_inf) * (epsilon_inf - 1.0);
    double invalpha_sic = ((eb_k - epsilon_inf) / alpha0_rot - n_mol) / (eb_k - 1.0);
    // =====================================================================
    // 新增：体相电势对齐 (Potential Alignment)
    // =====================================================================
    double sum_delta_phi_bulk = 0.0;
    double sum_weight_bulk = 0.0;

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        // 提取空腔函数 S_diel (0 表示真空/溶质内部，1 表示体相溶剂)
        double S_diel_val = (epsilon[ir] - 1.0) / (PARAM.inp.eb_k - 1.0);
        S_diel_val = std::max(0.0, std::min(1.0, S_diel_val));

        // 为了严格只选取“深处”的体相溶剂，我们对 S_diel 取一个高次方（比如 10 次方）
        // 这样只有 S_diel 极其接近 1.0 的区域，weight 才接近 1，否则迅速衰减到 0
        double weight = pow(S_diel_val, 10.0);

        // 我们要测量的，是溶剂态电势和真空态电势在远离溶质时的差值
        double dphi_val = phi_tilda_R[ir] - phi_tilda_R0[ir];

        sum_delta_phi_bulk += dphi_val * weight;
        sum_weight_bulk += weight;
    }

    // 并行归约，将所有 MPI 进程的 sum_delta_phi_bulk 和 sum_weight_bulk 加起来
    Parallel_Reduce::reduce_pool(sum_delta_phi_bulk);
    Parallel_Reduce::reduce_pool(sum_weight_bulk);

    double delta_V_align = 0.0;
    if (sum_weight_bulk > 1e-8) {
        // 算出平均的基准电势偏移量
        delta_V_align = sum_delta_phi_bulk / sum_weight_bulk;
    }

    if (GlobalV::MY_RANK == 0) {
        std::cout << "  Potential Alignment Shift (delta_V): " << delta_V_align << " Hartree/e" << std::endl;
    }

    // =====================================================================
    // 终极步骤：计算非线性溶剂化自由能 Ael (保持您上次写的 Langevin 积分代码)
    // =====================================================================
    this->Ael = 0.0;
    double Ael_local = 0.0;
    // ... [准备 E_field, E0_field, alpha_pol 等常数的代码] ...

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        // 【核心修改点】：在计算静电势差时，减去我们在体相中探测到的基准偏移量！
        delta_phi[ir] = (phi_tilda_R[ir] - phi_tilda_R0[ir]) - delta_V_align;
        
        // 叠加到传入给 SCF 循环的势能中
        tmp_Vel[ir] += delta_phi[ir];

        // --- 开始计算非线性自由能密度 ---

        double Ex = -E_field[ir].x; double Ey = -E_field[ir].y; double Ez = -E_field[ir].z;
        double E_norm = sqrt(Ex*Ex + Ey*Ey + Ez*Ez + 1e-20);
        
        double E0x = -E0_field[ir].x; double E0y = -E0_field[ir].y; double E0z = -E0_field[ir].z;
        double E0_norm2 = E0x*E0x + E0y*E0y + E0z*E0z;

        // 提取空腔函数 S_diel
        double S_diel_val = (epsilon[ir] - 1.0) / (eb_k - 1.0);
        S_diel_val = std::max(0.0, std::min(1.0, S_diel_val));

        // 重新计算局域场因子 f_loc (可以直接复用你之前写的二分法代码，这里为了简略用 f_loc 表示)
        // [请将你上一步写的 f_loc 求根代码复制到这里，算出当前网格点的 f_loc]
        double f_loc = 1.0; 
        // ... (你的二分法代码) ...

        double X = f_loc * PBETA * E_norm;

        // 计算 Langevin 积分项: ln(sinh(X) / X)
        // 【注意】：必须使用安全近似，防止 X 过大时 sinh(X) 导致 double 溢出！
        double ln_sinh_X_div_X = 0.0;
        if (X < 1e-4) {
            ln_sinh_X_div_X = X * X / 6.0; // 弱场泰勒展开
        } else if (X > 50.0) {
            ln_sinh_X_div_X = X - log(2.0 * X); // 强场渐进展开
        } else {
            ln_sinh_X_div_X = log(sinh(X) / X); // 正常区
        }

        // 旋转极化做功
        double W_rot = n_mol * S_diel_val * invBETA * ln_sinh_X_div_X;
        
        // 电子线性极化做功
        double W_pol = 0.5 * alpha_pol * S_diel_val * (f_loc * E_norm) * (f_loc * E_norm);

        // 总静电自由能密度差 = 溶剂化后的静电能 - 真空下的静电能 - 极化做功
        double U_sol = (1.0 / (8.0 * ModuleBase::PI)) * (E_norm * E_norm) - W_rot - W_pol;
        double U_vac = (1.0 / (8.0 * ModuleBase::PI)) * E0_norm2;

        Ael_local += (U_sol - U_vac);
    }

    // 并行归约求和并乘以网格体积
    Parallel_Reduce::reduce_pool(Ael_local);
    this->Ael = Ael_local * cell.omega / rho_basis->nxyz;

    // 清理内存
    delete[] E_field;
    delete[] E0_field;
    
    // ... [后面接着计算 eps_pot 并叠加到 v 的原有代码] ...

    eps_pot(PS_TOTN_real, cell.tpiba, Sol_phi, rho_basis, epsilon, epspot);
    for (int i = 0; i < rho_basis->nrxx; i++) { tmp_Vel[i] += epspot[i]; }

    ModuleBase::GlobalFunc::ZEROS(Vel.c, nspin * rho_basis->nrxx);
    if (nspin == 4) { for (int ir = 0; ir < rho_basis->nrxx; ir++) { Vel(0, ir) += tmp_Vel[ir]; v(0, ir) += Vel(0, ir); } }
    else { for (int is = 0; is < nspin; is++) { for (int ir = 0; ir < rho_basis->nrxx; ir++) { Vel(is, ir) += tmp_Vel[ir]; v(is, ir) += Vel(is, ir); } } }

    // 释放所有内存...
    delete[] B; delete[] PS_TOTN_real; delete[] epsilon; delete[] epsilon0; delete[] Sol_phi; delete[] Sol_phi0; delete[] dphi; delete[] rhs; delete[] phi_tilda_R; delete[] phi_tilda_R0; delete[] tmp_Vel;
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { delete[] chi_tensor[i][j]; delete[] chi_tensor0[i][j]; } }

     ModuleBase::timer::tick("surchem", "cal_vel");
}