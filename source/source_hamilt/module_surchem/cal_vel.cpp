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
             double* vwork,
             double delta_V_align) // <--- 【新增对齐参数】
{
    double *eprime_pure = new double[rho_basis->nrxx];
    double *eprime_diel = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(eprime_pure, rho_basis->nrxx);
    ModuleBase::GlobalFunc::ZEROS(eprime_diel, rho_basis->nrxx);

    shape_gradn(PS_TOTN_real, rho_basis, eprime_pure);

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        // eprime_diel 用于介电极化项，带有 (eb_k - 1)
        eprime_diel[ir] = eprime_pure[ir] * (PARAM.inp.eb_k - 1.0);
    }

    ModuleBase::Vector3<double> *nabla_phi = new ModuleBase::Vector3<double>[rho_basis->nrxx];
    double *phisq = new double[rho_basis->nrxx];

    XC_Functional::grad_rho(phi, nabla_phi, rho_basis, tpiba);

    double lambda_d_k = 3.0; // 3.0A 德拜长度
    double ekappa2_bulk = (lambda_d_k > 0.0) ? PARAM.inp.eb_k / (lambda_d_k * lambda_d_k) : 0.0;
    
    double *phi_real = new double[rho_basis->nrxx];
    rho_basis->recip2real(phi, phi_real);

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        phisq[ir] = pow(nabla_phi[ir].x, 2) + pow(nabla_phi[ir].y, 2) + pow(nabla_phi[ir].z, 2);
    }

    for (int ir = 0; ir < rho_basis->nrxx; ir++)
    {
        // 介电排斥压力，使用 eprime_diel
        vwork[ir] = eprime_diel[ir] * phisq[ir] / (8 * ModuleBase::PI);
        
        if (lambda_d_k > 0.0) {
            // 【核心修复】：必须使用对齐后的绝对电势，且使用纯粹的形函数导数 eprime_pure！
            double aligned_phi = phi_real[ir] - delta_V_align;
            vwork[ir] -= eprime_pure[ir] * ekappa2_bulk * aligned_phi * aligned_phi / (8 * ModuleBase::PI);
        }
    } 

    delete[] phi_real; 
    delete[] eprime_pure;
    delete[] eprime_diel;
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
    minimize_cg(cell, rho_basis, chi_tensor,nullptr, B, Sol_phi, ncgsol);

    ncgsol = 0;
    // Calculate Sol_phi0 with epsilon0.
    minimize_cg(cell, rho_basis, chi_tensor0, nullptr,B, Sol_phi0, ncgsol);

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
    eps_pot(PS_TOTN_real, cell.tpiba, Sol_phi, rho_basis, epsilon, epspot,0.0);

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
                                  double* ekappa2,
                                  std::complex<double>* rhs,
                                  double& rms)
{
    // =====================================================================
    // 1. 物理常数准备 (以常温纯水为例，后续建议移入 PARAM.inp 读取)
    // =====================================================================
// =====================================================================
    // 1. 物理常数输入 (外部常规单位：K, e*A, 1/A^3)
    // =====================================================================
    double SolTemp = 298.15;             // 温度 (K)
    double eb_k = PARAM.inp.eb_k;        // 体相介电常数 (如水: 78.4 或 80)
    double epsilon_inf = 1.78;           // 光学介电常数
    
    double p_mol_eA = 0.50;              // 溶剂偶极矩 (e*Angstrom)
    double n_mol_A3 = 0.0335;            // 溶剂摩尔密度 (1/Angstrom^3)
    double lambda_d_A = 3.0;             // 德拜长度 (Angstrom)

    // =====================================================================
    // 2. 核心单位转换：转换为 ABACUS 内部原子单位 (Rydberg, Bohr, e)
    // =====================================================================
    double BOHR = ModuleBase::BOHR_TO_A; // 1 Bohr ≈ 0.529177 A

    // 长度与体积转换 (转换为 Bohr)
    double p_mol = p_mol_eA / BOHR;           // e*Bohr
    double n_mol = n_mol_A3 * pow(BOHR, 3);   // 1/Bohr^3
    double lambda_d_k = lambda_d_A / BOHR;    // Bohr

    // 能量与温度转换 (转换为 Rydberg)
    // 玻尔兹曼常数 kb = 8.617333262e-5 eV/K
    // 1 Ry = 13.605698 eV (ModuleBase::Ry_to_eV)
    double kb_Ha = 8.617333262e-5 / (2.0 * ModuleBase::Ry_to_eV); 
    double invBETA = kb_Ha * SolTemp;        // 此时 kT 的单位严格为 Rydberg!

    // =====================================================================
    // 3. 物理衍生量计算 (单位现已完全匹配 ABACUS 底层)
    // =====================================================================
    double PBETA = p_mol / invBETA;
    double alpha0_rot = (1.0 / (4.0 * ModuleBase::PI)) * invBETA * pow(PBETA, 2) / 3.0; 
    double alpha_pol = alpha0_rot / (eb_k - epsilon_inf) * (epsilon_inf - 1.0);
    double invalpha_sic = ((eb_k - epsilon_inf) / alpha0_rot - n_mol) / (eb_k - 1.0);
    
    double ekappa2_bulk = 0.0;
    if (lambda_d_k > 0.0) {
        ekappa2_bulk = eb_k / (lambda_d_k * lambda_d_k); // 1/Bohr^2
    }

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
        if (ekappa2 != nullptr) {
            ekappa2[ir] = ekappa2_bulk * S_diel_val; 
        }

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
    Leps2(ucell, rho_basis, phi, chi, ekappa2 ,gradphi_G_work, lp, aux_grad_phi, aux_grad_grad_phi_real);

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

if (!this->is_phi_history_allocated || this->history_npw != rho_basis->npw) {
        
        // 先安全释放旧内存（如果有的话）
        if (this->is_phi_history_allocated) {
            delete[] this->phi_history;
            delete[] this->phi0_history;
        }
        
        // 按照最新的网格大小分配内存
        this->phi_history = new std::complex<double>[rho_basis->npw];
        this->phi0_history = new std::complex<double>[rho_basis->npw];
        ModuleBase::GlobalFunc::ZEROS(this->phi_history, rho_basis->npw);
        ModuleBase::GlobalFunc::ZEROS(this->phi0_history, rho_basis->npw);
        
        // 更新标记
        this->is_phi_history_allocated = true;
        this->history_npw = rho_basis->npw;
    }

    // 提取历史数组的指针参与本次计算
    std::complex<double> *Sol_phi = this->phi_history;
    std::complex<double> *Sol_phi0 = this->phi0_history;

    // 分配当前步修正量 dphi 和残差 rhs 的内存 (这两者不需要记忆，每次用新的即可)
    std::complex<double> *dphi = new std::complex<double>[rho_basis->npw];
    std::complex<double> *rhs = new std::complex<double>[rho_basis->npw];

    // ===============================================
    // 核心骨架：非线性牛顿迭代外层循环 (求解 Sol_phi)
    // ===============================================
    int iter = 0;
    double rms = 1.0;
    double old_rms = 100.0;
    double tol = 1e-5; // 外层收敛阈值
double *ekappa2 = new double[rho_basis->nrxx];
    ModuleBase::GlobalFunc::ZEROS(ekappa2, rho_basis->nrxx);
    std::cout << "Starting NLPCM Newton Iterations..." << std::endl;
    while (iter < 200 && rms > tol)
    {
        // 步骤 A: 喂入假物理，计算当前张量和残差 rhs
        update_nlpb_and_chi(cell, rho_basis, Sol_phi, B, epsilon, chi_tensor, ekappa2,rhs, rms);
        std::cout << "  Iter " << iter << ", RMS = " << rms << std::endl;

        if (rms <= tol) break;
        if (std::abs(rms - old_rms) / rms < 0.01) {
            std::cout << "  -> Newton iteration stagnated at grid limit. Exiting loop." << std::endl;
            break;
        }
        old_rms = rms; // 更新 old_rms
        double inner_tol = std::max(rms * 0.1, 1e-7);

        // 步骤 B: 调用内层线性求解器解 dphi
        int ncgsol_step = 0;
        minimize_cg_linear(cell, rho_basis, chi_tensor, ekappa2,rhs, dphi, ncgsol_step, inner_tol);
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
    // std::complex<double> *Sol_phi0 = new std::complex<double>[rho_basis->npw];
    double* chi_tensor0[3][3];
    for(int i=0; i<3; i++) for(int j=0; j<3; j++) { chi_tensor0[i][j] = new double[rho_basis->nrxx]; ModuleBase::GlobalFunc::ZEROS(chi_tensor0[i][j], rho_basis->nrxx); }
    for (int ir = 0; ir < rho_basis->nrxx; ir++) { chi_tensor0[0][0][ir] = epsilon0[ir]; chi_tensor0[1][1][ir] = epsilon0[ir]; chi_tensor0[2][2][ir] = epsilon0[ir]; }
    
    int ncgsol0 = 0;
    minimize_cg(cell, rho_basis, chi_tensor0, nullptr, B, Sol_phi0, ncgsol0);

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
// =====================================================================
    // 1. 物理常数输入 (外部常规单位：K, e*A, 1/A^3)
    // =====================================================================
    double SolTemp = 298.15;             // 温度 (K)
    double eb_k = PARAM.inp.eb_k;        // 体相介电常数 (如水: 78.4 或 80)
    double epsilon_inf = 1.78;           // 光学介电常数
    
    double p_mol_eA = 0.50;              // 溶剂偶极矩 (e*Angstrom)
    double n_mol_A3 = 0.0335;            // 溶剂摩尔密度 (1/Angstrom^3)
    double lambda_d_A = 3.0;             // 德拜长度 (Angstrom)

    // =====================================================================
    // 2. 核心单位转换：转换为 ABACUS 内部原子单位 (Rydberg, Bohr, e)
    // =====================================================================
    double BOHR = ModuleBase::BOHR_TO_A; // 1 Bohr ≈ 0.529177 A

    // 长度与体积转换 (转换为 Bohr)
    double p_mol = p_mol_eA / BOHR;           // e*Bohr
    double n_mol = n_mol_A3 * pow(BOHR, 3);   // 1/Bohr^3
    double lambda_d_k = lambda_d_A / BOHR;    // Bohr

    // 能量与温度转换 (转换为 Rydberg)
    // 玻尔兹曼常数 kb = 8.617333262e-5 eV/K
    // 1 Ry = 13.605698 eV (ModuleBase::Ry_to_eV)
    double kb_Ha = 8.617333262e-5 / (2.0 * ModuleBase::Ry_to_eV); 
    double invBETA = kb_Ha * SolTemp;        // 此时 kT 的单位严格为 Rydberg!

    // =====================================================================
    // 3. 物理衍生量计算 (单位现已完全匹配 ABACUS 底层)
    // =====================================================================
    double PBETA = p_mol / invBETA;
    double alpha0_rot = (1.0 / (4.0 * ModuleBase::PI)) * invBETA * pow(PBETA, 2) / 3.0; 
    double alpha_pol = alpha0_rot / (eb_k - epsilon_inf) * (epsilon_inf - 1.0);
    double invalpha_sic = ((eb_k - epsilon_inf) / alpha0_rot - n_mol) / (eb_k - 1.0);
    
    double ekappa2_bulk = 0.0;
    if (lambda_d_k > 0.0) {
        ekappa2_bulk = eb_k / (lambda_d_k * lambda_d_k); // 1/Bohr^2
    }
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
// 旋转极化做功
        double W_rot = n_mol * S_diel_val * invBETA * ln_sinh_X_div_X;
        double W_pol = 0.5 * n_mol * alpha_pol * S_diel_val * (f_loc * E_norm) * (f_loc * E_norm);

        // 【核心修复 1】：必须使用对齐后的电势来计算离子功
        double aligned_phi = phi_tilda_R[ir] - delta_V_align;
        double W_ion = 0.0;
        if (lambda_d_k > 0.0) {
            double ekappa2_bulk = PARAM.inp.eb_k / (lambda_d_k * lambda_d_k);
            W_ion = (ekappa2_bulk * S_diel_val / (8.0 * ModuleBase::PI)) * aligned_phi * aligned_phi;
        }

        // 【核心修复 2】：使用对齐后的电势计算静电做功 (d_rho * phi_aligned)
        double d_rho = TOTN_real[ir]; 
        double U_sol = d_rho * aligned_phi - (1.0 / (8.0 * ModuleBase::PI)) * (E_norm * E_norm) - W_rot - W_pol - W_ion;
        double U_vac = d_rho * phi_tilda_R0[ir] - (1.0 / (8.0 * ModuleBase::PI)) * E0_norm2;

        Ael_local += (U_sol - U_vac);
    }

    // 并行归约求和并乘以网格体积
    Parallel_Reduce::reduce_pool(Ael_local);
    // 【核心修复 3】：Ael_local 是 Hartree 单位，必须乘以 2.0 转换为 ABACUS 全局需要的 Rydberg 单位！
    this->Ael = 2.0 * Ael_local * cell.omega / rho_basis->nxyz;

    delete[] E_field;
    delete[] E0_field;
    
    // 【核心修复 4】：传入 delta_V_align 给 eps_pot
    eps_pot(PS_TOTN_real, cell.tpiba, Sol_phi, rho_basis, epsilon, epspot, delta_V_align);

    for (int i = 0; i < rho_basis->nrxx; i++) { 
        tmp_Vel[i] += epspot[i]; 
        // 这一步乘以 2.0 将势能从 Hartree 正确转换为 Rydberg，此逻辑完美！
        tmp_Vel[i] *= 2.0; 
    }

    ModuleBase::GlobalFunc::ZEROS(Vel.c, nspin * rho_basis->nrxx);
    if (nspin == 4) { for (int ir = 0; ir < rho_basis->nrxx; ir++) { Vel(0, ir) += tmp_Vel[ir]; v(0, ir) += Vel(0, ir); } }
    else { for (int is = 0; is < nspin; is++) { for (int ir = 0; ir < rho_basis->nrxx; ir++) { Vel(is, ir) += tmp_Vel[ir]; v(is, ir) += Vel(is, ir); } } }

    // 释放所有内存...
    delete[] B; delete[] PS_TOTN_real; delete[] epsilon; delete[] epsilon0; delete[] dphi; delete[] rhs; delete[] phi_tilda_R; delete[] phi_tilda_R0; delete[] tmp_Vel;
    delete[] ekappa2;
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) { delete[] chi_tensor[i][j]; delete[] chi_tensor0[i][j]; } }

     ModuleBase::timer::tick("surchem", "cal_vel");
}