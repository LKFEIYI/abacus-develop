#include "surchem.h"
#include <fstream>
#include <iomanip>

double surchem::Acav = 0;
double surchem::Ael = 0;

surchem::surchem()
{
    TOTN_real = nullptr;
    delta_phi = nullptr;
    epspot = nullptr;
    Vcav = ModuleBase::matrix();
    Vel = ModuleBase::matrix();
    qs = 0;
}

void surchem::allocate(const int &nrxx, const int &nspin)
{
    assert(nrxx >= 0);
    assert(nspin > 0);

    delete[] TOTN_real;
    delete[] delta_phi;
    delete[] epspot;
    if (nrxx > 0)
    {
        TOTN_real = new double[nrxx];
        delta_phi = new double[nrxx];
        epspot = new double[nrxx];
    }
    else
    {
        TOTN_real = nullptr;
        delta_phi = nullptr;
        epspot = nullptr;
    }
    Vcav.create(nspin, nrxx);
    Vel.create(nspin, nrxx);

    ModuleBase::GlobalFunc::ZEROS(delta_phi, nrxx);
    ModuleBase::GlobalFunc::ZEROS(TOTN_real, nrxx);
    ModuleBase::GlobalFunc::ZEROS(epspot, nrxx);
    return;
}


void surchem::test_smpbe_driver(const UnitCell& cell, const ModulePW::PW_Basis* rho_basis)
{
    if(ModuleBase::GlobalFunc::MY_RANK != 0) return; // 只在主核运行打印
    
    std::cout << "\n========================================================" << std::endl;
    std::cout << "  STARTING VASPsol++ (SMPBE) DIAGNOSTIC TEST " << std::endl;
    std::cout << "  Model: Gaussian Charge in Uniform Electrolyte" << std::endl;
    std::cout << "========================================================" << std::endl;

    int nrxx = rho_basis->nrxx;
    int npw = rho_basis->npw;

    // 1. 构造人工电荷密度 (Gaussian Charge)
    // rho(r) = Q * (alpha/pi)^1.5 * exp(-alpha * r^2)
    // 放在晶胞中心
    std::complex<double>* mock_TOTN = new std::complex<double>[npw];
    double* mock_TOTN_R = new double[nrxx];
    ModuleBase::GlobalFunc::ZEROS(mock_TOTN_R, nrxx);

    ModuleBase::Vector3<double> center(0.5, 0.5, 0.5); // 晶胞中心 (Direct coordinates)
    // 转换为笛卡尔坐标需要用到 latvec，这里简化假设正交或者直接在实空间操作
    // 为了简单，我们直接在实空间造一个高斯包
    
    double alpha = 2.0; // 高斯宽度参数
    double Q = -1.0;    // 电子电荷 (负值)
    double sigma = 0.5; // Bohr
    
    // 寻找中心点索引 (近似)
    // 注意：这里仅作粗略测试，不处理跨边界 PBC 距离，假设盒子足够大
    double L = cell.lat0; // Bohr
    double center_pos = L / 2.0;

    for(int i=0; i<nrxx; ++i) {
        ModuleBase::Vector3<double> pos = rho_basis->get_r(i); // 获取格点坐标 (Cartesian in Bohr)
        // 简单的最小镜像距离
        double dx = pos.x - center_pos;
        double dy = pos.y - center_pos;
        double dz = pos.z - center_pos;
        double r2 = dx*dx + dy*dy + dz*dz;
        
        mock_TOTN_R[i] = Q * pow(1.0/(sigma*sqrt(ModuleBase::PI)), 3) * exp(-r2 / (sigma*sigma));
    }
    
    // 2. 构造均匀介电背景 (Epsilon = 78.4)
    // 这样可以测试纯 Poisson-Boltzmann 物理，排除界面形状函数的干扰
    double* mock_eps = new double[nrxx];
    for(int i=0; i<nrxx; ++i) mock_eps[i] = 78.4; 

    // 3. 运行线性求解器 (Linear VASPsol)
    std::complex<double>* phi_linear = new std::complex<double>[npw];
    std::complex<double>* mock_B = new std::complex<double>[npw];
    
    // FFT charge to G space
    rho_basis->real2recip(mock_TOTN_R, mock_TOTN);
    for(int ig=0; ig<npw; ++ig) mock_B[ig] = -4.0 * ModuleBase::PI * mock_TOTN[ig];

    int ncg = 0;
    // 传入 nullptr 表示线性 (kappa=0)
    // 注意：minimize_cg 内部会因为 kappa=0 而不做 Debye 预处理，
    // 对于带净电荷的体系，minimize_cg 可能在 G=0 处发散。
    // 为了测试，我们手动给一个极小的 kappa 
    double* small_kappa = new double[nrxx];
    for(int i=0; i<nrxx; ++i) small_kappa[i] = 1e-4; // Background salt trace
    
    std::cout << "-> Running Linear Solver (Debye-Huckel)..." << std::endl;
    minimize_cg(cell, rho_basis, mock_eps, small_kappa, mock_B, phi_linear, ncg);

    // 4. 运行非线性求解器 (Nonlinear VASPsol++)
    // 设置高浓度盐以观察非线性效应
    double old_c = PARAM.inp.c_molar;
    PARAM.inp.c_molar = 1.0; // 强制 1.0 M
    
    std::cout << "-> Running Nonlinear Solver (Newton-Raphson, 1.0 M)..." << std::endl;
    
    // 复刻 cal_vel 中的非线性循环逻辑 (简化版)
    std::complex<double>* phi_nonlinear = new std::complex<double>[npw];
    ModuleBase::GlobalFunc::ZEROS(phi_nonlinear, npw);
    double* rho_ion_R = new double[nrxx];
    double* kappa2_R = new double[nrxx];
    double* phi_R_tmp = new double[nrxx];
    std::complex<double>* B_total = new std::complex<double>[npw];

    for(int iter=0; iter<10; ++iter) {
        rho_basis->recip2real(phi_nonlinear, phi_R_tmp);
        
        // 更新物理量
        cal_smpbe_physics(nrxx, phi_R_tmp, mock_eps, rho_ion_R, kappa2_R);
        
        // 构建 RHS
        for(int i=0; i<nrxx; ++i) {
            phi_R_tmp[i] = -4.0 * ModuleBase::PI * rho_ion_R[i] 
                           - mock_eps[i] * kappa2_R[i] * phi_R_tmp[i];
        }
        rho_basis->real2recip(phi_R_tmp, B_total);
        for(int ig=0; ig<npw; ++ig) B_total[ig] += mock_B[ig];
        
        // 求解
        std::complex<double>* phi_new = new std::complex<double>[npw];
        minimize_cg(cell, rho_basis, mock_eps, kappa2_R, B_total, phi_new, ncg);
        
        // 混合
        double change = 0.0;
        for(int ig=0; ig<npw; ++ig) {
            double diff = std::abs(phi_new[ig] - phi_nonlinear[ig]);
            change += diff;
            phi_nonlinear[ig] = 0.6 * phi_new[ig] + 0.4 * phi_nonlinear[ig];
        }
        delete[] phi_new;
        
        std::cout << "   Iter " << iter << ": Change = " << change << std::endl;
        if(change < 1e-4) break;
    }

    // 5. 结果对比与诊断
    double* phi_lin_R = new double[nrxx];
    double* phi_non_R = new double[nrxx];
    rho_basis->recip2real(phi_linear, phi_lin_R);
    rho_basis->recip2real(phi_nonlinear, phi_non_R);
    
    // 计算中心处的最大电势和离子浓度
    double max_phi_lin = 0.0;
    double max_phi_non = 0.0;
    double max_rho_ion = 0.0;
    
    // 计算理论最大离子浓度 (Packing limit)
    // c_max = 1 / a^3
    double a_ion_bohr = (PARAM.inp.ion_size < 0.1 ? 3.0 : PARAM.inp.ion_size) * 1.8897; 
    double c_max_au = 1.0 / pow(a_ion_bohr, 3);
    
    // 重新计算 rho_ion for nonlinear
    double* k_dummy = new double[nrxx];
    cal_smpbe_physics(nrxx, phi_non_R, mock_eps, rho_ion_R, k_dummy);
    
    for(int i=0; i<nrxx; ++i) {
        if(std::abs(phi_lin_R[i]) > max_phi_lin) max_phi_lin = std::abs(phi_lin_R[i]);
        if(std::abs(phi_non_R[i]) > max_phi_non) max_phi_non = std::abs(phi_non_R[i]);
        if(std::abs(rho_ion_R[i]) > max_rho_ion) max_rho_ion = std::abs(rho_ion_R[i]);
    }
    
    std::cout << "\n  --- RESULTS ---" << std::endl;
    std::cout << "  Salt Concentration: " << PARAM.inp.c_molar << " M" << std::endl;
    std::cout << "  Linear PB Max Potential:    " << max_phi_lin << " Ha" << std::endl;
    std::cout << "  Nonlinear PB Max Potential: " << max_phi_non << " Ha" << std::endl;
    std::cout << "  (Nonlinear should be larger due to saturation reducing screening near ion)" << std::endl;
    std::cout << "  ---------------------------" << std::endl;
    std::cout << "  Max Ion Density (Calc):     " << max_rho_ion << " e/Bohr^3" << std::endl;
    std::cout << "  Packing Limit (Theoretical):" << c_max_au  << " e/Bohr^3" << std::endl;
    
    if (max_rho_ion < c_max_au * 1.01) {
        std::cout << "  [SUCCESS] Ion density is within steric limits!" << std::endl;
    } else {
        std::cout << "  [WARNING] Ion density exceeds limit! Check SMPBE logic." << std::endl;
    }
    std::cout << "========================================================\n" << std::endl;

    // 恢复参数与清理
    PARAM.inp.c_molar = old_c;
    
    delete[] mock_TOTN; delete[] mock_TOTN_R; delete[] mock_eps;
    delete[] phi_linear; delete[] phi_nonlinear; delete[] mock_B;
    delete[] rho_ion_R; delete[] kappa2_R; delete[] phi_R_tmp;
    delete[] B_total; delete[] phi_lin_R; delete[] phi_non_R;
    delete[] k_dummy; delete[] small_kappa;
}

void surchem::clear()
{
    delete[] TOTN_real;
    delete[] delta_phi;
    delete[] epspot;
    this->TOTN_real = nullptr;
    this->delta_phi = nullptr;
    this->epspot = nullptr;
}

surchem::~surchem()
{
    this->clear();
}
