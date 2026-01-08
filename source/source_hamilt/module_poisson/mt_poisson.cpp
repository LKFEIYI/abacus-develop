#include "mt_poisson.h"
#include "module_base/constants.h" // 获取 PI, FOUR_PI
#include <iostream>

// 如果需要报错功能
#include "module_base/global_function.h" 

double MTPoisson::get_screen_val_g0(double L, std::string type) {
    // 对应 CP2K 逻辑: IF (grid%have_g0) screen_function%array(1) = pi*zlength*zlength/2.0_dp
    if (type == "2D") {
        return ModuleBase::PI * L * L / 2.0;
    }
    // 0D 暂时略过，因为需要实空间积分或者 alpha 处理
    return 0.0;
}

double MTPoisson::get_screen_val(double g2, 
                                 const ModuleBase::Vector3<double>& g_vec, 
                                 double L, 
                                 double alpha, 
                                 std::string type,
                                 int direction) 
{
    // 安全检查：G^2 非常小视为 G=0，返回0 (外部应该处理了 G=0 的情况)
    if (g2 < 1e-10) return 0.0;

    if (type == "2D") {
        // 1. 确定非周期方向的分量 (G_special)
        double g_special = 0.0;
        switch (direction) {
            case 0: g_special = g_vec.x; break; // X 方向真空
            case 1: g_special = g_vec.y; break; // Y 方向真空 (默认)
            case 2: g_special = g_vec.z; break; // Z 方向真空
            default: 
                // 默认回落到 Y
                g_special = g_vec.y; 
                break;
        }

        // 2. 调用核心公式
        return calculate_mt2d(g2, g_special, L);
    }
    else if (type == "0D") {
        // 0D 逻辑 (尚未完全移植，需要 erfc 等特殊处理)
        return 0.0;
    }
    
    return 0.0;
}

double MTPoisson::calculate_mt2d(double g2, double g_special, double L) {
    // 公式来源: CP2K mt_util.F (Martyna-Tuckerman, 1999)
    // V_screen(G) = - (4pi/G^2) * cos(G_z * L / 2) * exp(-G_xy * L / 2)
    // 这里 G_z 泛指非周期方向分量(g_special)，G_xy 泛指周期平面分量
    
    // 1. 计算标准库仑项 (注意符号，这里只计算系数)
    // 最终公式是: fac = (4pi/G^2) + V_screen
    // 所以 V_screen = (4pi/G^2) * [ -cos(...) * exp(...) ]
    double coulomb_term = ModuleBase::FOUR_PI / g2;

    // 2. 计算周期性平面内的 G 分量模长
    // G^2 = G_special^2 + G_planar^2  => G_planar = sqrt(G^2 - G_special^2)
    double g_planar = std::sqrt(std::abs(g2 - g_special * g_special));

    // 3. 计算指数项和余弦项
    double half_L = L / 2.0;
    double cos_term = std::cos(g_special * half_L);
    double exp_term = std::exp(-g_planar * half_L);

    // 4. 组合
    // 注意负号
    return -coulomb_term * cos_term * exp_term;
}