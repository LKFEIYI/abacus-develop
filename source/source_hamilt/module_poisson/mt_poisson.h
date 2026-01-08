#ifndef MT_POISSON_H
#define MT_POISSON_H

#include "module_base/vector3.h"
#include "module_base/constants.h"
#include <string>
#include <cmath>

// 假设 PARAM 或者 INPUT 结构体已经包含了 mt_slab_size 等参数
// 这里我们尽量让接口保持纯净，只传递数值，不直接依赖全局变量
// 这样方便单元测试

class MTPoisson {
public:
    /**
     * @brief 计算 G=0 处的修正值 (Gamma点)
     * * @param L 真空层厚度 (slab size), 单位: Bohr
     * @param type MT类型 ("2D", "0D" 等)
     * @return double 修正势的值 (Hartree)
     */
    static double get_screen_val_g0(double L, std::string type);

    /**
     * @brief 计算 G != 0 处的屏蔽势 V_screen(G)
     * * @param g2 G向量模的平方 (单位: Bohr^-2)
     * @param g_vec G向量 (单位: Bohr^-1)
     * @param L 真空层厚度 (slab size), 单位: Bohr
     * @param alpha 平滑参数 (用于0D/1D), 单位: Bohr^-1
     * @param type MT类型 ("2D", "0D" 等)
     * @param direction 非周期方向 (0=X, 1=Y, 2=Z), 默认为 1 (Y)
     * @return double 修正值，需与 4pi/G^2 叠加
     */
    static double get_screen_val(double g2, 
                                 const ModuleBase::Vector3<double>& g_vec, 
                                 double L, 
                                 double alpha, 
                                 std::string type,
                                 int direction = 1);

private:
    // 具体的 MT2D 实现逻辑
    static double calculate_mt2d(double g2, double g_special, double L);
};

#endif // MT_POISSON_H