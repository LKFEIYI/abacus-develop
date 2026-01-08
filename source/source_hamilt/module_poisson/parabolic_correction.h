// 修改前：const double* rho_elec
    // 修改后：const double* const* rho_elec, int nspin
    void apply_correction(const UnitCell& cell, 
                          const ModulePW::PW_Basis* rho_basis, 
                          double* v_hartree, 
                          const double* const* rho_elec, // 传入指针的指针
                          int nspin,                     // 传入自旋数
                          double nelec,
                          int correction_dir);

private:
    // ... 其他函数 ...
    
    // 修改后
    double calc_elec_dipole(const ModulePW::PW_Basis* rho_basis, 
                            const double* const* rho_elec, // 指针的指针
                            int nspin,                     // 自旋
                            int dir, 
                            double center_coord,
                            double omega);