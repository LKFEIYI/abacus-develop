#include "sccs_test.h"
#include "../sccs_diagnostics.h"
#include "../sccs_parameters.h"
#include "../sccs_response.h"
#include "../sccs_pw_coulomb.h"
#include "../sccs_pcc_0d_coulomb.h"
#include "source_basis/module_pw/pw_grid_geometry.h"

using SccsDiagnosticsTest = SccsTest::PwTest;

TEST_F(SccsDiagnosticsTest, FixedPointCheckAddsOneSolveWithoutChangingResponse)
{
    ModuleSccs::SccsConfig config = ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral);
    config.cavity.epsilon_bulk = 5.0;
    const double density_value = 1e-6;
    const std::vector<double> density(basis.nrxx, density_value);
    const std::vector<double> charge = cosine_mode(0);
    const std::vector<double> cold;
    ModuleSccs::PeriodicCoulombOperator plain(basis, tpiba);
    ModuleSccs::PeriodicCoulombOperator checked(basis, tpiba);
    ModuleSccs::PolarizationSolverParameters solver;
    ModuleSccs::SccsResponse first;
    ModuleSccs::SccsResponse second;
    ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                                    basis, tpiba, plain, first);
    solver.check_fixed_point = true;
    ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                                    basis, tpiba, checked, second);
    EXPECT_FALSE(first.polarization.fixed_point_checked);
    EXPECT_TRUE(second.polarization.fixed_point_checked);
    EXPECT_LT(second.polarization.fixed_point_defect_max, 1e-9);
    const int expected_forward = plain.transform_counts().forward_calls + 1;
    EXPECT_EQ(checked.transform_counts().forward_calls, expected_forward);
    const int expected_inverse = plain.transform_counts().inverse_calls + 1;
    EXPECT_EQ(checked.transform_counts().inverse_calls, expected_inverse);
    EXPECT_EQ(first.polarization.potential, second.polarization.potential);
    std::vector<double> polarization;
    ModuleSccs::continuum_polarization_charge(charge, second, basis, tpiba, polarization);
    for (int i = 0; i < basis.nrxx; ++i)
    {
        const double expected_charge = -0.8 * charge[i];
        EXPECT_NEAR(polarization[i], expected_charge, 1e-12);
    }
}

TEST_F(SccsDiagnosticsTest, NonuniformFilteredCavityPreservesRestartAndDerivatives)
{
    ModuleSccs::SccsConfig config = ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral);
    config.cavity.epsilon_bulk = 5.0;
    config.cavity.lowpass_p1 = basis.ggecut;
    config.cavity.lowpass_p2 = 0.5;
    const std::vector<double> charge = cosine_mode(0);
    std::vector<double> density(basis.nrxx);
    for (int i = 0; i < basis.nrxx; ++i) { density[i] = 1e-3 + 1e-4 * charge[i]; }
    const std::vector<double> cold;
    const ModuleBase::Matrix3 lattice;
    std::vector<ModuleBase::Vector3<double>> positions;
    ModulePW::grid_positions(basis, lattice, length, positions);
    const double half_length = 0.5 * length;
    const ModuleBase::Vector3<double> center(half_length, half_length, half_length);
    for (auto& position : positions) { position -= center; }
    elecstate::Pcc0dParameters parameters;
    parameters.length = length;
    ModuleSccs::Pcc0dCoulombOperator plain(basis, tpiba, positions, parameters);
    ModuleSccs::Pcc0dCoulombOperator checked(basis, tpiba, positions, parameters);
    ModuleSccs::PolarizationSolverParameters solver;
    ModuleSccs::SccsResponse first;
    ModuleSccs::SccsResponse second;
    ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                                    basis, tpiba, plain, first);
    solver.check_fixed_point = true;
    ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                                    basis, tpiba, checked, second);
    EXPECT_TRUE(second.polarization.fixed_point_checked);
    EXPECT_LT(second.polarization.fixed_point_defect_max, 1e-8);
    EXPECT_EQ(first.polarization.potential, second.polarization.potential);
    EXPECT_EQ(first.restart_potential, second.restart_potential);
    EXPECT_EQ(first.cavity_potential, second.cavity_potential);
    EXPECT_EQ(first.polarization.iterations, second.polarization.iterations);
    std::vector<double> polarization;
    ModuleSccs::continuum_polarization_charge(charge, second, basis, tpiba, polarization);
    EXPECT_EQ(polarization.size(), charge.size());
}
