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
    ModuleSccs::SccsConfig config;
    ASSERT_TRUE(ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral, config, error));
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
    ASSERT_TRUE(ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                basis, tpiba, plain, first, error));
    solver.check_fixed_point = true;
    ASSERT_TRUE(ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                basis, tpiba, checked, second, error));
    EXPECT_FALSE(first.polarization.fixed_point_checked);
    EXPECT_TRUE(second.polarization.fixed_point_checked);
    EXPECT_LT(second.polarization.fixed_point_defect_max, 1e-9);
    EXPECT_EQ(checked.transform_counts().forward_calls, plain.transform_counts().forward_calls + 1);
    EXPECT_EQ(checked.transform_counts().inverse_calls, plain.transform_counts().inverse_calls + 1);
    EXPECT_EQ(first.polarization.potential, second.polarization.potential);
    std::vector<double> polarization;
    ASSERT_TRUE(ModuleSccs::continuum_polarization_charge(charge, second, basis, tpiba, polarization, error));
    for (int i = 0; i < basis.nrxx; ++i) { EXPECT_NEAR(polarization[i], -0.8 * charge[i], 1e-12); }
}

TEST_F(SccsDiagnosticsTest, InvalidLocalShapeFailsCollectivelyWithoutOverwritingResult)
{
    const std::vector<double> charge = cosine_mode(0);
    const std::vector<double> potential(basis.nrxx, 0.0);
    const std::vector<double> invsqrt(basis.nrxx, 1.0);
    std::vector<double> coefficient(basis.nrxx, 0.0);
    if (SccsTest::pool_rank == 0) { coefficient.push_back(0.0); }
    ModuleSccs::PeriodicCoulombOperator coulomb(basis, tpiba);
    ModuleSccs::PolarizationResult result;
    result.fixed_point_defect_max = 123.0;
    EXPECT_FALSE(ModuleSccs::check_sccs_fixed_point(charge, coefficient, potential, invsqrt,
                 basis, coulomb, result, error));
    EXPECT_FALSE(error.empty());
    EXPECT_FALSE(result.fixed_point_checked);
    EXPECT_EQ(result.fixed_point_defect_max, 123.0);
    EXPECT_EQ(coulomb.transform_counts().forward_calls, 0);
    ModuleSccs::SccsResponse response;
    response.epsilon.assign(basis.nrxx, 5.0);
    response.polarization.potential = potential;
    response.grad_log_epsilon.resize(basis.nrxx);
    if (SccsTest::pool_rank == 0) { response.grad_log_epsilon.push_back({}); }
    std::vector<double> density(1, 123.0);
    EXPECT_FALSE(ModuleSccs::continuum_polarization_charge(charge, response, basis, tpiba, density, error));
    ASSERT_EQ(density.size(), 1u);
    EXPECT_EQ(density[0], 123.0);
}

TEST_F(SccsDiagnosticsTest, NonuniformFilteredCavityPreservesRestartAndDerivatives)
{
    ModuleSccs::SccsConfig config;
    ASSERT_TRUE(ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral, config, error));
    config.cavity.epsilon_bulk = 5.0;
    config.cavity.lowpass_p1 = basis.ggecut;
    config.cavity.lowpass_p2 = 0.5;
    const std::vector<double> charge = cosine_mode(0);
    std::vector<double> density(basis.nrxx);
    for (int i = 0; i < basis.nrxx; ++i) { density[i] = 1e-3 + 1e-4 * charge[i]; }
    const std::vector<double> cold;
    const ModuleBase::Matrix3 lattice;
    std::vector<ModuleBase::Vector3<double>> positions;
    ASSERT_TRUE(ModulePW::grid_positions(basis, lattice, length, positions, error));
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
    ASSERT_TRUE(ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                basis, tpiba, plain, first, error)) << error;
    solver.check_fixed_point = true;
    ASSERT_TRUE(ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold,
                basis, tpiba, checked, second, error)) << error;
    EXPECT_TRUE(second.polarization.fixed_point_checked);
    EXPECT_LT(second.polarization.fixed_point_defect_max, 1e-8);
    EXPECT_EQ(first.polarization.potential, second.polarization.potential);
    EXPECT_EQ(first.restart_potential, second.restart_potential);
    EXPECT_EQ(first.cavity_potential, second.cavity_potential);
    EXPECT_EQ(first.polarization.iterations, second.polarization.iterations);
    std::vector<double> polarization;
    ASSERT_TRUE(ModuleSccs::continuum_polarization_charge(charge, second, basis, tpiba, polarization, error));
    EXPECT_EQ(polarization.size(), charge.size());
}
