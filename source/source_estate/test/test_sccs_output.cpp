#include "source_estate/module_pot/sccs_output.h"
#include "source_estate/module_pot/pot_sccs.h"
#include "source_estate/module_pot/pot_pcc.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include "source_io/module_parameter/input_parameter.h"
#include <sstream>

using SccsOutputTest = SccsTest::PwTest;

TEST_F(SccsOutputTest, OutputLevelsPreserveStreamStateAndUseOriginalLabels)
{
    elecstate::SccsOutput result;
    result.valid = true;
    result.fixed_point_checked = true;
    result.iterations = 4;
    result.reaction_energy = -0.1;
    std::ostringstream output;
    output.setf(std::ios::scientific, std::ios::floatfield);
    output.precision(3);
    const auto flags = output.flags();
    elecstate::write_sccs_output(output, result, 0, -0.2, true, 0.05);
    EXPECT_TRUE(output.str().empty());
    elecstate::write_sccs_output(output, result, 1, -0.2, true, 0.05);
    EXPECT_NE(output.str().find("SCCS_ITER 4"), std::string::npos);
    EXPECT_EQ(output.str().find("SCCS_RESIDUAL"), std::string::npos);
    output.str("");
    elecstate::write_sccs_output(output, result, 2, -0.2, true, 0.05);
    EXPECT_NE(output.str().find("SCCS_CG_FIXED_POINT_DEFECT"), std::string::npos);
    EXPECT_NE(output.str().find("PCC0D_MOMENTS SCREENED"), std::string::npos);
    EXPECT_NE(output.str().find("CACHED_SOURCES 0"), std::string::npos);
    EXPECT_EQ(output.str().find("FILLED_VOLUME"), std::string::npos);
    EXPECT_EQ(output.precision(), 3);
    EXPECT_EQ(output.flags(), flags);
    output.str("");
    result.pcc.slab = true;
    result.pcc.normal.z = 1.0;
    elecstate::write_sccs_final_output(output, result, true, 0.05);
    EXPECT_NE(output.str().find("SCCS_DIAGNOSTIC polarization_quadrupole_normal"), std::string::npos);
    EXPECT_EQ(output.precision(), 3);
    EXPECT_EQ(output.flags(), flags);
}

TEST_F(SccsOutputTest, AdvancedDiagnosticsPreservePccPotentialEnergyAndForce)
{
    UnitCell cell;
    Atom atom;
    cell.lat0 = length;
    cell.tpiba = tpiba;
    cell.omega = basis.omega;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = &atom;
    atom.na = 1;
    atom.mass = 1.0;
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.25, 0.25, 0.25)};
    std::vector<double> density(basis.nrxx, 1e-6);
    double* channel = density.data();
    Charge charge;
    charge.nspin = 1;
    charge.rho = &channel;
    const char* boundaries[] = {"pcc_0d", "pcc_2d"};
    for (const char* boundary : boundaries)
    {
        Input_para input;
        input.assume_isolated = boundary;
        input.sccs_epsilon = 5.0;
        ModuleSccs::SccsConfig config;
        ModuleSccs::PolarizationSolverParameters solver;
        ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
        elecstate::PotSccs plain(&basis, config, solver);
        solver.check_fixed_point = true;
        elecstate::PotSccs checked(&basis, config, solver);
        elecstate::PotPcc::Dimension dimension = elecstate::PotPcc::Dimension::molecule;
        if (config.boundary == ModuleSccs::Boundary::Pcc2d) { dimension = elecstate::PotPcc::Dimension::slab; }
        elecstate::PotPcc pcc(&basis, dimension, 2);
        ModuleBase::matrix vacuum(1, basis.nrxx);
        pcc.cal_v_eff(&charge, &cell, vacuum);
        ModuleBase::matrix first(1, basis.nrxx);
        ModuleBase::matrix second(1, basis.nrxx);
        plain.cal_v_eff(&charge, &cell, first);
        checked.cal_v_eff(&charge, &cell, second);
        EXPECT_DOUBLE_EQ(plain.get_energy(), checked.get_energy());
        for (int i = 0; i < basis.nrxx; ++i) { EXPECT_DOUBLE_EQ(first(0, i), second(0, i)); }
        ModuleBase::matrix force1(1, 3);
        ModuleBase::matrix force2(1, 3);
        plain.add_solvation_force(cell, force1);
        checked.add_solvation_force(cell, force2);
        for (int i = 0; i < 3; ++i) { EXPECT_DOUBLE_EQ(force1(0, i), force2(0, i)); }
        const double pcc_energy = pcc.get_energy();
        std::ostringstream summary;
        checked.write_correction_iteration(summary, 2, 1e-6, pcc_energy);
        EXPECT_NE(summary.str().find("SCCS_GAUSS"), std::string::npos);
        EXPECT_NE(summary.str().find("SCCS_CG_FIXED_POINT_DEFECT"), std::string::npos);
        EXPECT_EQ(checked.correction_output_priority(), 3);
        EXPECT_EQ(pcc.correction_output_priority(), 2);
        config.start_drho = 1e-4;
        elecstate::PotSccs deferred(&basis, config, solver);
        EXPECT_EQ(deferred.correction_output_priority(), 1);
        std::ostringstream waiting;
        deferred.write_correction_iteration(waiting, 1, 1e-3, pcc_energy);
        EXPECT_NE(waiting.str().find("SCCS_DEFERRED"), std::string::npos);
    }
}

TEST_F(SccsOutputTest, InputDebugLevelIsExplicitAndInvalidValuesPreserveConfiguration)
{
    Input_para input;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    for (int level = 0; level <= 2; ++level)
    {
        input.sccs_debug = level;
        ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
        EXPECT_EQ(solver.check_fixed_point, level == 2);
    }
    const double epsilon = config.cavity.epsilon_bulk;
    const int invalid_levels[] = {-1, 3};
    for (int level : invalid_levels)
    {
        input.sccs_debug = level;
        EXPECT_FALSE(elecstate::make_sccs_config_from_input(input, config, solver, error));
        EXPECT_DOUBLE_EQ(config.cavity.epsilon_bulk, epsilon);
        EXPECT_TRUE(solver.check_fixed_point);
    }
}
