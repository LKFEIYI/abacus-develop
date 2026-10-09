#include "source_estate/module_pot/sccs_output.h"
#include "source_estate/module_pot/pot_sccs.h"
#include "source_estate/module_pot/pot_pcc.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include "source_io/module_parameter/input_parameter.h"
#include <sstream>

namespace
{
const elecstate::SccsResume no_resume = elecstate::SccsResume();
}

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
    const double electron_count = 1e-6 * basis.omega;
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
        elecstate::make_sccs_config_from_input(input, config, solver);
        elecstate::PotSccs plain(&basis, config, solver, electron_count, no_resume);
        solver.check_fixed_point = true;
        elecstate::PotSccs checked(&basis, config, solver, electron_count, no_resume);
        elecstate::PotPcc::Dimension dimension = elecstate::PotPcc::Dimension::molecule;
        if (config.boundary == ModuleSccs::Boundary::Pcc2d) { dimension = elecstate::PotPcc::Dimension::slab; }
        elecstate::PotPcc pcc(&basis, dimension, 2, electron_count);
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
        checked.write_iteration_output(summary, 2, 1e-6, pcc_energy);
        EXPECT_NE(summary.str().find("SCCS_GAUSS"), std::string::npos);
        EXPECT_NE(summary.str().find("TOLERANCE/e"), std::string::npos);
        EXPECT_NE(summary.str().find("SCCS_CG_FIXED_POINT_DEFECT"), std::string::npos);
        EXPECT_TRUE(checked.has_output());
        EXPECT_TRUE(pcc.has_result());
        config.start_drho = 1e-4;
        elecstate::PotSccs deferred(&basis, config, solver, electron_count, no_resume);
        EXPECT_FALSE(deferred.is_active());
        EXPECT_FALSE(deferred.has_output());
        std::ostringstream waiting;
        deferred.write_iteration_output(waiting, 1, 1e-3, pcc_energy);
        EXPECT_NE(waiting.str().find("SCCS_DEFERRED"), std::string::npos);
    }
}

TEST_F(SccsOutputTest, ScreeningToleranceKeepsOriginalScales)
{
    // Relative scale 1e-4 per electron dominates small cells.
    EXPECT_DOUBLE_EQ(elecstate::sccs_screening_tolerance(0.6, 1.0, 1e-8, 1000.0), 1e-4);
    EXPECT_DOUBLE_EQ(elecstate::sccs_screening_tolerance(50.0, 49.0, 1e-8, 1000.0), 5e-3);
    // A loose solver tolerance over a large cell dominates.
    EXPECT_DOUBLE_EQ(elecstate::sccs_screening_tolerance(1.0, 1.0, 1e-6, 1000.0), 1e-3);
}
