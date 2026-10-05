#include "source_io/module_parameter/read_inp_sccs.h"
#include "source_io/module_parameter/input_parameter.h"
#include <gtest/gtest.h>
#include <limits>

TEST(ReadInpSccs, LegacyBooleansAndModelSelection)
{
    std::string error;
    const std::string yes[] = {"true", "TRUE", "yes", "On", ".true.", "1", "t", "y"};
    const std::string no[] = {"false", "FALSE", "no", "Off", ".false.", "0", "f", "n"};
    int model = 0;
    for (const std::string& value : yes)
    {
        ASSERT_TRUE(ModuleIO::parse_solvation_model(value, model, error));
        EXPECT_EQ(model, 1);
    }
    for (const std::string& value : no)
    {
        ASSERT_TRUE(ModuleIO::parse_solvation_model(value, model, error));
        EXPECT_EQ(model, 0);
    }
    ASSERT_TRUE(ModuleIO::parse_solvation_model("2", model, error));
    EXPECT_EQ(model, 2);
    EXPECT_FALSE(ModuleIO::parse_solvation_model("3", model, error));
    EXPECT_EQ(model, 2);
    EXPECT_FALSE(ModuleIO::parse_solvation_model("2.0", model, error));
}

TEST(ReadInpSccs, SupportedScopeAndNumericalValidation)
{
    Input_para input;
    input.imp_sol = 2;
    input.device = "cpu";
    input.basis_type = "pw";
    std::string error;
    ASSERT_TRUE(ModuleIO::validate_sccs_input(input, error)) << error;
    input.cal_force = true;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.cal_stress = true;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.cal_stress = false;
    input.cal_force = false;
    input.assume_isolated = "pcc_0d";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.cal_force = true;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.cal_force = false;
    input.assume_isolated = "pcc_2d";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.cal_force = true;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.cal_force = false;
    input.calculation = "relax";
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.calculation = "scf";
    input.assume_isolated = "none";
    input.sccs_rho_min = input.sccs_rho_max;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_rho_min = 1e-4;
    input.sccs_preset = "unknown";
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_preset = "vacuum";
    input.basis_type = "lcao";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.imp_sol = 1;
    input.sccs_preset = "unused";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
}

TEST(ReadInpSccs, LowpassRequiresPairedFiniteValuesAndPcc)
{
    Input_para input;
    input.imp_sol = 2;
    std::string error;
    EXPECT_DOUBLE_EQ(input.sccs_lowpass_p1, -1.0);
    EXPECT_DOUBLE_EQ(input.sccs_lowpass_p2, -1.0);
    input.sccs_lowpass_p1 = 10.0;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_lowpass_p2 = 5.0;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.assume_isolated = "pcc_0d";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.assume_isolated = "pcc_2d";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_lowpass_p1 = 0.0;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_lowpass_p2 = 0.0;
    input.assume_isolated = "none";
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_lowpass_p1 = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_lowpass_p1 = -1.0;
    input.sccs_lowpass_p2 = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
}

TEST(ReadInpSccs, FullCavityWidthsAndModeValidation)
{
    Input_para input;
    input.imp_sol = 2;
    input.sccs_solvent_mode = "full";
    std::string error;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_corespread = {0.7, 0.0, -1.0};
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_corespread = {0.0};
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_corespread = {std::numeric_limits<double>::infinity()};
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_corespread.clear();
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_corespread = {0.5};
    input.sccs_solvent_mode = "invalid";
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
}

TEST(ReadInpSccs, DelayedStartRequiresRoomBeforeScfConvergenceAndIterationLimit)
{
    Input_para input;
    input.imp_sol = 2;
    input.scf_thr = 1e-9;
    input.scf_nmax = 10;
    std::string error;
    ASSERT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = 1e-3;
    input.sccs_start_nmax = 3;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = input.scf_thr;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = -1.0;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = 1e-3;
    input.sccs_start_nmax = 10;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_drho = 0.0;
    EXPECT_TRUE(ModuleIO::validate_sccs_input(input, error));
    input.sccs_start_nmax = 0;
    EXPECT_FALSE(ModuleIO::validate_sccs_input(input, error));
}
