#include "source_estate/module_pot/pot_sccs.h"
#include "source_io/module_parameter/input_parameter.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"

// Public storage fixtures: no INPUT initialization or privately owned atom maps.
UnitCell::UnitCell() {}
UnitCell::~UnitCell() {}
Magnetism::Magnetism() {}
Magnetism::~Magnetism() {}
SepPot::SepPot() {}
SepPot::~SepPot() {}
Sep_Cell::Sep_Cell() noexcept {}
Sep_Cell::~Sep_Cell() noexcept {}
Charge::Charge() {}
Charge::~Charge() {}

using PotSccsTest = SccsTest::PwTest;

TEST_F(PotSccsTest, IndependentInstancesAndSpinChannels)
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
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.25, 0.25, 0.25)};
    const double density_value = 1.0 / basis.omega;
    std::vector<double> density(basis.nrxx, density_value);
    double* channels[] = {density.data(), density.data()};
    Charge charge;
    charge.nspin = 1;
    charge.rho = channels;
    Input_para input;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    config.cavity.epsilon_bulk = 5.0;
    elecstate::PotSccs first(&basis, config, solver);
    config.cavity.epsilon_bulk = 1.0;
    elecstate::PotSccs vacuum(&basis, config, solver);
    EXPECT_DOUBLE_EQ(first.get_energy(), 0.0);
    ModuleBase::matrix potential(1, basis.nrxx);
    first.cal_v_eff(&charge, &cell, potential);
    const double first_energy = first.get_energy();
    EXPECT_LT(first_energy, 0.0);
    double el = 0.0;
    double cav = 0.0;
    first.get_solvation_energy(el, cav);
    EXPECT_DOUBLE_EQ(el, first_energy);
    EXPECT_DOUBLE_EQ(cav, 0.0);
    ModuleBase::matrix zero(1, basis.nrxx);
    vacuum.cal_v_eff(&charge, &cell, zero);
    EXPECT_NEAR(vacuum.get_energy(), 0.0, 1e-12);
    EXPECT_DOUBLE_EQ(first.get_energy(), first_energy);
    for (int ir = 0; ir < basis.nrxx; ++ir) { EXPECT_NEAR(zero(0, ir), 0.0, 1e-11); }
    for (double& value : density) { value *= 0.5; }
    charge.nspin = 2;
    ModuleBase::matrix spin_potential(2, basis.nrxx);
    first.cal_v_eff(&charge, &cell, spin_potential);
    EXPECT_NEAR(first.get_energy(), first_energy, 1e-12);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_NEAR(spin_potential(0, ir), potential(0, ir), 1e-11);
        EXPECT_DOUBLE_EQ(spin_potential(0, ir), spin_potential(1, ir));
    }
}

TEST_F(PotSccsTest, PresetConfigurationIsExplicit)
{
    Input_para input;
    input.sccs_preset = "water-neutral";
    input.sccs_epsilon = 2.0;
    input.sccs_maxiter = 42;
    input.sccs_surface_eta = 1e-6;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_DOUBLE_EQ(config.cavity.epsilon_bulk, 78.3);
    EXPECT_DOUBLE_EQ(config.cavity.density_min, 1e-4);
    EXPECT_LT(config.pressure, 0.0);
    EXPECT_DOUBLE_EQ(config.surface_regularization, 1e-6);
    EXPECT_EQ(solver.max_iterations, 42);
    input.sccs_maxiter = 0;
    EXPECT_FALSE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_EQ(solver.max_iterations, 42);
}

TEST_F(PotSccsTest, IonicForceAddsRydbergDerivativeOnce)
{
    UnitCell cell;
    Atom atom;
    cell.lat0 = length;
    cell.tpiba = tpiba;
    cell.omega = basis.omega;
    cell.ntype = 1;
    cell.nat = 2;
    cell.atoms = &atom;
    atom.na = 2;
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.21, 0.32, 0.43),
                ModuleBase::Vector3<double>(0.64, 0.51, 0.27)};
    const double density_value = 2.0 / basis.omega;
    std::vector<double> density(basis.nrxx, density_value);
    double* channels[] = {density.data()};
    Charge charge;
    charge.nspin = 1;
    charge.rho = channels;
    Input_para input;
    input.sccs_epsilon = 5.0;
    input.sccs_tol_rms = 1e-13;
    input.sccs_tol_max = 1e-12;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    elecstate::PotSccs component(&basis, config, solver);
    ModuleBase::matrix potential(1, basis.nrxx);
    component.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix force(2, 3);
    component.add_solvation_force(cell, force);
    const double step = 1e-4;
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const double original = atom.tau[ia][axis];
            atom.tau[ia][axis] = original + step / length;
            potential.zero_out();
            component.cal_v_eff(&charge, &cell, potential);
            const double positive = component.get_energy();
            atom.tau[ia][axis] = original - step / length;
            potential.zero_out();
            component.cal_v_eff(&charge, &cell, potential);
            const double negative = component.get_energy();
            atom.tau[ia][axis] = original;
            const double finite_difference = -(positive - negative) / (2.0 * step);
            EXPECT_NEAR(force(ia, axis), finite_difference, 1e-8);
        }
    }
    potential.zero_out();
    component.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix accumulated(2, 3);
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis) { accumulated(ia, axis) = 7.0; }
    }
    component.add_solvation_force(cell, accumulated);
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(accumulated(ia, axis), 7.0 + force(ia, axis), 1e-10);
        }
    }
}

TEST_F(PotSccsTest, LowpassControlsSurvivePresetAndInvalidInputPreservesConfig)
{
    Input_para input;
    input.assume_isolated = "pcc_0d";
    input.sccs_preset = "water-cation";
    input.sccs_lowpass_p1 = 10.0;
    input.sccs_lowpass_p2 = 5.0;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_DOUBLE_EQ(config.cavity.lowpass_p1, 10.0);
    EXPECT_DOUBLE_EQ(config.cavity.lowpass_p2, 5.0);
    EXPECT_DOUBLE_EQ(config.cavity.density_min, 2e-4);
    input.assume_isolated = "pcc_2d";
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_EQ(config.boundary, ModuleSccs::Boundary::Pcc2d);
    input.assume_isolated = "none";
    EXPECT_FALSE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_EQ(config.boundary, ModuleSccs::Boundary::Pcc2d);
    EXPECT_DOUBLE_EQ(config.cavity.lowpass_p1, 10.0);
}

TEST_F(PotSccsTest, FullCavityForceIncludesNonElectrostaticDerivative)
{
    UnitCell cell;
    Atom atom;
    cell.lat0 = length;
    cell.tpiba = tpiba;
    cell.omega = basis.omega;
    cell.ntype = 1;
    cell.nat = 2;
    cell.atoms = &atom;
    atom.na = 2;
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.21, 0.32, 0.43),
                ModuleBase::Vector3<double>(0.64, 0.51, 0.27)};
    const double density_value = 2.0 / basis.omega;
    std::vector<double> density(basis.nrxx, density_value);
    double* channels[] = {density.data()};
    Charge charge;
    charge.nspin = 1;
    charge.rho = channels;
    Input_para input;
    input.sccs_epsilon = 1.0;
    input.sccs_tol_rms = 1e-13;
    input.sccs_tol_max = 1e-12;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    config.core_electrons = true;
    config.core_spreads = {2.0};
    config.surface_tension = 1e-5;
    config.pressure = 1e-6;
    elecstate::PotSccs component(&basis, config, solver);
    ModuleBase::matrix potential(1, basis.nrxx);
    component.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix force(2, 3);
    component.add_solvation_force(cell, force);
    const double step = 1e-4;
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const double original = atom.tau[ia][axis];
            atom.tau[ia][axis] = original + step / length;
            potential.zero_out();
            component.cal_v_eff(&charge, &cell, potential);
            const double positive = component.get_energy();
            atom.tau[ia][axis] = original - step / length;
            potential.zero_out();
            component.cal_v_eff(&charge, &cell, potential);
            const double negative = component.get_energy();
            atom.tau[ia][axis] = original;
            const double finite_difference = -(positive - negative) / (2.0 * step);
            EXPECT_NEAR(force(ia, axis), finite_difference, 1e-8);
        }
    }
    potential.zero_out();
    component.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix accumulated(2, 3);
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis) { accumulated(ia, axis) = 7.0; }
    }
    component.add_solvation_force(cell, accumulated);
    for (int ia = 0; ia < 2; ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            EXPECT_NEAR(accumulated(ia, axis), 7.0 + force(ia, axis), 1e-10);
        }
    }
}

TEST_F(PotSccsTest, FullCavityInputUsesOriginalParameterNameAndKeepsPresetWidths)
{
    Input_para input;
    input.sccs_preset = "water-cation";
    input.sccs_solvent_mode = "full";
    input.sccs_corespread = {0.8, 0.0, -1.0};
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_TRUE(config.core_electrons);
    EXPECT_EQ(config.core_spreads, input.sccs_corespread);
    EXPECT_DOUBLE_EQ(config.cavity.epsilon_bulk, 78.3);
}

TEST_F(PotSccsTest, DelayedActivationHasOneIrreversibleTransition)
{
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    config.start_drho = 1e-3;
    config.start_nmax = 3;
    elecstate::PotSccs threshold(&basis, config, solver);
    EXPECT_FALSE(threshold.sccs_is_active());
    EXPECT_FALSE(threshold.update_scf_state(1, 0.1));
    EXPECT_TRUE(threshold.update_scf_state(2, 1e-3));
    EXPECT_TRUE(threshold.sccs_is_active());
    EXPECT_FALSE(threshold.update_scf_state(3, 0.2));
    EXPECT_FALSE(threshold.update_scf_state(1, 0.2));
    elecstate::PotSccs forced(&basis, config, solver);
    EXPECT_FALSE(forced.update_scf_state(2, 0.1));
    EXPECT_TRUE(forced.update_scf_state(3, 0.1));
    config.start_drho = 0.0;
    elecstate::PotSccs immediate(&basis, config, solver);
    EXPECT_TRUE(immediate.sccs_is_active());
    EXPECT_FALSE(immediate.update_scf_state(1, 0.0));
}

TEST_F(PotSccsTest, WaitingSccsContributesNoPotentialEnergyOrForce)
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
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.25, 0.25, 0.25)};
    const double value = 1.0 / basis.omega;
    std::vector<double> density(basis.nrxx, value);
    double* channels[] = {density.data()};
    Charge charge;
    charge.nspin = 1;
    charge.rho = channels;
    ModuleSccs::SccsConfig config;
    ASSERT_TRUE(ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral, config, error));
    config.start_drho = 1e-3;
    ModuleSccs::PolarizationSolverParameters solver;
    elecstate::PotSccs delayed(&basis, config, solver);
    ModuleBase::matrix potential(1, basis.nrxx);
    delayed.cal_v_eff(&charge, &cell, potential);
    EXPECT_DOUBLE_EQ(delayed.get_energy(), 0.0);
    for (int ir = 0; ir < basis.nrxx; ++ir) { EXPECT_DOUBLE_EQ(potential(0, ir), 0.0); }
    ModuleBase::matrix force(1, 3);
    force(0, 0) = 42.0;
    delayed.add_solvation_force(cell, force);
    EXPECT_DOUBLE_EQ(force(0, 0), 42.0);
    EXPECT_TRUE(delayed.update_scf_state(2, 1e-4));
    delayed.cal_v_eff(&charge, &cell, potential);
    EXPECT_LT(delayed.get_energy(), 0.0);
}

TEST_F(PotSccsTest, PresetKeepsDelayedStartControlsAndInvalidInputPreservesConfig)
{
    Input_para input;
    input.sccs_preset = "water-cation";
    input.sccs_start_drho = 1e-3;
    input.sccs_start_nmax = 3;
    input.scf_nmax = 10;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_DOUBLE_EQ(config.start_drho, 1e-3);
    EXPECT_EQ(config.start_nmax, 3);
    input.sccs_start_drho = input.scf_thr;
    EXPECT_FALSE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_DOUBLE_EQ(config.start_drho, 1e-3);
}

TEST_F(PotSccsTest, IonicReregistrationPreservesOnlyCompatibleActivation)
{
    ModuleSccs::SccsConfig config;
    config.start_drho = 1e-3;
    config.start_nmax = 3;
    ModuleSccs::PolarizationSolverParameters solver;
    elecstate::PotSccs previous(&basis, config, solver);
    elecstate::PotSccs waiting(&basis, config, solver);
    waiting.inherit_scf_state(previous);
    EXPECT_FALSE(waiting.sccs_is_active());
    ASSERT_TRUE(previous.update_scf_state(3, 0.1));
    elecstate::PotSccs next_ionic_step(&basis, config, solver);
    next_ionic_step.inherit_scf_state(previous);
    EXPECT_TRUE(next_ionic_step.sccs_is_active());
    EXPECT_FALSE(next_ionic_step.update_scf_state(1, 0.1));
    EXPECT_DOUBLE_EQ(next_ionic_step.get_energy(), 0.0);
    const std::vector<double>* potential = next_ionic_step.solvent_electrostatic_potential();
    EXPECT_TRUE(potential->empty());
    config.start_drho = 1e-2;
    elecstate::PotSccs changed_controls(&basis, config, solver);
    changed_controls.inherit_scf_state(previous);
    EXPECT_FALSE(changed_controls.sccs_is_active());
    elecstate::PotBase other;
    changed_controls.inherit_scf_state(other);
    EXPECT_FALSE(changed_controls.sccs_is_active());
}
