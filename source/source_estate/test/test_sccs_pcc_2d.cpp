#include "source_estate/module_pot/pot_sccs.h"
#include "source_estate/module_pot/pot_pcc.h"
#include "source_estate/module_pot/sccs_pcc_2d.h"
#include "source_cell/cell_tools.h"
#include "source_hamilt/module_sccs/sccs_coulomb.h"
#include "source_hamilt/module_sccs/sccs_functional.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/sccs_ionic_force.h"
#include "source_hamilt/module_sccs/sccs_response.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include "source_io/module_parameter/input_parameter.h"

class SccsPcc2dTest : public SccsTest::PwTest
{
protected:
    void SetUp() override
    {
#ifdef __MPI
        basis.initmpi(SccsTest::pool_size, SccsTest::pool_rank, POOL_WORLD);
#endif
        cell.latvec.e33 = 1.5;
        basis.initgrids(length, cell.latvec, 20.0);
        basis.initparameters(false, 20.0, 1, false);
        basis.setuptransform();
        basis.collect_local_pw();
        tpiba = ModuleBase::TWO_PI / length;
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
        const double density_value = 0.6 / basis.omega;
        density.assign(basis.nrxx, density_value);
        channel = density.data();
        charge.nspin = 1;
        charge.rho = &channel;
    }

    UnitCell cell;
    Atom atom;
    Charge charge;
    std::vector<double> density;
    double* channel = nullptr;
};

TEST_F(SccsPcc2dTest, VacuumLimitRetainsPointIonPccExactlyOnce)
{
    Input_para input;
    input.assume_isolated = "pcc_2d";
    input.sccs_preset = "vacuum";
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_EQ(config.boundary, ModuleSccs::Boundary::Pcc2d);
    elecstate::PotSccs solvent(&basis, config, solver);
    elecstate::PotPcc correction(&basis, elecstate::PotPcc::Dimension::slab, input.pcc_2d_axis);
    ModuleBase::matrix vacuum(1, basis.nrxx);
    correction.cal_v_eff(&charge, &cell, vacuum);
    const double energy = correction.get_energy();
    ModuleBase::matrix combined = vacuum;
    solvent.cal_v_eff(&charge, &cell, combined);
    EXPECT_NEAR(solvent.get_energy(), 0.0, 1e-12);
    const double total = correction.get_energy() + solvent.get_energy();
    EXPECT_NEAR(total, energy, 1e-12);
    ModuleBase::matrix point_force(cell.nat, 3);
    correction.add_force(cell, point_force);
    ModuleBase::matrix combined_force = point_force;
    solvent.add_solvation_force(cell, combined_force);
    for (int axis = 0; axis < 3; ++axis)
    {
        EXPECT_NEAR(combined_force(0, axis), point_force(0, axis), 1e-12);
    }
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_NEAR(combined(0, ir), vacuum(0, ir), 1e-12);
    }
}

TEST_F(SccsPcc2dTest, ChargedDielectricAddsOnlyReactionAndCavityTerms)
{
    Input_para input;
    input.assume_isolated = "pcc_2d";
    input.sccs_epsilon = 5.0;
    input.sccs_rho_min = 1e-3;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    const std::vector<unitcell::AtomData> atoms = unitcell::get_atom_data(cell.atoms, cell.ntype, cell.lat0);
    std::unique_ptr<ModuleSccs::CoulombOperator> coulomb;
    ASSERT_TRUE(elecstate::make_sccs_pcc_2d_operator(cell, basis, atoms, input.pcc_2d_axis, coulomb, error)) << error;
    std::vector<double> ions;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_density(atoms, basis, tpiba,
                     ModuleSccs::gaussian_ion_spread, ions, error)) << error;
    std::vector<double> solute_charge(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir) { solute_charge[ir] = ions[ir] - density[ir]; }
    const std::vector<double> cold;
    ModuleSccs::SccsResponse response;
    ASSERT_TRUE(ModuleSccs::solve_sccs_response(density, solute_charge, config.cavity, solver, cold,
                            basis, tpiba, *coulomb, response, error)) << error;
    ModuleSccs::FunctionalResult functional;
    ASSERT_TRUE(ModuleSccs::evaluate_functional(solute_charge, response, config, basis, tpiba,
                                                *coulomb, functional, error)) << error;
    elecstate::PotSccs solvent(&basis, config, solver);
    elecstate::PotPcc correction(&basis, elecstate::PotPcc::Dimension::slab, input.pcc_2d_axis);
    ModuleBase::matrix potential(1, basis.nrxx);
    correction.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix point_potential = potential;
    const double point_energy = correction.get_energy();
    solvent.cal_v_eff(&charge, &cell, potential);
    const double reaction_energy_rydberg = 2.0 * functional.reaction_energy;
    EXPECT_NEAR(solvent.get_energy(), reaction_energy_rydberg, 1e-12);
    const double combined_energy = solvent.get_energy() + correction.get_energy();
    const double expected_energy = point_energy + 2.0 * functional.reaction_energy;
    EXPECT_NEAR(combined_energy, expected_energy, 1e-12);
    std::vector<ModuleBase::Vector3<double>> reaction_force;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, functional.reaction_potential, basis,
                   tpiba, ModuleSccs::gaussian_ion_spread, reaction_force, error)) << error;
    ModuleBase::matrix point_force(cell.nat, 3);
    correction.add_force(cell, point_force);
    ModuleBase::matrix combined_force = point_force;
    solvent.add_solvation_force(cell, combined_force);
    for (int axis = 0; axis < 3; ++axis)
    {
        const double expected = point_force(0, axis) + 2.0 * reaction_force[0][axis];
        EXPECT_NEAR(combined_force(0, axis), expected, 1e-12);
    }
    const auto* electrostatic = solvent.solvent_electrostatic_potential();
    ASSERT_NE(electrostatic, nullptr);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const double expected = point_potential(0, ir) + 2.0 * functional.electron_potential[ir];
        EXPECT_NEAR(potential(0, ir), expected, 1e-12);
        const double expected_potential = -2.0 * functional.reaction_potential[ir];
        EXPECT_NEAR((*electrostatic)[ir], expected_potential, 1e-12);
    }
}

TEST_F(SccsPcc2dTest, OpenAxisIsExplicitAndInvalidGeometryPreservesOperator)
{
    Input_para input;
    input.assume_isolated = "pcc_2d";
    input.sccs_preset = "vacuum";
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    for (int axis = 0; axis < 3; ++axis)
    {
        input.pcc_2d_axis = axis;
        ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
        EXPECT_EQ(config.pcc_2d_axis, axis);
    }
    input.pcc_2d_axis = 3;
    EXPECT_FALSE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    const std::vector<unitcell::AtomData> atoms = unitcell::get_atom_data(cell.atoms, cell.ntype, cell.lat0);
    std::unique_ptr<ModuleSccs::CoulombOperator> coulomb;
    const int axis = 2;
    ASSERT_TRUE(elecstate::make_sccs_pcc_2d_operator(cell, basis, atoms, axis, coulomb, error));
    const ModuleSccs::CoulombOperator* original = coulomb.get();
    if (basis.poolrank == 0) { cell.latvec.e31 = 0.2; }
    EXPECT_FALSE(elecstate::make_sccs_pcc_2d_operator(cell, basis, atoms, axis, coulomb, error));
    EXPECT_EQ(coulomb.get(), original);
    EXPECT_FALSE(error.empty());
}

TEST_F(SccsPcc2dTest, FarFieldScreeningPreservesOriginalTolerance)
{
    Input_para input;
    input.assume_isolated = "pcc_2d";
    input.sccs_epsilon = 5.0;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    ModuleSccs::SccsResponse response;
    const double electrons = 0.6;
    const double ions = 1.0;
    response.far_field_polarization_charge = -0.32;
    EXPECT_TRUE(elecstate::validate_sccs_pcc_2d_screening(response, config, solver, electrons,
                                                         ions, basis.omega, error));
    response.far_field_polarization_charge += 5e-5;
    EXPECT_TRUE(elecstate::validate_sccs_pcc_2d_screening(response, config, solver, electrons,
                                                         ions, basis.omega, error));
    response.far_field_polarization_charge += 1e-3;
    EXPECT_FALSE(elecstate::validate_sccs_pcc_2d_screening(response, config, solver, electrons,
                                                          ions, basis.omega, error));
    EXPECT_NE(error.find("far-field polarization charge"), std::string::npos);
}
