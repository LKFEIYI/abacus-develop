#include "source_estate/module_pot/pot_sccs.h"
#include "source_estate/module_pot/pot_pcc.h"
#include "source_estate/module_pot/sccs_pcc_0d.h"
#include "source_cell/cell_tools.h"
#include "source_hamilt/module_sccs/sccs_coulomb.h"
#include "source_hamilt/module_sccs/sccs_functional.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/sccs_response.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include "source_io/module_parameter/input_parameter.h"

class SccsPcc0dTest : public SccsTest::PwTest
{
protected:
    void SetUp() override
    {
        SccsTest::PwTest::SetUp();
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
        density.assign(basis.nrxx, 0.6 / basis.omega);
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

TEST_F(SccsPcc0dTest, VacuumLimitRetainsPointIonPccExactlyOnce)
{
    Input_para input;
    input.assume_isolated = "pcc_0d";
    input.sccs_preset = "vacuum";
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    EXPECT_EQ(config.boundary, ModuleSccs::Boundary::Pcc0d);
    elecstate::PotSccs solvent(&basis, config, solver);
    elecstate::PotPcc correction(&basis);
    ModuleBase::matrix vacuum(1, basis.nrxx);
    correction.cal_v_eff(&charge, &cell, vacuum);
    const double energy = correction.get_energy();
    ModuleBase::matrix combined = vacuum;
    solvent.cal_v_eff(&charge, &cell, combined);
    EXPECT_NEAR(solvent.get_energy(), 0.0, 1e-12);
    const double total = correction.get_energy() + solvent.get_energy();
    EXPECT_NEAR(total, energy, 1e-12);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_NEAR(combined(0, ir), vacuum(0, ir), 1e-12);
    }
}

TEST_F(SccsPcc0dTest, ChargedDielectricAddsOnlyReactionAndCavityTerms)
{
    Input_para input;
    input.assume_isolated = "pcc_0d";
    input.sccs_epsilon = 5.0;
    ModuleSccs::SccsConfig config;
    ModuleSccs::PolarizationSolverParameters solver;
    ASSERT_TRUE(elecstate::make_sccs_config_from_input(input, config, solver, error));
    const std::vector<unitcell::AtomData> atoms = unitcell::get_atom_data(cell.atoms, cell.ntype, cell.lat0);
    std::unique_ptr<ModuleSccs::CoulombOperator> coulomb;
    ASSERT_TRUE(elecstate::make_sccs_pcc_0d_operator(cell, basis, atoms, coulomb, error)) << error;
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
    elecstate::PotPcc correction(&basis);
    ModuleBase::matrix potential(1, basis.nrxx);
    correction.cal_v_eff(&charge, &cell, potential);
    ModuleBase::matrix point_potential = potential;
    const double point_energy = correction.get_energy();
    solvent.cal_v_eff(&charge, &cell, potential);
    EXPECT_NEAR(solvent.get_energy(), 2.0 * functional.reaction_energy, 1e-12);
    const double combined_energy = solvent.get_energy() + correction.get_energy();
    const double expected_energy = point_energy + 2.0 * functional.reaction_energy;
    EXPECT_NEAR(combined_energy, expected_energy, 1e-12);
    const auto* electrostatic = solvent.solvent_electrostatic_potential();
    ASSERT_NE(electrostatic, nullptr);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const double expected = point_potential(0, ir) + 2.0 * functional.electron_potential[ir];
        EXPECT_NEAR(potential(0, ir), expected, 1e-12);
        EXPECT_NEAR((*electrostatic)[ir], -2.0 * functional.reaction_potential[ir], 1e-12);
    }
}
