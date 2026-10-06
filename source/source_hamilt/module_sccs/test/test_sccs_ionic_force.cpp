#include "sccs_test.h"
#include "../sccs_ionic_force.h"
#include "../sccs_ionic_charge.h"
#include "../sccs_response.h"
#include "../sccs_parameters.h"
#include "../sccs_functional.h"

#include "source_cell/cell_tools.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "../sccs_pcc_0d_coulomb.h"
#include <limits>

using SccsIonicForceTest = SccsTest::PwTest;

TEST_F(SccsIonicForceTest, AnalyticFourierForceAndGaugeInvariance)
{
    std::vector<unitcell::AtomData> atoms(1);
    atoms[0].valence_charge = 2.0;
    atoms[0].position.x = length / 4.0;
    std::vector<double> potential = cosine_mode(0);
    const double spread = ModuleSccs::gaussian_ion_spread;
    std::vector<ModuleBase::Vector3<double>> forces;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, spread, forces, error)) << error;
    const double exponent = -0.25 * spread * spread * tpiba * tpiba;
    const double expected = 2.0 * tpiba * std::exp(exponent);
    EXPECT_NEAR(forces[0].x, expected, 1e-12);
    EXPECT_NEAR(forces[0].y, 0.0, 1e-12);
    EXPECT_NEAR(forces[0].z, 0.0, 1e-12);
    atoms[0].position.x += length;
    for (double& value : potential) { value += 4.0; }
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, spread, forces, error)) << error;
    EXPECT_NEAR(forces[0].x, expected, 1e-12);
    potential.assign(basis.nrxx, 3.0);
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, spread, forces, error)) << error;
    EXPECT_NEAR(forces[0].x, 0.0, 1e-12);
    EXPECT_NEAR(forces[0].y, 0.0, 1e-12);
    EXPECT_NEAR(forces[0].z, 0.0, 1e-12);
}

TEST_F(SccsIonicForceTest, ReactionEnergyFiniteDifferenceAtFixedNonuniformCavity)
{
    std::vector<unitcell::AtomData> atoms(2);
    atoms[0].valence_charge = 1.0;
    atoms[0].position = ModuleBase::Vector3<double>(2.1, 3.2, 4.3);
    atoms[1].valence_charge = 1.0;
    atoms[1].position = ModuleBase::Vector3<double>(6.4, 5.1, 2.7);
    std::vector<double> density = cosine_mode(0);
    for (double& value : density) { value = (2.0 / basis.omega) * (1.0 + 0.1 * value); }
    ModuleSccs::SccsConfig config;
    config.cavity.density_min = 1e-3;
    config.cavity.density_max = 3e-3;
    config.cavity.epsilon_bulk = 5.0;
    config.surface_regularization = 1e-8;
    ModuleSccs::PolarizationSolverParameters solver;
    solver.tolerance_rms = 1e-13;
    solver.tolerance_max = 1e-12;
    const double spread = ModuleSccs::gaussian_ion_spread;
    const std::vector<double> cold_start;
    auto evaluate = [&](const std::vector<unitcell::AtomData>& displaced,
                        ModuleSccs::FunctionalResult& functional) -> bool {
        std::vector<double> charge;
        if (!ModuleSccs::gaussian_ionic_density(displaced, basis, tpiba, spread, charge, error)) { return false; }
        for (int ir = 0; ir < basis.nrxx; ++ir) { charge[ir] -= density[ir]; }
        ModuleSccs::SccsResponse response;
        if (!ModuleSccs::solve_sccs_response(density, charge, config.cavity, solver, cold_start,
                                          basis, tpiba, response, error)) { return false; }
        return ModuleSccs::evaluate_functional(charge, response, config, basis, tpiba, functional, error);
    };
    ModuleSccs::FunctionalResult baseline;
    ASSERT_TRUE(evaluate(atoms, baseline)) << error;
    std::vector<ModuleBase::Vector3<double>> forces;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, baseline.reaction_potential, basis,
                                               tpiba, spread, forces, error)) << error;
    const double step = 1e-4;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            std::vector<unitcell::AtomData> plus = atoms;
            std::vector<unitcell::AtomData> minus = atoms;
            plus[ia].position[axis] += step;
            minus[ia].position[axis] -= step;
            ModuleSccs::FunctionalResult positive;
            ModuleSccs::FunctionalResult negative;
            ASSERT_TRUE(evaluate(plus, positive)) << error;
            ASSERT_TRUE(evaluate(minus, negative)) << error;
            const double finite_difference = -(positive.reaction_energy - negative.reaction_energy) / (2.0 * step);
            EXPECT_NEAR(forces[ia][axis], finite_difference, 1e-8);
        }
    }
    config.cavity.epsilon_bulk = 1.0;
    ASSERT_TRUE(evaluate(atoms, baseline)) << error;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, baseline.reaction_potential, basis,
                                               tpiba, spread, forces, error)) << error;
    for (const ModuleBase::Vector3<double>& force : forces)
    {
        EXPECT_NEAR(force.x, 0.0, 1e-12);
        EXPECT_NEAR(force.y, 0.0, 1e-12);
        EXPECT_NEAR(force.z, 0.0, 1e-12);
    }
}

TEST_F(SccsIonicForceTest, DistributedInvalidInputPreservesOutput)
{
    std::vector<unitcell::AtomData> atoms(1);
    atoms[0].valence_charge = 1.0;
    std::vector<double> potential(basis.nrxx, 0.0);
    if (basis.poolrank == 0) { atoms[0].position.x = std::numeric_limits<double>::quiet_NaN(); }
    const ModuleBase::Vector3<double> sentinel(7.0, 8.0, 9.0);
    std::vector<ModuleBase::Vector3<double>> forces(1, sentinel);
    EXPECT_FALSE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, 0.5, forces, error));
    EXPECT_DOUBLE_EQ(forces[0].x, 7.0);
    atoms[0].position.x = 0.0;
    EXPECT_FALSE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, 0.0, forces, error));
    EXPECT_DOUBLE_EQ(forces[0].y, 8.0);
    if (basis.poolrank == 0 && basis.nrxx > 0) { potential[0] = std::numeric_limits<double>::infinity(); }
    EXPECT_FALSE(ModuleSccs::gaussian_ionic_force(atoms, potential, basis, tpiba, 0.5, forces, error));
    EXPECT_DOUBLE_EQ(forces[0].z, 9.0);
}

TEST_F(SccsIonicForceTest, FullCavityForceTracksPerAtomWidthAndDisabling)
{
    std::vector<unitcell::AtomData> atoms(2);
    atoms[0].valence_charge = 6.0;
    atoms[1].valence_charge = 1.0;
    for (unitcell::AtomData& atom : atoms) { atom.position.x = length / 4.0; }
    const std::vector<double> potential = cosine_mode(0);
    std::vector<ModuleBase::Vector3<double>> force;
    std::vector<double> widths = {0.8, 0.0};
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, potential, basis, tpiba, widths, force, error));
    const double gaussian_exponent = -0.25 * 0.8 * 0.8 * tpiba * tpiba;
    const double expected = 6.0 * tpiba * std::exp(gaussian_exponent);
    EXPECT_NEAR(force[0].x, expected, 1e-12);
    EXPECT_DOUBLE_EQ(force[1].x, 0.0);
    widths = {0.8};
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, potential, basis, tpiba, widths, force, error));
    const double expected_hydrogen_force = expected / 6.0;
    EXPECT_NEAR(force[1].x, expected_hydrogen_force, 1e-12);
    widths = {0.0};
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, potential, basis, tpiba, widths, force, error));
    EXPECT_DOUBLE_EQ(force[0].x, 0.0);
    EXPECT_DOUBLE_EQ(force[1].x, 0.0);
}

TEST_F(SccsIonicForceTest, FullCavityPccLowpassTotalCorrectionForceFiniteDifference)
{
    std::vector<unitcell::AtomData> atoms(2);
    atoms[0].valence_charge = 1.0;
    atoms[0].position = ModuleBase::Vector3<double>(3.1, 3.2, 4.3);
    atoms[1].valence_charge = 1.0;
    atoms[1].position = ModuleBase::Vector3<double>(6.4, 5.1, 5.7);
    const ModuleBase::Matrix3 lattice;
    std::vector<ModuleBase::Vector3<double>> positions;
    ASSERT_TRUE(ModulePW::grid_positions(basis, lattice, length, positions, error));
    const double half = 0.5 * length;
    const ModuleBase::Vector3<double> center(half, half, half);
    for (auto& position : positions) { position -= center; }
    elecstate::Pcc0dParameters parameters;
    parameters.length = length;
    ModuleSccs::Pcc0dCoulombOperator coulomb(basis, tpiba, positions, parameters);
    ModuleSccs::SccsConfig config;
    config.boundary = ModuleSccs::Boundary::Pcc0d;
    config.core_electrons = true;
    config.core_spreads = {2.0, 0.0};
    config.cavity.epsilon_bulk = 5.0;
    config.cavity.density_min = 1e-4;
    config.cavity.density_max = 5e-3;
    config.cavity.lowpass_p1 = basis.ggecut;
    config.cavity.lowpass_p2 = 0.5;
    config.surface_regularization = 1e-8;
    config.surface_tension = 1e-5;
    config.pressure = 1e-6;
    ModuleSccs::PolarizationSolverParameters solver;
    solver.tolerance_rms = 1e-13;
    solver.tolerance_max = 1e-13;
    const std::vector<double> cold;
    auto evaluate = [&](const std::vector<unitcell::AtomData>& displaced,
                        ModuleSccs::FunctionalResult& functional) {
        std::vector<double> ions;
        std::vector<double> core;
        if (!ModuleSccs::gaussian_ionic_density(displaced, basis, tpiba,
                ModuleSccs::gaussian_ion_spread, ions, error)) { return false; }
        if (!ModuleSccs::gaussian_core_density(displaced, basis, tpiba,
                config.core_spreads, core, error)) { return false; }
        std::vector<double> cavity_density(basis.nrxx);
        for (int ir = 0; ir < basis.nrxx; ++ir)
        {
            const double electron_density = 2.0 / basis.omega;
            cavity_density[ir] = electron_density + core[ir];
            ions[ir] -= electron_density;
        }
        ModuleSccs::SccsResponse response;
        if (!ModuleSccs::solve_sccs_response(cavity_density, ions, config.cavity, solver,
                cold, basis, tpiba, coulomb, response, error)) { return false; }
        return ModuleSccs::evaluate_functional(ions, response, config, basis, tpiba,
                                               coulomb, functional, error);
    };
    ModuleSccs::FunctionalResult baseline;
    ASSERT_TRUE(evaluate(atoms, baseline)) << error;
    std::vector<ModuleBase::Vector3<double>> ionic;
    std::vector<ModuleBase::Vector3<double>> core;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_force(atoms, baseline.reaction_potential, basis,
                tpiba, ModuleSccs::gaussian_ion_spread, ionic, error));
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, baseline.cavity_potential, basis,
                tpiba, config.core_spreads, core, error));
    const double step = 1e-4;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            auto plus = atoms;
            auto minus = atoms;
            plus[ia].position[axis] += step;
            minus[ia].position[axis] -= step;
            ModuleSccs::FunctionalResult positive;
            ModuleSccs::FunctionalResult negative;
            ASSERT_TRUE(evaluate(plus, positive)) << error;
            ASSERT_TRUE(evaluate(minus, negative)) << error;
            const double energy_plus = positive.reaction_energy + positive.surface_energy + positive.volume_energy;
            const double energy_minus = negative.reaction_energy + negative.surface_energy + negative.volume_energy;
            const double finite_difference = -(energy_plus - energy_minus) / (2.0 * step);
            const double predicted = ionic[ia][axis] + core[ia][axis];
            EXPECT_NEAR(predicted, finite_difference, 1e-8);
        }
    }
}

TEST_F(SccsIonicForceTest, SingletonAndExplicitListUseSameDensityAndForceSelection)
{
    std::vector<unitcell::AtomData> atoms(2);
    atoms[0].atomic_number = 8;
    atoms[0].valence_charge = 6.0;
    atoms[1].atomic_number = 1;
    atoms[1].valence_charge = 1.0;
    for (auto& atom : atoms) { atom.position.x = length / 4.0; }
    const std::vector<double> potential = cosine_mode(0);
    std::vector<ModuleBase::Vector3<double>> forces;
    std::vector<double> widths = {0.8};
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, potential, basis, tpiba, widths, forces, error));
    EXPECT_DOUBLE_EQ(forces[1].x, 0.0);
    const double oxygen_force = forces[0].x;
    widths = {0.8, 0.8};
    ASSERT_TRUE(ModuleSccs::gaussian_core_force(atoms, potential, basis, tpiba, widths, forces, error));
    EXPECT_DOUBLE_EQ(forces[0].x, oxygen_force);
    const double hydrogen_force = oxygen_force / 6.0;
    EXPECT_NEAR(forces[1].x, hydrogen_force, 1e-12);
}
