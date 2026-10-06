#include "source_estate/module_pot/sccs_fixed_sources.h"
#include "source_estate/module_pot/pot_sccs.h"
#include "source_cell/cell_tools.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include <sstream>

class SccsFixedSourcesTest : public SccsTest::PwTest
{
protected:
    void SetUp() override
    {
        SccsTest::PwTest::SetUp();
        cell.lat0 = length;
        cell.tpiba = tpiba;
        cell.omega = basis.omega;
        unitcell::AtomData atom;
        atom.position = {2.5, 2.5, 2.5};
        atom.mass = 1.0;
        atom.valence_charge = 1.0;
        atom.atomic_number = 1;
        atoms.push_back(atom);
        atom.position.x = 4.0;
        atom.mass = 16.0;
        atom.valence_charge = 6.0;
        atom.atomic_number = 8;
        atoms.push_back(atom);
        config.core_electrons = true;
        config.core_spreads = {0.5, 0.7};
    }
    void rebuild_and_compare()
    {
        EXPECT_FALSE(cache.update(cell, basis, atoms, config));
        std::vector<double> expected;
        ModuleSccs::gaussian_ionic_density(atoms, basis, cell.tpiba,
                                           ModuleSccs::gaussian_ion_spread, expected);
        EXPECT_EQ(cache.ionic_density(), expected);
        ModuleSccs::gaussian_core_density(atoms, basis, cell.tpiba, config.core_spreads, expected);
        EXPECT_EQ(cache.core_density(), expected);
        EXPECT_TRUE(cache.update(cell, basis, atoms, config));
    }
    UnitCell cell;
    std::vector<unitcell::AtomData> atoms;
    ModuleSccs::SccsConfig config;
    elecstate::SccsFixedSources cache;
};

TEST_F(SccsFixedSourcesTest, ReusesSourcesUntilTheAtomsChange)
{
    EXPECT_TRUE(cache.ionic_density().empty());
    rebuild_and_compare();
    const double* stored_density = cache.ionic_density().data();
    EXPECT_TRUE(cache.update(cell, basis, atoms, config));
    EXPECT_EQ(cache.ionic_density().data(), stored_density);
    EXPECT_EQ(cache.positions().size(), static_cast<std::size_t>(basis.nrxx));
    atoms[1].mass = 17.0; // Not an input to the Gaussians.
    EXPECT_TRUE(cache.update(cell, basis, atoms, config));
    atoms[0].position.x += 0.1;
    rebuild_and_compare();
    atoms[1].valence_charge = 5.0;
    rebuild_and_compare();
    atoms[0].atomic_number = 2; // Changes the singleton automatic exclusion.
    rebuild_and_compare();
}

TEST_F(SccsFixedSourcesTest, CachedHostMatchesFreshFullCavityEnergyPotentialAndForce)
{
    Atom atom;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = &atom;
    atom.na = 1;
    atom.mass = 1.0;
    atom.ncpp.zv = 1.0;
    atom.tau = {ModuleBase::Vector3<double>(0.25, 0.25, 0.25)};
    const double density_value = 0.6 / basis.omega;
    std::vector<double> density(basis.nrxx, density_value);
    double* channel = density.data();
    Charge charge;
    charge.nspin = 1;
    charge.rho = &channel;
    config = ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral);
    config.boundary = ModuleSccs::Boundary::Pcc0d;
    config.cavity.epsilon_bulk = 5.0;
    config.core_electrons = true;
    config.core_spreads = {0.5};
    ModuleSccs::PolarizationSolverParameters solver;
    solver.check_fixed_point = true;
    solver.tolerance_rms = 1e-12;
    solver.tolerance_max = 1e-12;
    elecstate::PotSccs cached(&basis, config, solver, false);
    for (int step = 0; step < 3; ++step)
    {
        if (step == 1) { for (double& value : density) { value *= 0.95; } }
        if (step == 2) { atom.tau[0].x += 0.01; }
        elecstate::PotSccs fresh(&basis, config, solver, false);
        ModuleBase::matrix first(1, basis.nrxx);
        ModuleBase::matrix second(1, basis.nrxx);
        cached.cal_v_eff(&charge, &cell, first);
        fresh.cal_v_eff(&charge, &cell, second);
        EXPECT_NEAR(cached.get_energy(), fresh.get_energy(), 1e-11);
        for (int i = 0; i < basis.nrxx; ++i) { EXPECT_NEAR(first(0, i), second(0, i), 1e-9); }
        ModuleBase::matrix force1(1, 3);
        ModuleBase::matrix force2(1, 3);
        cached.add_solvation_force(cell, force1);
        fresh.add_solvation_force(cell, force2);
        for (int i = 0; i < 3; ++i) { EXPECT_NEAR(force1(0, i), force2(0, i), 1e-9); }
        std::ostringstream output;
        cached.write_iteration_output(output, 2, 1e-5, 0.0);
        const char* label = step == 1 ? "CACHED_SOURCES 1" : "CACHED_SOURCES 0";
        EXPECT_NE(output.str().find(label), std::string::npos);
    }
}
