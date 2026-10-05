#include "source_estate/module_pot/sccs_fixed_sources.h"
#include "source_estate/module_pot/pot_sccs.h"
#include "source_cell/cell_tools.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/test/sccs_test.h"
#include <limits>
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
        bool reused = true;
        ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error)) << error;
        EXPECT_FALSE(reused);
        std::vector<double> expected;
        ASSERT_TRUE(ModuleSccs::gaussian_ionic_density(atoms, basis, cell.tpiba,
                    ModuleSccs::gaussian_ion_spread, expected, error));
        EXPECT_EQ(cache.ionic_density(), expected);
        ASSERT_TRUE(ModuleSccs::gaussian_core_density(atoms, basis, cell.tpiba, config.core_spreads, expected, error));
        EXPECT_EQ(cache.core_density(), expected);
        ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
        EXPECT_TRUE(reused);
    }
    UnitCell cell;
    std::vector<unitcell::AtomData> atoms;
    ModuleSccs::SccsConfig config;
    elecstate::SccsFixedSources cache;
};

TEST_F(SccsFixedSourcesTest, ReusesSourcesAndInvalidatesAtomAndPerAtomWidthChanges)
{
    EXPECT_TRUE(cache.ionic_density().empty());
    rebuild_and_compare();
    const double* stored_density = cache.ionic_density().data();
    bool reused = false;
    config.cavity.epsilon_bulk = 78.3; // Not an input to fixed sources.
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
    EXPECT_EQ(cache.ionic_density().data(), stored_density);
    atoms[0].position.x += 0.1;
    rebuild_and_compare();
    atoms[1].valence_charge = 5.0;
    rebuild_and_compare();
    atoms[1].mass = 17.0;
    rebuild_and_compare();
    atoms[0].atomic_number = 2;
    rebuild_and_compare();
    config.core_spreads[0] = 0.0;
    rebuild_and_compare();
    config.core_spreads = {0.5}; // Singleton automatic exclusion uses Z and zv.
    rebuild_and_compare();
    atoms[0].atomic_number = 1;
    rebuild_and_compare();
    config.core_electrons = false;
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_FALSE(reused);
    EXPECT_TRUE(cache.core_density().empty());
    config.core_spreads = {0.8}; // Unused widths do not invalidate electronic mode.
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
}

TEST_F(SccsFixedSourcesTest, InvalidatesCellCutoffAndInPlaceReciprocalGeometry)
{
    rebuild_and_compare();
    cell.latvec.e12 = 0.1;
    rebuild_and_compare();
    cell.lat0 += 0.1;
    rebuild_and_compare();
    cell.omega += 1.0;
    rebuild_and_compare();
    cell.tpiba += 0.01;
    rebuild_and_compare();
    basis.omega += 1.0;
    rebuild_and_compare();
    basis.ggecut += 0.1;
    rebuild_and_compare();
    ASSERT_GT(basis.npw, 0);
    basis.gcar[0].x += 0.01;
    rebuild_and_compare();
    basis.gg[0] += 0.01;
    rebuild_and_compare();
}

TEST_F(SccsFixedSourcesTest, OneRankMissRebuildsAllRanksAndFailurePreservesCache)
{
    rebuild_and_compare();
    if (SccsTest::pool_rank == 0) { atoms[0].mass += 1.0; }
    bool reused = true;
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_FALSE(reused);
    const std::vector<double> saved = cache.ionic_density();
    const double* stored_density = cache.ionic_density().data();
    const double old_x = atoms[0].position.x;
    if (SccsTest::pool_rank == 0) { atoms[0].position.x = std::numeric_limits<double>::quiet_NaN(); }
    reused = true;
    EXPECT_FALSE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
    EXPECT_EQ(cache.ionic_density(), saved);
    EXPECT_EQ(cache.ionic_density().data(), stored_density);
    atoms[0].position.x = old_x;
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
    const int old_plane_count = basis.nplane;
    if (SccsTest::pool_rank == 0) { ++basis.nplane; }
    EXPECT_FALSE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_EQ(cache.ionic_density(), saved);
    basis.nplane = old_plane_count;
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
}

TEST_F(SccsFixedSourcesTest, InvalidFullWidthsPreserveBothSourcesAndRecovery)
{
    rebuild_and_compare();
    const auto ions = cache.ionic_density();
    const auto core = cache.core_density();
    const auto widths = config.core_spreads;
    if (SccsTest::pool_rank == 0) { config.core_spreads.push_back(0.3); }
    bool reused = true;
    EXPECT_FALSE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_EQ(cache.ionic_density(), ions);
    EXPECT_EQ(cache.core_density(), core);
    EXPECT_TRUE(reused);
    config.core_spreads = widths;
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_TRUE(reused);
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
    std::vector<double> density(basis.nrxx, 0.6 / basis.omega);
    double* channel = density.data();
    Charge charge;
    charge.nspin = 1;
    charge.rho = &channel;
    ASSERT_TRUE(ModuleSccs::make_sccs_config(ModuleSccs::Preset::WaterNeutral, config, error));
    config.boundary = ModuleSccs::Boundary::Pcc0d;
    config.cavity.epsilon_bulk = 5.0;
    config.core_electrons = true;
    config.core_spreads = {0.5};
    ModuleSccs::PolarizationSolverParameters solver;
    solver.check_fixed_point = true;
    solver.tolerance_rms = 1e-12;
    solver.tolerance_max = 1e-12;
    elecstate::PotSccs cached(&basis, config, solver);
    for (int step = 0; step < 3; ++step)
    {
        if (step == 1) { for (double& value : density) { value *= 0.95; } }
        if (step == 2) { atom.tau[0].x += 0.01; }
        elecstate::PotSccs fresh(&basis, config, solver);
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
        cached.write_correction_iteration(output, 2, 1e-5, 0.0);
        const char* label = step == 1 ? "CACHED_SOURCES 1" : "CACHED_SOURCES 0";
        EXPECT_NE(output.str().find(label), std::string::npos);
    }
}

TEST_F(SccsFixedSourcesTest, RebuildsForAChangedPwGridAndBasisInstance)
{
    rebuild_and_compare();
    ModulePW::PW_Basis other("cpu", "double");
#ifdef __MPI
    other.initmpi(SccsTest::pool_size, SccsTest::pool_rank, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice;
    other.initgrids(length, lattice, 40.0);
    other.initparameters(false, 40.0, 1, false);
    other.setuptransform();
    other.collect_local_pw();
    ASSERT_NE(other.nxyz, basis.nxyz);
    bool reused = true;
    ASSERT_TRUE(cache.update(cell, other, atoms, config, reused, error));
    EXPECT_FALSE(reused);
    EXPECT_EQ(cache.positions().size(), static_cast<std::size_t>(other.nrxx));
    std::vector<double> expected;
    ASSERT_TRUE(ModuleSccs::gaussian_ionic_density(atoms, other, cell.tpiba,
                ModuleSccs::gaussian_ion_spread, expected, error));
    EXPECT_EQ(cache.ionic_density(), expected);
    ASSERT_TRUE(cache.update(cell, other, atoms, config, reused, error));
    EXPECT_TRUE(reused);
    ASSERT_TRUE(cache.update(cell, basis, atoms, config, reused, error));
    EXPECT_FALSE(reused);
    EXPECT_EQ(cache.positions().size(), static_cast<std::size_t>(basis.nrxx));
}
