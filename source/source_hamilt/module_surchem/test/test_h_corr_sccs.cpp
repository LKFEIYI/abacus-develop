#ifdef __MPI
#include "source_base/parallel_global.h"
#include <mpi.h>
#endif

#include "../surchem.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_basis/module_pw/pw_basis.h"

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <vector>

namespace
{

TEST(HCorrSccs, DelaysActivationUntilDensityOrIterationThreshold)
{
    SurchemParameters parameters;
    parameters.use_sccs = true;
    parameters.start_drho = 1.0e-2;
    parameters.start_nmax = 5;

    surchem solvent;
    solvent.set_parameters(parameters);
    EXPECT_TRUE(solvent.uses_sccs());
    EXPECT_FALSE(solvent.sccs_is_active());
    EXPECT_FALSE(solvent.try_activate_sccs(1, 2.0e-2));
    EXPECT_FALSE(solvent.sccs_is_active());
    EXPECT_TRUE(solvent.try_activate_sccs(2, 9.0e-3));
    EXPECT_TRUE(solvent.sccs_is_active());
    EXPECT_FALSE(solvent.try_activate_sccs(3, 2.0e-2));

    solvent.set_parameters(parameters);
    EXPECT_FALSE(solvent.try_activate_sccs(4, 2.0e-2));
    EXPECT_TRUE(solvent.try_activate_sccs(5, 2.0e-2));
    EXPECT_TRUE(solvent.sccs_is_active());
}

TEST(HCorrSccs, KeepsDelayedActivationLatchedWhenPotentialStorageIsCleared)
{
    SurchemParameters parameters;
    parameters.use_sccs = true;
    parameters.start_drho = 1.0e-2;
    parameters.start_nmax = 5;

    surchem solvent;
    solvent.set_parameters(parameters);
    EXPECT_TRUE(solvent.try_activate_sccs(2, 9.0e-3));

    // LCAO rebuilds PotSurChem between ionic steps, whose destructor clears
    // the allocated potential storage. The one-way activation must survive it.
    solvent.clear();
    EXPECT_TRUE(solvent.sccs_is_active());
    EXPECT_FALSE(solvent.try_activate_sccs(1, 2.0e-2));
}

TEST(HCorrSccs, ConvertsHartreeResultToRydbergPotentialAndEnergy)
{
    ModulePW::PW_Basis basis("cpu", "double");
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);
    const double length = 10.0;
    const double volume = length * length * length;
    basis.initgrids(length, lattice, 20.0);
    basis.initparameters(false, 20.0, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();

    UnitCell cell;
    cell.lat0 = length;
    cell.latvec = lattice;
    cell.omega = volume;
    cell.tpiba = ModuleBase::TWO_PI / length;
    cell.tpiba2 = cell.tpiba * cell.tpiba;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = new Atom[1];
    cell.atoms[0].na = 1;
    cell.atoms[0].ncpp.zv = 1.0;
    cell.atoms[0].tau.push_back(ModuleBase::Vector3<double>(0.5, 0.5, 0.5));

    SurchemParameters parameters;
    parameters.use_sccs = true;
    parameters.expected_electron_count = 0.0;
    parameters.expected_ionic_charge = 1.0;
    parameters.sccs_config.cavity.density_min = 1.0e-2;
    parameters.sccs_config.cavity.density_max = 2.0e-2;
    parameters.sccs_config.cavity.epsilon_bulk = 5.0;
    parameters.sccs_config.surface_regularization = 1.0e-6;
    parameters.sccs_config.boundary = ModuleSccs::Boundary::Pcc0d;
    parameters.sccs_config.max_iterations = 100;
    parameters.sccs_config.mixing = 0.7;
    parameters.sccs_config.tolerance_rms = 1.0e-14;
    parameters.sccs_config.tolerance_max = 1.0e-14;

    surchem solvent;
    solvent.set_parameters(parameters);
    std::vector<double> electron_density(basis.nrxx, 0.0);
    const double* density_channels[1] = {electron_density.data()};
    std::vector<double> local_potential(basis.nrxx, 0.0);
    ModuleBase::matrix potential;
    solvent.v_correction_sccs(cell,
                              basis,
                              1,
                              density_channels,
                              local_potential.data(),
                              potential);

    const ModuleSccs::SccsResult& result = solvent.sccs_result();
    EXPECT_NEAR(result.charge.net_charge, 1.0, 1.0e-12);
    EXPECT_NEAR(result.point_solute_moments.charge, 1.0, 1.0e-12);
    EXPECT_NEAR(result.vacuum_pcc_energy,
                0.5 * 2.837297479480619 / length,
                1.0e-12);
    EXPECT_NEAR(surchem::Ael,
                2.0 * (result.electrostatic.reaction_energy
                       + result.vacuum_pcc_energy
                       + result.ionic_shape_pcc_energy),
                1.0e-14);
    EXPECT_NEAR(surchem::Acav,
                2.0 * (result.non_electrostatic.surface_energy
                       + result.non_electrostatic.volume_energy),
                1.0e-14);
    ASSERT_EQ(potential.nr, 1);
    ASSERT_EQ(potential.nc, basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_TRUE(std::isfinite(potential(0, ir)));
        EXPECT_NEAR(potential(0, ir), 2.0 * result.electron_potential_hartree[ir], 1.0e-14);
    }
}

TEST(HCorrSccs, AppliesNeutralPcc2dPointIonEnergyAndPotential)
{
    ModulePW::PW_Basis basis("cpu", "double");
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice(0.8, 0.0, 0.0,
                                      0.0, 1.2, 0.0,
                                      0.0, 0.0, 1.0);
    const double scale = 10.0;
    const double volume = 0.8 * 1.2 * scale * scale * scale;
    basis.initgrids(scale, lattice, 20.0);
    basis.initparameters(false, 20.0, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();

    UnitCell cell;
    cell.lat0 = scale;
    cell.latvec = lattice;
    cell.omega = volume;
    cell.tpiba = ModuleBase::TWO_PI / scale;
    cell.tpiba2 = cell.tpiba * cell.tpiba;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = new Atom[1];
    cell.atoms[0].na = 1;
    cell.atoms[0].ncpp.zv = 1.0;
    cell.atoms[0].tau.push_back(ModuleBase::Vector3<double>(0.4, 0.78, 0.5));

    SurchemParameters parameters;
    parameters.use_sccs = true;
    parameters.expected_electron_count = 1.0;
    parameters.expected_ionic_charge = 1.0;
    parameters.sccs_config.cavity.density_min = 1.0e-2;
    parameters.sccs_config.cavity.density_max = 2.0e-2;
    parameters.sccs_config.cavity.epsilon_bulk = 1.0;
    parameters.sccs_config.surface_regularization = 1.0e-6;
    parameters.sccs_config.boundary = ModuleSccs::Boundary::Pcc2d;
    parameters.sccs_config.max_iterations = 100;
    parameters.sccs_config.mixing = 0.7;
    parameters.sccs_config.tolerance_rms = 1.0e-14;
    parameters.sccs_config.tolerance_max = 1.0e-14;

    surchem solvent;
    solvent.set_parameters(parameters);
    std::vector<double> electron_density(basis.nrxx, 1.0 / volume);
    const double* density_channels[1] = {electron_density.data()};
    std::vector<double> local_potential(basis.nrxx, 0.0);
    ModuleBase::matrix potential;
    solvent.v_correction_sccs(cell,
                              basis,
                              1,
                              density_channels,
                              local_potential.data(),
                              potential);

    const ModuleSccs::SccsResult& result = solvent.sccs_result();
    const double dipole_y = (0.78 - 0.6) * scale;
    const double expected_energy = 2.0 * ModuleBase::PI * dipole_y * dipole_y / volume;
    EXPECT_NEAR(result.charge.net_charge, 0.0, 1.0e-12);
    EXPECT_NEAR(result.point_solute_moments_2d.charge, 0.0, 1.0e-12);
    EXPECT_NEAR(result.point_solute_moments_2d.dipole_y, dipole_y, 1.0e-12);
    EXPECT_NEAR(result.vacuum_pcc_energy, expected_energy, 1.0e-12);
    EXPECT_NEAR(result.electrostatic.reaction_energy, 0.0, 1.0e-14);
    EXPECT_NEAR(surchem::Ael, 2.0 * expected_energy, 1.0e-12);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_TRUE(std::isfinite(potential(0, ir)));
        EXPECT_NEAR(potential(0, ir),
                    2.0 * result.electron_potential_hartree[ir],
                    1.0e-14);
    }
}

TEST(HCorrSccs, AppliesChargedPcc2dEnergyAndPotential)
{
    ModulePW::PW_Basis basis("cpu", "double");
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice(0.8, 0.0, 0.0,
                                      0.0, 1.2, 0.0,
                                      0.0, 0.0, 1.0);
    const double scale = 10.0;
    const double volume = 0.8 * 1.2 * scale * scale * scale;
    basis.initgrids(scale, lattice, 20.0);
    basis.initparameters(false, 20.0, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();

    UnitCell cell;
    cell.lat0 = scale;
    cell.latvec = lattice;
    cell.omega = volume;
    cell.tpiba = ModuleBase::TWO_PI / scale;
    cell.tpiba2 = cell.tpiba * cell.tpiba;
    cell.ntype = 1;
    cell.nat = 1;
    cell.atoms = new Atom[1];
    cell.atoms[0].na = 1;
    cell.atoms[0].ncpp.zv = 1.0;
    cell.atoms[0].tau.push_back(ModuleBase::Vector3<double>(0.4, 0.78, 0.5));

    SurchemParameters parameters;
    parameters.use_sccs = true;
    parameters.expected_electron_count = 0.8;
    parameters.expected_ionic_charge = 1.0;
    parameters.normalization_tolerance = 1.0e-10;
    parameters.sccs_config.cavity.density_min = 1.0e-2;
    parameters.sccs_config.cavity.density_max = 2.0e-2;
    parameters.sccs_config.cavity.epsilon_bulk = 5.0;
    parameters.sccs_config.surface_regularization = 1.0e-6;
    parameters.sccs_config.boundary = ModuleSccs::Boundary::Pcc2d;
    parameters.sccs_config.max_iterations = 100;
    parameters.sccs_config.mixing = 0.7;
    parameters.sccs_config.tolerance_rms = 1.0e-14;
    parameters.sccs_config.tolerance_max = 1.0e-14;

    surchem solvent;
    solvent.set_parameters(parameters);
    std::vector<double> electron_density(basis.nrxx, 0.8 / volume);
    const double* density_channels[1] = {electron_density.data()};
    std::vector<double> local_potential(basis.nrxx, 0.0);
    ModuleBase::matrix potential;
    solvent.v_correction_sccs(cell,
                              basis,
                              1,
                              density_channels,
                              local_potential.data(),
                              potential);

    const ModuleSccs::SccsResult& result = solvent.sccs_result();
    EXPECT_NEAR(result.charge.net_charge, 0.2, 1.0e-12);
    EXPECT_NEAR(result.point_solute_moments_2d.charge, 0.2, 1.0e-12);
    EXPECT_NEAR(result.solute_moments_2d.charge, 0.2, 1.0e-12);
    EXPECT_NEAR(result.polarization_moments_2d.charge, -0.16, 1.0e-12);
    EXPECT_NEAR(result.screened_moments_2d.charge, 0.04, 1.0e-12);
    EXPECT_TRUE(std::isfinite(result.vacuum_pcc_energy));
    EXPECT_TRUE(std::isfinite(result.electrostatic.reaction_energy));
    const ModuleSccs::Pcc2dParameters pcc_parameters
        = ModuleSccs::pcc_2d_geometry(cell.latvec, cell.lat0, 1.0e-10).parameters;
    const double expected_ionic_shape_energy
        = ModuleBase::PI * result.polarization_moments_2d.charge
          * (result.solute_moments_2d.quadrupole_yy
             - result.point_solute_moments_2d.quadrupole_yy)
          / (pcc_parameters.periodic_area * pcc_parameters.cell_length_y);
    EXPECT_NEAR(result.ionic_shape_pcc_energy,
                expected_ionic_shape_energy,
                1.0e-14);
    EXPECT_NEAR(surchem::Ael,
                2.0 * (result.electrostatic.reaction_energy
                       + result.vacuum_pcc_energy
                       + result.ionic_shape_pcc_energy),
                1.0e-14);
    std::ostringstream diagnostics;
    solvent.write_sccs_diagnostics(diagnostics);
    const std::string diagnostic_text = diagnostics.str();
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC reaction_energy_hartree"),
              std::string::npos);
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC smooth_vacuum_pcc_energy_hartree"),
              std::string::npos);
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC point_vacuum_pcc_energy_hartree"),
              std::string::npos);
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC ionic_shape_pcc_energy_hartree"),
              std::string::npos);
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC electrostatic_energy_rydberg"),
              std::string::npos);
    EXPECT_NE(diagnostic_text.find("SCCS_DIAGNOSTIC screened_charge"),
              std::string::npos);
    std::ostringstream iteration_output;
    solvent.write_sccs_iteration(iteration_output);
    const std::string iteration_text = iteration_output.str();
    EXPECT_NE(iteration_text.find("SCCS_ITER "), std::string::npos);
    EXPECT_NE(iteration_text.find("SCCS_TIME/s "), std::string::npos);
    EXPECT_NE(iteration_text.find("E_SOL/Ry "), std::string::npos);
    EXPECT_NE(iteration_text.find(
                  "SCCS_ITER "
                  + std::to_string(result.response.polarization.iterations)),
              std::string::npos);
    std::istringstream iteration_stream(iteration_text);
    std::string iteration_label;
    int iteration_count = 0;
    std::string time_label;
    double elapsed_seconds = -1.0;
    std::string energy_label;
    double solvation_energy_rydberg = 0.0;
    iteration_stream >> iteration_label >> iteration_count
                     >> time_label >> elapsed_seconds
                     >> energy_label >> solvation_energy_rydberg;
    EXPECT_EQ(iteration_label, "SCCS_ITER");
    EXPECT_EQ(iteration_count, result.response.polarization.iterations);
    EXPECT_EQ(time_label, "SCCS_TIME/s");
    EXPECT_GE(elapsed_seconds, 0.0);
    EXPECT_EQ(energy_label, "E_SOL/Ry");
    EXPECT_NEAR(solvation_energy_rydberg, surchem::Ael + surchem::Acav, 1.0e-7);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_TRUE(std::isfinite(potential(0, ir)));
        EXPECT_NEAR(potential(0, ir),
                    2.0 * result.electron_potential_hartree[ir],
                    1.0e-14);
    }
}

} // namespace

int main(int argc, char** argv)
{
#ifdef __MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_split(MPI_COMM_WORLD, 0, 1, &POOL_WORLD);
#endif
    testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    MPI_Comm_free(&POOL_WORLD);
    MPI_Finalize();
#endif
    return result;
}
