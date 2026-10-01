#include "../common/charge_reduction.h"
#ifdef __MPI
#include "source_base/parallel_global.h"
#include <mpi.h>
#endif

#include "../sccs/sccs_pw_coulomb.h"
#include "../sccs/sccs_pw_nonel.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis.h"

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace
{

TEST(SccsPwNonel, EnergyDerivativeMatchesPotential)
{
    ModulePW::PW_Basis basis("cpu", "double");
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);
    const double length = 10.0;
    basis.initgrids(length, lattice, 30.0);
    basis.initparameters(false, 30.0, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();

    std::vector<std::complex<double>> solute_g(basis.npw);
    std::vector<std::complex<double>> direction_g(basis.npw);
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        if (std::abs(std::abs(basis.gdirect[ig].x) - 1.0) < 1.0e-12
            && std::abs(basis.gdirect[ig].y) < 1.0e-12
            && std::abs(basis.gdirect[ig].z) < 1.0e-12)
        {
            solute_g[ig] = 0.12;
            direction_g[ig] = 0.03;
        }
    }
    if (basis.ig_gge0 >= 0)
    {
        solute_g[basis.ig_gge0] = 0.5;
        direction_g[basis.ig_gge0] = 0.02;
    }
    std::vector<double> solute(basis.nrxx);
    std::vector<double> direction(basis.nrxx);
    basis.recip2real(solute_g.data(), solute.data());
    basis.recip2real(direction_g.data(), direction.data());

    const double volume_element = length * length * length / static_cast<double>(basis.nxyz);
    ModuleSccs::NonElectrostaticParameters parameters;
    parameters.surface_tension = 0.7;
    parameters.pressure = -0.04;
    parameters.surface_regularization = 1.0e-3;
    const ModuleSurchem::SerialChargeReduction reduction;
    const ModuleSccs::NonElectrostaticResult center
        = ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                    ModuleBase::TWO_PI / length,
                                                    volume_element,
                                                    parameters,
                                                    solute,
                                                    reduction);

    const double step = 1.0e-5;
    std::vector<double> plus(solute.size());
    std::vector<double> minus(solute.size());
    double analytic = 0.0;
    for (std::size_t index = 0; index < solute.size(); ++index)
    {
        plus[index] = solute[index] + step * direction[index];
        minus[index] = solute[index] - step * direction[index];
        analytic += center.boundary_potential[index] * direction[index] * volume_element;
    }
    const ModuleSccs::NonElectrostaticResult plus_result
        = ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                    ModuleBase::TWO_PI / length,
                                                    volume_element,
                                                    parameters,
                                                    plus,
                                                    reduction);
    const ModuleSccs::NonElectrostaticResult minus_result
        = ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                    ModuleBase::TWO_PI / length,
                                                    volume_element,
                                                    parameters,
                                                    minus,
                                                    reduction);
    const double finite_difference
        = ((plus_result.surface_energy + plus_result.volume_energy)
           - (minus_result.surface_energy + minus_result.volume_energy))
          / (2.0 * step);
    EXPECT_NEAR(finite_difference, analytic, 1.0e-7);
}

void setup_cubic_basis(ModulePW::PW_Basis& basis, const double length)
{
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    const ModuleBase::Matrix3 lattice(1.0, 0.0, 0.0,
                                      0.0, 1.0, 0.0,
                                      0.0, 0.0, 1.0);
    basis.initgrids(length, lattice, 30.0);
    basis.initparameters(false, 30.0, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();
}

TEST(SccsPwNonel, UniformSoluteHasNoRegularizedSurface)
{
    ModulePW::PW_Basis basis("cpu", "double");
    const double length = 10.0;
    setup_cubic_basis(basis, length);
    const double volume = length * length * length;
    const double volume_element = volume / static_cast<double>(basis.nxyz);
    ModuleSccs::NonElectrostaticParameters parameters;
    parameters.surface_tension = 0.02;
    parameters.pressure = 0.003;
    parameters.surface_regularization = 1.0e-6;
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> solute(basis.nrxx, 1.0);
    const ModuleSccs::NonElectrostaticResult result
        = ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                    ModuleBase::TWO_PI / length,
                                                    volume_element,
                                                    parameters,
                                                    solute,
                                                    reduction);
    EXPECT_NEAR(result.surface, 0.0, 1.0e-12);
    EXPECT_NEAR(result.volume, volume, 1.0e-9);
    EXPECT_NEAR(result.surface_energy, 0.0, 1.0e-14);
    EXPECT_NEAR(result.volume_energy, parameters.pressure * volume, 1.0e-10);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_NEAR(result.boundary_potential[ir], parameters.pressure, 1.0e-14);
    }
}

// Given the same gradient and surface derivative -div(grad s/|grad s|_r), the
// chain evaluation reproduces the spectral surface, volume and potential.
TEST(SccsPwNonel, ChainEvaluationMatchesSpectralForTheSameDerivatives)
{
    ModulePW::PW_Basis basis("cpu", "double");
    const double length = 10.0;
    setup_cubic_basis(basis, length);
    const double tpiba = ModuleBase::TWO_PI / length;
    const double volume_element = length * length * length / static_cast<double>(basis.nxyz);
    ModuleSccs::NonElectrostaticParameters parameters;
    parameters.surface_tension = 0.02;
    parameters.pressure = 0.003;
    parameters.surface_regularization = 1.0e-6;
    const ModuleSurchem::SerialChargeReduction reduction;
    std::vector<std::complex<double>> solute_g(basis.npw);
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        if (std::abs(std::abs(basis.gdirect[ig].y) - 1.0) < 1.0e-12
            && std::abs(basis.gdirect[ig].x) < 1.0e-12
            && std::abs(basis.gdirect[ig].z) < 1.0e-12)
        {
            solute_g[ig] = 0.2;
        }
    }
    if (basis.ig_gge0 >= 0)
    {
        solute_g[basis.ig_gge0] = 0.5;
    }
    std::vector<double> solute(basis.nrxx);
    basis.recip2real(solute_g.data(), solute.data());
    const std::vector<ModuleBase::Vector3<double>> gradient
        = ModuleSccs::periodic_gradient(solute, basis, tpiba);
    const double regularization_square
        = parameters.surface_regularization * parameters.surface_regularization;
    std::vector<std::complex<double>> divergence_g(basis.npw, std::complex<double>(0.0, 0.0));
    std::vector<double> component(basis.nrxx);
    std::vector<std::complex<double>> component_g(basis.npw);
    for (int d = 0; d < 3; ++d)
    {
        for (int ir = 0; ir < basis.nrxx; ++ir)
        {
            const double norm_square = gradient[ir].norm2() + regularization_square;
            component[ir] = gradient[ir][d] / std::sqrt(norm_square);
        }
        basis.real2recip(component.data(), component_g.data());
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            divergence_g[ig] += ModuleBase::IMAG_UNIT * tpiba * basis.gcar[ig][d] * component_g[ig];
        }
    }
    std::vector<double> surface_derivative(basis.nrxx);
    basis.recip2real(divergence_g.data(), surface_derivative.data());
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        surface_derivative[ir] = -surface_derivative[ir];
    }

    const ModuleSccs::NonElectrostaticResult spectral
        = ModuleSccs::evaluate_pw_non_electrostatic(basis, tpiba, volume_element, parameters,
                                                    solute, reduction);
    const ModuleSccs::NonElectrostaticResult chain
        = ModuleSccs::evaluate_chain_non_electrostatic(volume_element, parameters, solute,
                                                       gradient, surface_derivative, reduction);
    EXPECT_GT(spectral.surface, 1.0);
    EXPECT_NEAR(chain.surface, spectral.surface, 1.0e-12 * spectral.surface);
    EXPECT_NEAR(chain.volume, spectral.volume, 1.0e-12 * spectral.volume);
    EXPECT_NEAR(chain.surface_energy, spectral.surface_energy, 1.0e-12 * spectral.surface_energy);
    EXPECT_NEAR(chain.volume_energy, spectral.volume_energy, 1.0e-12 * spectral.volume_energy);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_NEAR(chain.boundary_potential[ir], spectral.boundary_potential[ir], 1.0e-12);
    }
    const std::vector<double> short_derivative(basis.nrxx - 1, 0.0);
    EXPECT_THROW(ModuleSccs::evaluate_chain_non_electrostatic(volume_element, parameters, solute,
                                                              gradient, short_derivative, reduction),
                 std::invalid_argument);
}

TEST(SccsPwNonel, RejectsMismatchedArraysAndRegularization)
{
    ModulePW::PW_Basis basis("cpu", "double");
    const double length = 10.0;
    setup_cubic_basis(basis, length);
    const double volume_element = length * length * length / static_cast<double>(basis.nxyz);
    ModuleSccs::NonElectrostaticParameters parameters;
    parameters.surface_regularization = 1.0e-6;
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> short_solute(basis.nrxx - 1, 1.0);
    EXPECT_THROW(ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                           ModuleBase::TWO_PI / length,
                                                           volume_element,
                                                           parameters,
                                                           short_solute,
                                                           reduction),
                 std::invalid_argument);
    const std::vector<double> solute(basis.nrxx, 1.0);
    parameters.surface_regularization = 0.0;
    EXPECT_THROW(ModuleSccs::evaluate_pw_non_electrostatic(basis,
                                                           ModuleBase::TWO_PI / length,
                                                           volume_element,
                                                           parameters,
                                                           solute,
                                                           reduction),
                 std::invalid_argument);
}

} // namespace

int main(int argc, char** argv)
{
#ifdef __MPI
    int process_count = 1;
    int thread_count = 1;
    int rank = 0;
    Parallel_Global::read_pal_param(argc, argv, process_count, thread_count, rank);
    POOL_WORLD = MPI_COMM_WORLD;
    KP_WORLD = MPI_COMM_NULL;
    INT_BGROUP = MPI_COMM_NULL;
    BP_WORLD = MPI_COMM_NULL;
    GRID_WORLD = MPI_COMM_NULL;
    DIAG_WORLD = MPI_COMM_NULL;
#endif
    testing::InitGoogleTest(&argc, argv);
    // Error-path tests throw inside timed functions and leave their
    // ModuleBase::timer entries running; production turns these exceptions
    // into WARNING_QUIT, so the timers are not under test here.
    ModuleBase::timer::disable();
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
