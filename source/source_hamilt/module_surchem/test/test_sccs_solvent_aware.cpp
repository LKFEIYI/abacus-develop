#ifdef __MPI
#include "source_base/parallel_global.h"
#include <mpi.h>
#endif

#include "../common/charge_reduction.h"
#include "../sccs/sccs_cavity.h"
#include "../sccs/sccs_pw_coulomb.h"
#include "../sccs/sccs_solvent_aware.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace
{

const ModuleBase::Matrix3 cubic_lattice(1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0);

void setup_cubic_basis(ModulePW::PW_Basis& basis, const double length, const double ecut)
{
#ifdef __MPI
    basis.initmpi(1, 0, POOL_WORLD);
#endif
    basis.initgrids(length, cubic_lattice, ecut);
    basis.initparameters(false, ecut, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();
}

// Minimum-image displacement of grid point ir from the origin in a cube.
ModuleBase::Vector3<double> cubic_displacement(const ModulePW::PW_Basis& basis,
                                               const double length,
                                               const int ir)
{
    const int ix = ir / (basis.ny * basis.nplane);
    const int iy = ir / basis.nplane - ix * basis.ny;
    const int iz = ir % basis.nplane + basis.startz_current;
    ModuleBase::Vector3<double> r(length * ix / basis.nx, length * iy / basis.ny,
                                  length * iz / basis.nz);
    for (int d = 0; d < 3; ++d)
    {
        r[d] -= length * std::round(r[d] / length);
    }
    return r;
}

// Smooth sphere 1/2 erfc((|r - center| - radius)/spread) around a point given
// as a displacement from the origin.
std::vector<double> sphere(const ModulePW::PW_Basis& basis,
                           const double length,
                           const ModuleBase::Vector3<double>& center,
                           const double radius,
                           const double spread)
{
    std::vector<double> values(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        ModuleBase::Vector3<double> r = cubic_displacement(basis, length, ir) - center;
        for (int d = 0; d < 3; ++d)
        {
            r[d] -= length * std::round(r[d] / length);
        }
        values[ir] = 0.5 * std::erfc((r.norm() - radius) / spread);
    }
    return values;
}

std::vector<double> spectral_laplacian(const std::vector<double>& values,
                                       const ModulePW::PW_Basis& basis,
                                       const double tpiba)
{
    std::vector<std::complex<double>> values_g(basis.npw);
    basis.real2recip(values.data(), values_g.data());
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        values_g[ig] *= -tpiba * tpiba * basis.gg[ig];
    }
    std::vector<double> laplacian(values.size());
    basis.recip2real(values_g.data(), laplacian.data());
    return laplacian;
}

// Spectral Hessian components of values in the hessian_axes order.
std::vector<std::vector<double>> spectral_hessian(const std::vector<double>& values,
                                                  const ModulePW::PW_Basis& basis,
                                                  const double tpiba)
{
    std::vector<std::complex<double>> values_g(basis.npw);
    basis.real2recip(values.data(), values_g.data());
    std::vector<std::vector<double>> hessian(ModuleSccs::hessian_component_count);
    std::vector<std::complex<double>> component_g(basis.npw);
    for (int component = 0; component < ModuleSccs::hessian_component_count; ++component)
    {
        int first = 0;
        int second = 0;
        ModuleSccs::hessian_axes(component, first, second);
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            component_g[ig] = -tpiba * tpiba * basis.gcar[ig][first] * basis.gcar[ig][second]
                              * values_g[ig];
        }
        hessian[component].resize(values.size());
        basis.recip2real(component_g.data(), hessian[component].data());
    }
    return hessian;
}

double grid_sum(const std::vector<double>& left, const std::vector<double>& right)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        sum += left[i] * right[i];
    }
    return sum;
}

// Normalized transform of the isolated probe, 4 pi int r^2 p(r) j0(G r) dr
// over the same cut at width + 5 spread, divided by its G = 0 value.
double isolated_probe_transform(const double g, const double width, const double spread)
{
    const double cutoff = width + 5.0 * spread;
    const int intervals = 40000;
    const double step = cutoff / intervals;
    double value = 0.0;
    double norm = 0.0;
    for (int i = 0; i <= intervals; ++i)
    {
        const double r = i * step;
        const double weight = (i == 0 || i == intervals) ? 1.0 : (i % 2 == 1 ? 4.0 : 2.0);
        const double radial = r * r * std::erfc((r - width) / spread);
        const double bessel = g * r > 1.0e-12 ? std::sin(g * r) / (g * r) : 1.0;
        value += weight * radial * bessel;
        norm += weight * radial;
    }
    return value / norm;
}

// Probe kernel entry of the first shell along x, G = 2 pi / length.
double first_shell_kernel(const ModulePW::PW_Basis& basis, const std::vector<double>& kernel)
{
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        if (std::abs(basis.gdirect[ig].x - 1.0) < 1.0e-12 && std::abs(basis.gdirect[ig].y) < 1.0e-12
            && std::abs(basis.gdirect[ig].z) < 1.0e-12)
        {
            return kernel[ig];
        }
    }
    throw std::runtime_error("first G shell not found");
}

ModuleSccs::SolventAwareParameters probe_parameters(const double solvent_radius)
{
    ModuleSccs::SolventAwareParameters parameters;
    parameters.solvent_radius = solvent_radius;
    return parameters;
}

TEST(SccsSolventAware, RejectsParametersOutsideTheEnvironRanges)
{
    ModuleSccs::SolventAwareParameters parameters;
    EXPECT_NO_THROW(ModuleSccs::validate_solvent_aware_parameters(parameters));
    EXPECT_FALSE(ModuleSccs::uses_solvent_aware(parameters));
    parameters.solvent_radius = 3.0;
    EXPECT_NO_THROW(ModuleSccs::validate_solvent_aware_parameters(parameters));
    EXPECT_TRUE(ModuleSccs::uses_solvent_aware(parameters));

    ModuleSccs::SolventAwareParameters invalid = parameters;
    invalid.solvent_radius = -1.0;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid = parameters;
    invalid.radial_scale = 0.9;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid = parameters;
    invalid.radial_spread = 0.0;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid = parameters;
    invalid.filling_threshold = 0.0;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid.filling_threshold = 1.0;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid = parameters;
    invalid.filling_spread = 0.0;
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
    invalid = parameters;
    invalid.filling_spread = std::nan("");
    EXPECT_THROW(ModuleSccs::validate_solvent_aware_parameters(invalid), std::invalid_argument);
}

// In a cell wider than the probe the kernel is the transform of Environ's
// minimum-image probe, normalized on the grid, and it keeps constants.
TEST(SccsSolventAware, WideCellProbeMatchesEnvironMinimumImageProbe)
{
    const double length = 24.0;
    ModulePW::PW_Basis basis("cpu", "double");
    // At 120 Ry the grid error of the sampled erfc edge is below 1e-10.
    setup_cubic_basis(basis, length, 120.0);
    const ModuleSccs::SolventAwareParameters parameters = probe_parameters(3.0);
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);

    const double width = parameters.solvent_radius * parameters.radial_scale;
    const double spread = parameters.radial_spread;
    std::vector<double> environ_probe(basis.nrxx, 0.0);
    double integral = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const double distance = cubic_displacement(basis, length, ir).norm();
        if (distance <= width + 5.0 * spread)
        {
            environ_probe[ir] = std::erfc((distance - width) / spread);
        }
        integral += environ_probe[ir];
    }
    const double volume = length * length * length;
    integral *= volume / basis.nxyz;
    std::vector<std::complex<double>> environ_g(basis.npw);
    basis.real2recip(environ_probe.data(), environ_g.data());
    double maximum_difference = 0.0;
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        const std::complex<double> expected = environ_g[ig] * (volume / integral);
        maximum_difference = std::max(maximum_difference, std::abs(kernel[ig] - expected));
    }
    EXPECT_LT(maximum_difference, 1.0e-13);
    ASSERT_GE(basis.ig_gge0, 0);
    EXPECT_NEAR(kernel[basis.ig_gge0], 1.0, 1.0e-12);
    EXPECT_NEAR(first_shell_kernel(basis, kernel),
                isolated_probe_transform(ModuleBase::TWO_PI / length, width, spread), 1.0e-8);

    const std::vector<double> constant(basis.nrxx, 0.37);
    const std::vector<double> convolved = ModuleSccs::convolve_probe(kernel, basis, constant);
    double constant_error = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        constant_error = std::max(constant_error, std::abs(convolved[ir] - 0.37));
    }
    // Rounding of the grid sums over about 6e5 points.
    EXPECT_LT(constant_error, 1.0e-12);
}

// A cell narrower than the probe: the periodic images keep the kernel equal to
// the transform of the whole isolated probe, which the minimum-image probe
// misses.
TEST(SccsSolventAware, NarrowCellProbeSumsPeriodicImages)
{
    const double length = 10.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 120.0);
    const ModuleSccs::SolventAwareParameters parameters = probe_parameters(3.0);
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const double width = parameters.solvent_radius * parameters.radial_scale;
    const double spread = parameters.radial_spread;
    const double expected = isolated_probe_transform(ModuleBase::TWO_PI / length, width, spread);
    EXPECT_NEAR(first_shell_kernel(basis, kernel), expected, 1.0e-8);
    ASSERT_GE(basis.ig_gge0, 0);
    EXPECT_NEAR(kernel[basis.ig_gge0], 1.0, 1.0e-12);

    std::vector<double> minimum_image(basis.nrxx, 0.0);
    double integral = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const double distance = cubic_displacement(basis, length, ir).norm();
        if (distance <= width + 5.0 * spread)
        {
            minimum_image[ir] = std::erfc((distance - width) / spread);
        }
        integral += minimum_image[ir];
    }
    const double volume = length * length * length;
    integral *= volume / basis.nxyz;
    std::vector<std::complex<double>> minimum_image_g(basis.npw);
    basis.real2recip(minimum_image.data(), minimum_image_g.data());
    std::vector<double> truncated(basis.npw);
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        truncated[ig] = minimum_image_g[ig].real() * volume / integral;
    }
    // Measured 6.7e-3.
    EXPECT_GT(std::abs(first_shell_kernel(basis, truncated) - expected), 1.0e-3);
}

TEST(SccsSolventAware, KeepsUniformSoluteAndSolvent)
{
    const double length = 16.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 20.0);
    const ModuleSccs::SolventAwareParameters parameters = probe_parameters(1.5);
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const std::vector<double> solvent(basis.nrxx, 0.0);
    const std::vector<double> solute(basis.nrxx, 1.0);
    const ModuleSccs::SolventAwareBoundary empty
        = ModuleSccs::solvent_aware_boundary(solvent, kernel, parameters, basis);
    const ModuleSccs::SolventAwareBoundary full
        = ModuleSccs::solvent_aware_boundary(solute, kernel, parameters, basis);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_EQ(empty.boundary[ir], 0.0);
        EXPECT_EQ(empty.dfilling[ir], 0.0);
        EXPECT_NEAR(full.boundary[ir], 1.0, 1.0e-15);
    }
}

// A thick shell around a small void: the void cannot hold the probe and is
// filled, while the open solvent outside the shell stays solvent.
TEST(SccsSolventAware, FillsAnEnclosedVoidAndKeepsTheOpenSolvent)
{
    const double length = 30.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 20.0);
    const ModuleSccs::SolventAwareParameters parameters = probe_parameters(3.0);
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    const std::vector<double> outer = sphere(basis, length, origin, 8.0, 0.5);
    const std::vector<double> inner = sphere(basis, length, origin, 1.5, 0.5);
    std::vector<double> shell(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        shell[ir] = outer[ir] - inner[ir];
    }
    const ModuleSccs::SolventAwareBoundary filled
        = ModuleSccs::solvent_aware_boundary(shell, kernel, parameters, basis);
    double center_local = -1.0;
    double center_filled = -1.0;
    double outside_filled = 0.0;
    double filled_volume = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        EXPECT_GE(filled.boundary[ir], shell[ir] - 1.0e-15);
        EXPECT_LE(filled.boundary[ir], 1.0 + 1.0e-15);
        const double distance = cubic_displacement(basis, length, ir).norm();
        if (distance < 1.0e-12)
        {
            center_local = shell[ir];
            center_filled = filled.boundary[ir];
        }
        if (distance > 10.5)
        {
            outside_filled = std::max(outside_filled, filled.boundary[ir]);
        }
        filled_volume += (filled.boundary[ir] - shell[ir]) * length * length * length / basis.nxyz;
    }
    EXPECT_LT(center_local, 1.0e-4);
    EXPECT_GT(center_filled, 0.999);
    EXPECT_LT(outside_filled, 1.0e-6);
    // The void holds about 4 pi 1.5^3 / 3 = 14 bohr^3.
    EXPECT_GT(filled_volume, 5.0);
    EXPECT_LT(filled_volume, 40.0);
}

// E(s) = int g s_sa(s): the adjoint applied to g is dE/ds, including the
// nonlocal probe term that the pointwise (1 - f) g misses.
TEST(SccsSolventAware, AdjointIsTheDerivativeOfTheFilledBoundary)
{
    const double length = 16.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 20.0);
    ModuleSccs::SolventAwareParameters parameters = probe_parameters(1.5);
    parameters.filling_threshold = 0.45;
    parameters.filling_spread = 0.1;
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    const std::vector<double> local = sphere(basis, length, origin, 3.5, 0.6);
    const std::vector<double> direction
        = sphere(basis, length, ModuleBase::Vector3<double>(2.5, 1.0, 0.0), 2.0, 0.8);
    std::vector<double> weight(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const ModuleBase::Vector3<double> r = cubic_displacement(basis, length, ir);
        weight[ir] = 1.0 + 0.5 * std::cos(ModuleBase::TWO_PI * r.x / length)
                     + 0.3 * std::sin(ModuleBase::TWO_PI * r.y / length);
    }
    const ModuleSccs::SolventAwareBoundary filled
        = ModuleSccs::solvent_aware_boundary(local, kernel, parameters, basis);
    const std::vector<double> adjoint
        = ModuleSccs::solvent_aware_adjoint(local, filled, kernel, basis, weight);
    std::vector<double> pointwise(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        pointwise[ir] = (1.0 - filled.filling[ir]) * weight[ir];
    }

    const double step = 1.0e-5;
    std::vector<double> plus(basis.nrxx);
    std::vector<double> minus(basis.nrxx);
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        plus[ir] = local[ir] + step * direction[ir];
        minus[ir] = local[ir] - step * direction[ir];
    }
    const double energy_plus
        = grid_sum(weight, ModuleSccs::solvent_aware_boundary(plus, kernel, parameters, basis).boundary);
    const double energy_minus
        = grid_sum(weight, ModuleSccs::solvent_aware_boundary(minus, kernel, parameters, basis).boundary);
    const double finite_difference = (energy_plus - energy_minus) / (2.0 * step);
    const double analytic = grid_sum(adjoint, direction);
    const double local_only = grid_sum(pointwise, direction);
    std::cout << std::setprecision(12) << "SCCS_SA_ADJOINT finite_difference " << finite_difference << " analytic "
              << analytic << " pointwise " << local_only << std::endl;
    // Measured 9e-10 relative; the pointwise term alone misses by 27 percent.
    EXPECT_NEAR(analytic, finite_difference, 1.0e-8 * std::abs(finite_difference));
    EXPECT_GT(std::abs(local_only - finite_difference), 1.0e-2 * std::abs(finite_difference));
}

// For a well-resolved boundary the Environ chain formulas agree with spectral
// derivatives of s_sa up to the grid error of the composition. The filling
// switches over filling_spread / |grad c|, which must span several grid
// points; with Environ's filling_spread 0.02 it usually does not.
TEST(SccsSolventAware, ChainDerivativesMatchSpectralDerivativesOfTheFilledBoundary)
{
    const double length = 16.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 80.0);
    const double tpiba = ModuleBase::TWO_PI / length;
    ModuleSccs::SolventAwareParameters parameters = probe_parameters(1.5);
    parameters.filling_threshold = 0.45;
    parameters.filling_spread = 0.3;
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const std::vector<double> local
        = sphere(basis, length, ModuleBase::Vector3<double>(0.0, 0.0, 0.0), 3.5, 1.5);
    const ModuleSccs::SolventAwareBoundary filled
        = ModuleSccs::solvent_aware_boundary(local, kernel, parameters, basis);
    std::vector<ModuleBase::Vector3<double>> gradient
        = ModuleSccs::periodic_gradient(local, basis, tpiba);
    std::vector<double> laplacian = spectral_laplacian(local, basis, tpiba);
    const std::vector<ModuleBase::Vector3<double>> fraction_gradient
        = ModuleSccs::convolve_probe_gradient(kernel, basis, gradient);
    const std::vector<ModuleBase::Vector3<double>> short_fraction(basis.nrxx - 1);
    EXPECT_THROW(ModuleSccs::solvent_aware_chain_derivatives(local, filled, short_fraction, kernel,
                                                             basis, gradient, laplacian),
                 std::invalid_argument);
    ModuleSccs::solvent_aware_chain_derivatives(local, filled, fraction_gradient, kernel, basis,
                                                gradient, laplacian);

    const std::vector<ModuleBase::Vector3<double>> expected_gradient
        = ModuleSccs::periodic_gradient(filled.boundary, basis, tpiba);
    const std::vector<double> expected_laplacian = spectral_laplacian(filled.boundary, basis, tpiba);
    double gradient_error = 0.0;
    double gradient_scale = 0.0;
    double laplacian_error = 0.0;
    double laplacian_scale = 0.0;
    double filling_change = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        gradient_error = std::max(gradient_error, (gradient[ir] - expected_gradient[ir]).norm());
        gradient_scale = std::max(gradient_scale, expected_gradient[ir].norm());
        laplacian_error = std::max(laplacian_error, std::abs(laplacian[ir] - expected_laplacian[ir]));
        laplacian_scale = std::max(laplacian_scale, std::abs(expected_laplacian[ir]));
        filling_change = std::max(filling_change, filled.boundary[ir] - local[ir]);
    }
    std::cout << "SCCS_SA_CHAIN gradient " << gradient_error << " / " << gradient_scale
              << " laplacian " << laplacian_error << " / " << laplacian_scale << " filling "
              << filling_change << std::endl;
    EXPECT_GT(filling_change, 0.05);
    // Measured 8e-6 and 1e-4 of the maxima.
    EXPECT_LT(gradient_error, 5.0e-5 * gradient_scale);
    EXPECT_LT(laplacian_error, 5.0e-4 * laplacian_scale);
}

// The chain Hessian of s_sa gives the surface derivative -div(g/|g|_r) of a
// well-resolved filled boundary; compare with spectral derivatives of s_sa.
TEST(SccsSolventAware, ChainSurfaceMatchesSpectralSurfaceOfTheFilledBoundary)
{
    const double length = 16.0;
    ModulePW::PW_Basis basis("cpu", "double");
    setup_cubic_basis(basis, length, 80.0);
    const double tpiba = ModuleBase::TWO_PI / length;
    const double volume_element = length * length * length / static_cast<double>(basis.nxyz);
    ModuleSccs::SolventAwareParameters parameters = probe_parameters(1.5);
    parameters.filling_threshold = 0.45;
    parameters.filling_spread = 0.3;
    const double regularization = 1.0e-8;
    const ModuleSurchem::SerialChargeReduction reduction;
    const std::vector<double> kernel
        = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, length, parameters, reduction);
    const std::vector<double> local
        = sphere(basis, length, ModuleBase::Vector3<double>(0.0, 0.0, 0.0), 3.5, 1.5);
    const ModuleSccs::SolventAwareBoundary filled
        = ModuleSccs::solvent_aware_boundary(local, kernel, parameters, basis);
    const std::vector<ModuleBase::Vector3<double>> local_gradient
        = ModuleSccs::periodic_gradient(local, basis, tpiba);
    const std::vector<ModuleBase::Vector3<double>> fraction_gradient
        = ModuleSccs::convolve_probe_gradient(kernel, basis, local_gradient);
    const std::vector<std::vector<double>> local_hessian = spectral_hessian(local, basis, tpiba);
    const ModuleSccs::SolventAwareSurface surface
        = ModuleSccs::solvent_aware_surface(local, filled, kernel, basis, local_gradient,
                                            fraction_gradient, local_hessian, regularization);

    const std::vector<ModuleBase::Vector3<double>> expected_gradient
        = ModuleSccs::periodic_gradient(filled.boundary, basis, tpiba);
    const std::vector<std::vector<double>> expected_hessian
        = spectral_hessian(filled.boundary, basis, tpiba);
    double chain_surface = 0.0;
    double spectral_surface = 0.0;
    double gradient_error = 0.0;
    double gradient_scale = 0.0;
    double derivative_error = 0.0;
    double derivative_scale = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const ModuleBase::Vector3<double>& g = expected_gradient[ir];
        double projected = 0.0;
        double trace = 0.0;
        for (int component = 0; component < ModuleSccs::hessian_component_count; ++component)
        {
            int first = 0;
            int second = 0;
            ModuleSccs::hessian_axes(component, first, second);
            const double weight = first == second ? 1.0 : 2.0;
            projected += weight * g[first] * g[second] * expected_hessian[component][ir];
            if (first == second)
            {
                trace += expected_hessian[component][ir];
            }
        }
        const double norm_square = g.norm2() + regularization * regularization;
        const double norm = std::sqrt(norm_square);
        chain_surface += surface.gradient[ir].norm() * volume_element;
        spectral_surface += g.norm() * volume_element;
        gradient_error = std::max(gradient_error, (surface.gradient[ir] - g).norm());
        gradient_scale = std::max(gradient_scale, g.norm());
        // Weight by |g| as the surface energy does; the curvature itself is
        // ill-conditioned where the boundary is flat.
        const double expected_derivative = (projected - norm_square * trace) / (norm_square * norm);
        derivative_error = std::max(derivative_error,
                                    norm * std::abs(surface.surface_derivative[ir] - expected_derivative));
        derivative_scale = std::max(derivative_scale, norm * std::abs(expected_derivative));
    }
    std::cout << std::setprecision(12) << "SCCS_SA_SURFACE chain " << chain_surface << " spectral "
              << spectral_surface << " gradient " << gradient_error << " / " << gradient_scale
              << " derivative " << derivative_error << " / " << derivative_scale << std::endl;
    EXPECT_GT(spectral_surface, 1.0);
    EXPECT_NEAR(chain_surface, spectral_surface, 1.0e-5 * spectral_surface);
    EXPECT_LT(gradient_error, 5.0e-5 * gradient_scale);
    EXPECT_LT(derivative_error, 5.0e-4 * derivative_scale);
    int first = 0;
    int second = 0;
    EXPECT_THROW(ModuleSccs::hessian_axes(ModuleSccs::hessian_component_count, first, second),
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
    // ModuleBase::timer entries running.
    ModuleBase::timer::disable();
    const int result = RUN_ALL_TESTS();
#ifdef __MPI
    Parallel_Global::finalize_mpi();
#endif
    return result;
}
