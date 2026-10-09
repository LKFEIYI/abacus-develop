#ifndef SCCS_SOLVENT_AWARE_TEST_H
#define SCCS_SOLVENT_AWARE_TEST_H

#include "sccs_test.h"
#include "../sccs_cavity.h"
#include "../sccs_solvent_aware.h"

#include "source_base/parallel_reduce.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <algorithm>
#include <complex>
#include <vector>

// Grid helpers and the cubic-cell fixture of the solvent-aware tests.
namespace SccsTest
{
const ModuleBase::Matrix3 cubic_lattice(1.0, 0.0, 0.0,
                                        0.0, 1.0, 0.0,
                                        0.0, 0.0, 1.0);
const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);

inline void setup_basis(ModulePW::PW_Basis& basis, double scale, const ModuleBase::Matrix3& lattice, double ecut)
{
#ifdef __MPI
    basis.initmpi(pool_size, pool_rank, POOL_WORLD);
#endif
    basis.initgrids(scale, lattice, ecut);
    basis.initparameters(false, ecut, 1, false);
    basis.setuptransform();
    basis.collect_local_pw();
}

// Fold a displacement into the cube [-length/2, length/2).
inline void fold_into_cube(double length, ModuleBase::Vector3<double>& r)
{
    for (int d = 0; d < 3; ++d)
    {
        const double fraction = r[d] / length;
        r[d] -= length * std::round(fraction);
    }
}

// Minimum-image displacement of grid point ir from the origin in a cube.
inline ModuleBase::Vector3<double> cubic_displacement(const ModulePW::PW_Basis& basis, double length, int ir)
{
    const int ix = ir / (basis.ny * basis.nplane);
    const int iy = ir / basis.nplane - ix * basis.ny;
    const int iz = ir % basis.nplane + basis.startz_current;
    const double x = length * ix / basis.nx;
    const double y = length * iy / basis.ny;
    const double z = length * iz / basis.nz;
    ModuleBase::Vector3<double> r(x, y, z);
    fold_into_cube(length, r);
    return r;
}

inline std::vector<ModuleBase::Vector3<double>> spectral_gradient(const std::vector<double>& values,
                                                                  const ModulePW::PW_Basis& basis,
                                                                  double tpiba)
{
    std::vector<std::complex<double>> values_g(basis.npw);
    basis.real2recip(values.data(), values_g.data());
    std::vector<ModuleBase::Vector3<double>> gradient(values.size());
    XC_Functional::grad_rho(values_g.data(), gradient.data(), &basis, tpiba);
    return gradient;
}

inline std::vector<double> spectral_laplacian(const std::vector<double>& values,
                                              const ModulePW::PW_Basis& basis,
                                              double tpiba)
{
    std::vector<std::complex<double>> values_g(basis.npw);
    basis.real2recip(values.data(), values_g.data());
    std::vector<double> laplacian(values.size());
    XC_Functional::laplacian_rho(values_g.data(), laplacian.data(), &basis, tpiba);
    return laplacian;
}

inline double pool_dot(const std::vector<double>& left, const std::vector<double>& right)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) { sum += left[i] * right[i]; }
    Parallel_Reduce::reduce_pool(sum);
    return sum;
}

inline double pool_max(const ModulePW::PW_Basis& basis, double value)
{
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, value);
    return value;
}

// values + step * direction, for central differences.
inline std::vector<double> shifted(const std::vector<double>& values,
                                   const std::vector<double>& direction,
                                   double step)
{
    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { result[i] = values[i] + step * direction[i]; }
    return result;
}

inline ModuleSccs::SolventAwareParameters probe_parameters(double solvent_radius)
{
    ModuleSccs::SolventAwareParameters parameters;
    parameters.solvent_radius = solvent_radius;
    return parameters;
}

// Discrete surface S = sum (|g|_r - r) of the filled chain gradient of the
// cavity density and its exact derivative dS/dn.
struct DensitySurface
{
    double surface = 0.0;
    ModuleSccs::SolventAwareNonelectrostatic chain;
};

// A cubic cell and the probe kernel of the solvent-aware parameters.
class SolventAwareTest : public testing::Test
{
protected:
    SolventAwareTest() : basis("cpu", "double") {}

    void set_up_cube(double edge, double ecut, const ModuleSccs::SolventAwareParameters& probe)
    {
        length = edge;
        tpiba = ModuleBase::TWO_PI / edge;
        parameters = probe;
        setup_basis(basis, edge, cubic_lattice, ecut);
        kernel = ModuleSccs::solvent_probe_kernel(basis, cubic_lattice, edge, probe);
    }

    ModuleBase::Vector3<double> displacement(int ir) const { return cubic_displacement(basis, length, ir); }

    // Smooth sphere erfc((|r - center| - radius)/spread)/2.
    std::vector<double> sphere(const ModuleBase::Vector3<double>& center, double radius, double spread) const
    {
        std::vector<double> values(basis.nrxx);
        for (int ir = 0; ir < basis.nrxx; ++ir)
        {
            ModuleBase::Vector3<double> r = displacement(ir) - center;
            fold_into_cube(length, r);
            const double argument = (r.norm() - radius) / spread;
            values[ir] = 0.5 * std::erfc(argument);
        }
        return values;
    }

    ModuleSccs::SolventAwareBoundary fill(const std::vector<double>& local) const
    {
        return ModuleSccs::solvent_aware_boundary(local, kernel, parameters, basis);
    }

    // Two Gaussians of density along x, centred at +-offset; their midpoint is a density saddle.
    std::vector<double> gaussian_pair(double offset) const
    {
        std::vector<double> density(basis.nrxx);
        const ModuleBase::Vector3<double> first(offset, 0.0, 0.0);
        const ModuleBase::Vector3<double> second(-offset, 0.0, 0.0);
        for (int ir = 0; ir < basis.nrxx; ++ir)
        {
            const ModuleBase::Vector3<double> r = displacement(ir);
            const double exponent_first = -(r - first).norm2();
            const double exponent_second = -(r - second).norm2();
            const double first_gaussian = std::exp(exponent_first);
            const double second_gaussian = std::exp(exponent_second);
            density[ir] = 4.8e-3 * (first_gaussian + second_gaussian);
        }
        return density;
    }

    DensitySurface surface_of_density(const std::vector<double>& density,
                                      const ModuleSccs::CavityParameters& cavity,
                                      double regularization) const
    {
        const std::size_t size = density.size();
        ModuleSccs::FilledCavity filled;
        filled.local.resize(size);
        filled.d2solute_drho2.resize(size);
        std::vector<double> dsolute(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            const ModuleSccs::CavityPoint point = ModuleSccs::evaluate_cavity(density[i], cavity);
            filled.local[i] = point.solute;
            dsolute[i] = point.dsolute_drho;
            filled.d2solute_drho2[i] = point.d2solute_drho2;
        }
        filled.filling = fill(filled.local);
        filled.density_gradient = spectral_gradient(density, basis, tpiba);
        std::vector<ModuleBase::Vector3<double>> local_gradient(size);
        for (std::size_t i = 0; i < size; ++i) { local_gradient[i] = filled.density_gradient[i] * dsolute[i]; }
        filled.fraction_gradient = ModuleSccs::convolve_probe_gradient(kernel, basis, local_gradient);
        // Unit surface tension and no pressure: density_potential is dS/dn.
        const double surface_tension = 1.0;
        const double pressure = 0.0;
        DensitySurface result;
        result.chain = ModuleSccs::solvent_aware_nonelectrostatic(filled, dsolute, kernel, basis, tpiba, regularization,
                                                                  surface_tension, pressure);
        for (std::size_t i = 0; i < size; ++i)
        {
            const double norm_square = result.chain.gradient[i].norm2() + regularization * regularization;
            result.surface += std::sqrt(norm_square) - regularization;
        }
        Parallel_Reduce::reduce_pool(result.surface);
        return result;
    }

    // Central difference of the discrete surface along direction against the
    // projected analytic derivative; returns the relative error.
    double surface_derivative_error(const std::vector<double>& density,
                                    const std::vector<double>& direction,
                                    const ModuleSccs::CavityParameters& cavity,
                                    double regularization,
                                    double step) const
    {
        const DensitySurface center = surface_of_density(density, cavity, regularization);
        const double backward = -step;
        const std::vector<double> plus = shifted(density, direction, step);
        const std::vector<double> minus = shifted(density, direction, backward);
        const double surface_plus = surface_of_density(plus, cavity, regularization).surface;
        const double surface_minus = surface_of_density(minus, cavity, regularization).surface;
        const double finite_difference = (surface_plus - surface_minus) / (2.0 * step);
        const double analytic = pool_dot(center.chain.density_potential, direction);
        EXPECT_GT(std::abs(finite_difference), 1.0e-3);
        const double miss = std::abs(analytic - finite_difference);
        return miss / std::abs(finite_difference);
    }

    double length = 0.0;
    double tpiba = 0.0;
    ModulePW::PW_Basis basis;
    ModuleSccs::SolventAwareParameters parameters;
    std::vector<double> kernel;
};
} // namespace SccsTest

#endif
