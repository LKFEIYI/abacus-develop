#include "sccs_solvent_aware.h"

#include "../common/charge_reduction.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis.h"

#include <cmath>
#include <complex>
#include <stdexcept>

namespace ModuleSccs
{
namespace
{

// Environ tools_math erfc threshold: dsfunct2 and d2sfunct2 vanish beyond it.
const double filling_argument_cutoff = 6.0;
// Environ function_erfc cuts the probe at width + 5 spread.
const double probe_cutoff_spreads = 5.0;

ModuleBase::Vector3<double> lattice_row(const ModuleBase::Matrix3& lattice,
                                        const int row,
                                        const double scale)
{
    if (row == 0)
    {
        return ModuleBase::Vector3<double>(lattice.e11, lattice.e12, lattice.e13) * scale;
    }
    if (row == 1)
    {
        return ModuleBase::Vector3<double>(lattice.e21, lattice.e22, lattice.e23) * scale;
    }
    return ModuleBase::Vector3<double>(lattice.e31, lattice.e32, lattice.e33) * scale;
}

// Fractional grid coordinate folded into [-1/2, 1/2).
double folded_fraction(const int index, const int count)
{
    double fraction = static_cast<double>(index) / static_cast<double>(count);
    if (fraction >= 0.5)
    {
        fraction -= 1.0;
    }
    return fraction;
}

std::vector<ModuleBase::Vector3<double>> convolve_field(
    const std::vector<double>& kernel,
    const ModulePW::PW_Basis& basis,
    const std::vector<ModuleBase::Vector3<double>>& field)
{
    const std::size_t size = field.size();
    std::vector<double> component(size);
    std::vector<ModuleBase::Vector3<double>> result(size);
    for (int d = 0; d < 3; ++d)
    {
        for (std::size_t i = 0; i < size; ++i)
        {
            component[i] = field[i][d];
        }
        const std::vector<double> convolved = convolve_probe(kernel, basis, component);
        for (std::size_t i = 0; i < size; ++i)
        {
            result[i][d] = convolved[i];
        }
    }
    return result;
}

} // namespace

bool uses_solvent_aware(const SolventAwareParameters& parameters)
{
    return parameters.solvent_radius > 0.0;
}

void validate_solvent_aware_parameters(const SolventAwareParameters& parameters)
{
    if (!std::isfinite(parameters.solvent_radius) || parameters.solvent_radius < 0.0)
    {
        throw std::invalid_argument("SCCS solvent radius must be finite and non-negative");
    }
    if (!std::isfinite(parameters.radial_scale) || parameters.radial_scale < 1.0)
    {
        throw std::invalid_argument("SCCS solvent-aware radial scale must be at least one");
    }
    if (!std::isfinite(parameters.radial_spread) || parameters.radial_spread <= 0.0)
    {
        throw std::invalid_argument("SCCS solvent-aware radial spread must be positive");
    }
    if (!std::isfinite(parameters.filling_threshold) || parameters.filling_threshold <= 0.0
        || parameters.filling_threshold >= 1.0)
    {
        throw std::invalid_argument("SCCS filling threshold must lie between zero and one");
    }
    if (!std::isfinite(parameters.filling_spread) || parameters.filling_spread <= 0.0)
    {
        throw std::invalid_argument("SCCS filling spread must be positive");
    }
}

std::vector<double> solvent_probe_kernel(const ModulePW::PW_Basis& basis,
                                         const ModuleBase::Matrix3& lattice_vectors,
                                         const double lattice_scale,
                                         const SolventAwareParameters& parameters,
                                         const ModuleSurchem::ChargeReduction& reduction)
{
    ModuleBase::timer::start("ModuleSccs", "solvent_probe_kernel");
    validate_solvent_aware_parameters(parameters);
    if (!uses_solvent_aware(parameters))
    {
        throw std::invalid_argument("SCCS solvent probe requires a positive solvent radius");
    }
    if (basis.nrxx != basis.nx * basis.ny * basis.nplane || !std::isfinite(lattice_scale)
        || lattice_scale <= 0.0)
    {
        throw std::invalid_argument("SCCS solvent probe requires an initialized grid and lattice");
    }
    const ModuleBase::Vector3<double> a1 = lattice_row(lattice_vectors, 0, lattice_scale);
    const ModuleBase::Vector3<double> a2 = lattice_row(lattice_vectors, 1, lattice_scale);
    const ModuleBase::Vector3<double> a3 = lattice_row(lattice_vectors, 2, lattice_scale);
    const double volume = std::abs(a1 * (a2 ^ a3));
    const double width = parameters.solvent_radius * parameters.radial_scale;
    const double spread = parameters.radial_spread;
    const double cutoff = width + probe_cutoff_spreads * spread;
    // Images n with |r0 + n a| <= cutoff for r0 in the folded cell; the cell
    // height along a_k is the volume over the area of the other two vectors.
    const int image_count_1 = static_cast<int>(std::ceil(cutoff * (a2 ^ a3).norm() / volume + 0.5));
    const int image_count_2 = static_cast<int>(std::ceil(cutoff * (a3 ^ a1).norm() / volume + 0.5));
    const int image_count_3 = static_cast<int>(std::ceil(cutoff * (a1 ^ a2).norm() / volume + 0.5));

    std::vector<double> probe(basis.nrxx, 0.0);
    double integral = 0.0;
    for (int ir = 0; ir < basis.nrxx; ++ir)
    {
        const int ix = ir / (basis.ny * basis.nplane);
        const int iy = ir / basis.nplane - ix * basis.ny;
        const int iz = ir % basis.nplane + basis.startz_current;
        const ModuleBase::Vector3<double> folded = a1 * folded_fraction(ix, basis.nx)
                                                   + a2 * folded_fraction(iy, basis.ny)
                                                   + a3 * folded_fraction(iz, basis.nz);
        double value = 0.0;
        for (int n1 = -image_count_1; n1 <= image_count_1; ++n1)
        {
            for (int n2 = -image_count_2; n2 <= image_count_2; ++n2)
            {
                for (int n3 = -image_count_3; n3 <= image_count_3; ++n3)
                {
                    const ModuleBase::Vector3<double> image
                        = folded + a1 * static_cast<double>(n1) + a2 * static_cast<double>(n2)
                          + a3 * static_cast<double>(n3);
                    const double distance = image.norm();
                    if (distance <= cutoff)
                    {
                        value += std::erfc((distance - width) / spread);
                    }
                }
            }
        }
        probe[ir] = value;
        integral += value;
    }
    reduction.reduce_sum(integral);
    integral *= volume / static_cast<double>(basis.nxyz);
    if (!std::isfinite(integral) || integral <= 0.0)
    {
        throw std::runtime_error("SCCS solvent probe has no weight on the grid");
    }
    // Environ convolution: fwfft(f1) fwfft(f2) omega with the 1/N forward
    // transform, so the kernel is omega times the transform of the normalized
    // probe. The probe is even on the grid, so its transform is real.
    std::vector<std::complex<double>> probe_g(basis.npw);
    basis.real2recip(probe.data(), probe_g.data());
    std::vector<double> kernel(basis.npw);
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        kernel[ig] = probe_g[ig].real() * volume / integral;
    }
    ModuleBase::timer::end("ModuleSccs", "solvent_probe_kernel");
    return kernel;
}

std::vector<double> convolve_probe(const std::vector<double>& kernel,
                                   const ModulePW::PW_Basis& basis,
                                   const std::vector<double>& values)
{
    if (kernel.size() != static_cast<std::size_t>(basis.npw)
        || values.size() != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument("SCCS probe convolution arrays must match the PW basis");
    }
    std::vector<std::complex<double>> values_g(basis.npw);
    basis.real2recip(values.data(), values_g.data());
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        values_g[ig] *= kernel[ig];
    }
    std::vector<double> result(values.size());
    basis.recip2real(values_g.data(), result.data());
    return result;
}

SolventAwareBoundary solvent_aware_boundary(const std::vector<double>& local,
                                            const std::vector<double>& kernel,
                                            const SolventAwareParameters& parameters,
                                            const ModulePW::PW_Basis& basis)
{
    const std::vector<double> fraction = convolve_probe(kernel, basis, local);
    const std::size_t size = local.size();
    const double threshold = parameters.filling_threshold;
    const double spread = parameters.filling_spread;
    const double sqrt_pi = std::sqrt(ModuleBase::PI);
    SolventAwareBoundary filled;
    filled.filling.resize(size);
    filled.dfilling.resize(size);
    filled.d2filling.resize(size);
    filled.boundary.resize(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        // Environ: fill = 1 - sfunct2, dfill = -dsfunct2, d2fill = -d2sfunct2.
        const double argument = (fraction[i] - threshold) / spread;
        filled.filling[i] = 1.0 - 0.5 * std::erfc(argument);
        if (std::abs(argument) <= filling_argument_cutoff)
        {
            const double gaussian = std::exp(-argument * argument) / sqrt_pi;
            filled.dfilling[i] = gaussian / spread;
            filled.d2filling[i] = -2.0 * argument * gaussian / (spread * spread);
        }
        filled.boundary[i] = local[i] + (1.0 - local[i]) * filled.filling[i];
    }
    return filled;
}

void solvent_aware_chain_derivatives(const std::vector<double>& local,
                                     const SolventAwareBoundary& filled,
                                     const std::vector<double>& kernel,
                                     const ModulePW::PW_Basis& basis,
                                     std::vector<ModuleBase::Vector3<double>>& gradient,
                                     std::vector<double>& laplacian)
{
    const std::size_t size = local.size();
    if (gradient.size() != size || laplacian.size() != size || filled.boundary.size() != size)
    {
        throw std::invalid_argument("SCCS solvent-aware derivative arrays must match the grid");
    }
    const std::vector<ModuleBase::Vector3<double>> fraction_gradient
        = convolve_field(kernel, basis, gradient);
    const std::vector<double> fraction_laplacian = convolve_probe(kernel, basis, laplacian);
    for (std::size_t i = 0; i < size; ++i)
    {
        const double empty = 1.0 - filled.filling[i];
        const double solvent = 1.0 - local[i];
        const double projection = gradient[i] * fraction_gradient[i];
        laplacian[i] = laplacian[i] * empty - 2.0 * projection * filled.dfilling[i]
                       + solvent
                             * (filled.d2filling[i] * fraction_gradient[i].norm2()
                                + filled.dfilling[i] * fraction_laplacian[i]);
        gradient[i] = gradient[i] * empty
                      + fraction_gradient[i] * (solvent * filled.dfilling[i]);
    }
}

std::vector<double> solvent_aware_adjoint(const std::vector<double>& local,
                                          const SolventAwareBoundary& filled,
                                          const std::vector<double>& kernel,
                                          const ModulePW::PW_Basis& basis,
                                          const std::vector<double>& boundary_potential)
{
    const std::size_t size = local.size();
    if (boundary_potential.size() != size || filled.boundary.size() != size)
    {
        throw std::invalid_argument("SCCS solvent-aware adjoint arrays must match the grid");
    }
    std::vector<double> weighted(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        weighted[i] = (1.0 - local[i]) * filled.dfilling[i] * boundary_potential[i];
    }
    std::vector<double> result = convolve_probe(kernel, basis, weighted);
    for (std::size_t i = 0; i < size; ++i)
    {
        result[i] += (1.0 - filled.filling[i]) * boundary_potential[i];
    }
    return result;
}

} // namespace ModuleSccs
