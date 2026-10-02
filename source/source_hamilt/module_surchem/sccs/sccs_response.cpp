#include "sccs_response.h"

#include "sccs_pw_coulomb.h"
#include "../common/charge_reduction.h"
#include "../common/thread_sum.h"

#include "source_base/constants.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace ModuleSccs
{

std::vector<double> continuum_polarization_charge(
    const std::vector<double>& solute_charge,
    const SccsResponse& response)
{
    const std::size_t size = solute_charge.size();
    if (response.epsilon.size() != size || response.grad_log_epsilon.size() != size
        || response.polarization.field.gradient.size() != size)
    {
        throw std::invalid_argument("SCCS polarization source arrays must have the same size");
    }
    std::vector<double> polarization(size);
    for (std::size_t index = 0; index < size; ++index)
    {
        const double epsilon = response.epsilon[index];
        if (!std::isfinite(epsilon) || epsilon < 1.0)
        {
            throw std::domain_error("SCCS polarization source requires finite epsilon >= 1");
        }
        double projection = 0.0;
        for (int direction = 0; direction < 3; ++direction)
        {
            projection += response.grad_log_epsilon[index][direction]
                          * response.polarization.field.gradient[index][direction];
        }
        polarization[index] = projection / ModuleBase::FOUR_PI
                              + solute_charge[index] * (1.0 / epsilon - 1.0);
        if (!std::isfinite(polarization[index]))
        {
            throw std::domain_error("SCCS polarization source must be finite");
        }
    }
    return polarization;
}

namespace
{

// Environ boundary_of_density: the boundary s(n) of the cavity density and
// ds/dn. The dielectric follows from s in dielectric_of_boundary.
SccsResponse prepare_cavity(const std::vector<double>& density, const CavityParameters& cavity)
{
    SccsResponse result;
    const std::size_t size = density.size();
    result.solute.resize(size);
    result.dsolute_drho.resize(size);
    // evaluate_cavity throws on a non-finite density; count those here so that
    // nothing is thrown inside the parallel loop.
    long invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024) reduction(+ : invalid)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(density[i]))
        {
            ++invalid;
            continue;
        }
        const CavityPoint point = ModuleSccs::evaluate_cavity(density[i], cavity);
        result.solute[i] = point.solute;
        result.dsolute_drho[i] = point.dsolute_drho;
    }
    if (invalid > 0)
    {
        throw std::domain_error("SCCS cavity density must be finite");
    }
    return result;
}

// Environ solvent_aware_boundary: keep s(n) as local_solute and replace the
// dielectric boundary by its filling s_sa.
void fill_cavity(const SolventAwareParameters& solvent_aware,
                 const std::vector<double>& probe_kernel,
                 const ModulePW::PW_Basis& basis,
                 const ModuleSurchem::ChargeReduction& reduction,
                 SccsResponse& result)
{
    result.local_solute = result.solute;
    result.filling = solvent_aware_boundary(result.local_solute, probe_kernel, solvent_aware, basis);
    result.solute = result.filling.boundary;
    const std::vector<double>& filled = result.solute;
    const std::vector<double>& local = result.local_solute;
    const auto add_filling = [&filled, &local](const std::size_t i, std::array<double, 1>& sum) {
        sum[0] += filled[i] - local[i];
    };
    double filled_volume = ModuleSurchem::thread_sums<1>(filled.size(), add_filling)[0];
    reduction.reduce_sum(filled_volume);
    result.filled_volume = filled_volume * basis.omega / basis.nxyz;
}

// Pool RMS and maximum absolute value of a distributed grid array.
void reduced_rms_max(const std::vector<double>& values,
                     const ModuleSurchem::ChargeReduction& reduction,
                     double& rms,
                     double& maximum)
{
    // Per thread: the sum of squares and the maximum magnitude.
    const auto add_value = [&values](const std::size_t i, std::array<double, 2>& partial) {
        partial[0] += values[i] * values[i];
        const double magnitude = std::abs(values[i]);
        partial[1] = std::max(partial[1], magnitude);
    };
    const std::vector<std::array<double, 2>> partials
        = ModuleSurchem::thread_partials<2>(values.size(), add_value);
    double square = 0.0;
    double local_maximum = 0.0;
    for (std::size_t thread = 0; thread < partials.size(); ++thread)
    {
        square += partials[thread][0];
        local_maximum = std::max(local_maximum, partials[thread][1]);
    }
    double count = static_cast<double>(values.size());
    reduction.reduce_sum(square);
    reduction.reduce_max(local_maximum);
    reduction.reduce_sum(count);
    if (!std::isfinite(square) || !std::isfinite(local_maximum) || count <= 0.0)
    {
        throw std::runtime_error("SCCS sqrt-CG residual is not finite");
    }
    const double mean_square = square / count;
    rms = std::sqrt(mean_square);
    maximum = local_maximum;
}

// Update the residual norms of polarization; true when both pass
// sccs_tol_rms and sccs_tol_max.
bool residual_converged(const std::vector<double>& residual,
                        const PolarizationSolverParameters& solver,
                        const ModuleSurchem::ChargeReduction& reduction,
                        PolarizationResult& polarization)
{
    reduced_rms_max(residual, reduction, polarization.residual_rms, polarization.residual_max);
    return polarization.residual_rms <= solver.tolerance_rms
           && polarization.residual_max <= solver.tolerance_max;
}

// Grid inner product of two distributed arrays.
double grid_dot(const std::vector<double>& left,
                const std::vector<double>& right,
                const ModulePW::PW_Basis& basis,
                const ModuleSurchem::ChargeReduction& reduction)
{
    const auto add_product = [&left, &right](const std::size_t i, std::array<double, 1>& sum) {
        sum[0] += left[i] * right[i];
    };
    double value = ModuleSurchem::thread_sums<1>(left.size(), add_product)[0];
    reduction.reduce_sum(value);
    return value * basis.omega / basis.nxyz;
}

// P r = eps^-1/2 G eps^-1/2 r with the periodic or PCC-corrected Coulomb
// operator G. The cavity is fixed during a solve, so the scratch arrays are
// reused by every application; only the scalar potential is needed.
class SqrtPreconditioner
{
  public:
    SqrtPreconditioner(const std::vector<double>& invsqrt, const CoulombOperator& coulomb)
        : invsqrt_(invsqrt), coulomb_(coulomb)
    {
        this->weighted_.resize(invsqrt.size());
    }

    void apply(const std::vector<double>& rhs, std::vector<double>& value)
    {
        const std::size_t size = this->invsqrt_.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            this->weighted_[i] = rhs[i] * this->invsqrt_[i];
        }
        this->coulomb_.apply_potential(this->weighted_, this->potential_);
        value.resize(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            value[i] = this->potential_[i] * this->invsqrt_[i];
        }
    }

  private:
    const std::vector<double>& invsqrt_;
    const CoulombOperator& coulomb_;
    std::vector<double> weighted_;
    std::vector<double> potential_;
};

// Spectral Laplacian of the real field with coefficients values_g, each
// multiplied by filter[ig] unless filter is empty. It is symmetric under the
// grid inner product, so it is its own transpose in the cavity derivative.
void spectral_laplacian(const std::vector<std::complex<double>>& values_g,
                        const std::vector<double>& filter,
                        const ModulePW::PW_Basis& basis,
                        const double tpiba,
                        std::vector<double>& laplacian)
{
    std::vector<std::complex<double>> laplacian_g(basis.npw);
    if (filter.empty())
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            laplacian_g[ig] = values_g[ig] * (-tpiba * tpiba * basis.gg[ig]);
        }
    }
    else
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            laplacian_g[ig] = values_g[ig] * (-tpiba * tpiba * basis.gg[ig] * filter[ig]);
        }
    }
    laplacian.resize(basis.nrxx);
    basis.recip2real(laplacian_g.data(), laplacian.data());
}

// Environ boundary_of_density with deriv_method 'chain': grad s = s' grad n
// from the spectral grad n of the density coefficients density_g;
// density_gradient returns grad n.
void chain_boundary_gradient(const std::vector<std::complex<double>>& density_g,
                             const std::vector<double>& dsolute_drho,
                             const ModulePW::PW_Basis& basis,
                             const double tpiba,
                             std::vector<ModuleBase::Vector3<double>>& density_gradient,
                             std::vector<ModuleBase::Vector3<double>>& gradient)
{
    const std::vector<double> no_filter;
    density_gradient = ModuleSccs::spectral_gradient(density_g, no_filter, basis, tpiba);
    const std::size_t size = dsolute_drho.size();
    gradient.resize(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        for (int d = 0; d < 3; ++d)
        {
            gradient[i][d] = dsolute_drho[i] * density_gradient[i][d];
        }
    }
}

// The chain lapl s = s' lapl n + s'' |grad n|^2 matching
// chain_boundary_gradient.
void chain_boundary_laplacian(const std::vector<double>& density,
                              const std::vector<std::complex<double>>& density_g,
                              const std::vector<ModuleBase::Vector3<double>>& density_gradient,
                              const CavityParameters& cavity,
                              const ModulePW::PW_Basis& basis,
                              const double tpiba,
                              std::vector<double>& laplacian)
{
    const std::vector<double> no_filter;
    std::vector<double> density_laplacian;
    spectral_laplacian(density_g, no_filter, basis, tpiba, density_laplacian);
    const std::size_t size = density.size();
    laplacian.resize(size);
    long invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024) reduction(+ : invalid)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(density[i]))
        {
            ++invalid;
            continue;
        }
        const CavityPoint point = ModuleSccs::evaluate_cavity(density[i], cavity);
        double gradient_square = 0.0;
        for (int d = 0; d < 3; ++d)
        {
            gradient_square += density_gradient[i][d] * density_gradient[i][d];
        }
        laplacian[i] = point.dsolute_drho * density_laplacian[i]
                       + point.d2solute_drho2 * gradient_square;
    }
    if (invalid > 0)
    {
        throw std::domain_error("SCCS cavity density must be finite");
    }
}

// Environ dielectric_of_boundary for an electronic boundary, in Ha units:
// eps = exp(L (1 - s)) with L = ln(eps_bulk), grad ln eps = -L grad s, and the
// sqrt-CG coefficient eps (lapl ln eps / 2 + |grad ln eps|^2 / 4) / (4 pi).
void dielectric_of_boundary(const CavityParameters& cavity,
                            const std::vector<ModuleBase::Vector3<double>>& gradient,
                            const std::vector<double>& laplacian,
                            SccsResponse& result,
                            std::vector<double>& coefficient)
{
    const std::size_t size = result.solute.size();
    const double log_bulk = std::log(cavity.epsilon_bulk);
    result.epsilon.resize(size);
    result.grad_log_epsilon.resize(size);
    coefficient.resize(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        const double log_epsilon = log_bulk * (1.0 - result.solute[i]);
        result.epsilon[i] = std::exp(log_epsilon);
        double gradient_square = 0.0;
        for (int d = 0; d < 3; ++d)
        {
            result.grad_log_epsilon[i][d] = -log_bulk * gradient[i][d];
            gradient_square += gradient[i][d] * gradient[i][d];
        }
        const double lap_log = -log_bulk * laplacian[i];
        coefficient[i] = result.epsilon[i]
                         * (0.5 * lap_log + 0.25 * log_bulk * log_bulk * gradient_square)
                         / ModuleBase::FOUR_PI;
    }
}

// Environ 3.1.1 core_fft_lowpass: with lowpass_p1 and lowpass_p2 positive,
// every switching-function derivative is multiplied by
// 0.5 erfc(p1 G^2/Gcut^2 - p2), Gcut^2 being the density cutoff. One weight
// per G vector; empty without the lowpass.
std::vector<double> switching_filter(const CavityParameters& cavity,
                                     const ModulePW::PW_Basis& basis)
{
    std::vector<double> filter;
    if (!uses_switching_lowpass(cavity))
    {
        return filter;
    }
    filter.resize(basis.npw);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        const double argument = cavity.lowpass_p1 * basis.gg[ig] / basis.ggecut - cavity.lowpass_p2;
        filter[ig] = 0.5 * std::erfc(argument);
    }
    return filter;
}

// With PCC the potential at the cavity edge carries the open-boundary monopole
// and dipole. The chain-rule f drops steeply to zero at density_min, and the
// sampled f v source then fails to converge with the grid. Differentiate the
// boundary s on the FFT grid instead (Environ deriv_method 'fft'); the filtered
// grad s is also the one of the exact cavity derivative. One forward transform
// serves the gradient and the Laplacian.
void switching_boundary_derivatives(const std::vector<double>& boundary,
                                    const std::vector<double>& filter,
                                    const ModulePW::PW_Basis& basis,
                                    const double tpiba,
                                    std::vector<ModuleBase::Vector3<double>>& gradient,
                                    std::vector<double>& laplacian)
{
    std::vector<std::complex<double>> boundary_g(basis.npw);
    basis.real2recip(boundary.data(), boundary_g.data());
    gradient = ModuleSccs::spectral_gradient(boundary_g, filter, basis, tpiba);
    spectral_laplacian(boundary_g, filter, basis, tpiba, laplacian);
}

// PCC dielectric derivatives of the filled boundary without the lowpass: the
// FFT derivatives of the local boundary s, as for the PCC boundary without
// filling, and the analytic filling of solvent_aware_chain_derivatives with
// grad c = p * grad s. Unfiltered FFT derivatives of s_sa itself ring where the
// filling switches within a grid spacing and stall the SCF. With the lowpass,
// s_sa is differentiated on the filtered grid instead (Environ deriv_method
// 'fft' with deriv_lowpass), which keeps the exact cavity derivative of
// switching_boundary_potential; the analytic filling would put pointwise
// higher derivatives of the filling, which the filter does not smooth, into
// that exact derivative and stall the SCF where the filling only starts.
void filled_fft_derivatives(const SccsResponse& result,
                            const std::vector<double>& filter,
                            const std::vector<double>& probe_kernel,
                            const ModulePW::PW_Basis& basis,
                            const double tpiba,
                            std::vector<ModuleBase::Vector3<double>>& gradient,
                            std::vector<double>& laplacian)
{
    switching_boundary_derivatives(result.local_solute, filter, basis, tpiba, gradient, laplacian);
    const std::vector<ModuleBase::Vector3<double>> fraction_gradient
        = convolve_probe_gradient(probe_kernel, basis, gradient);
    solvent_aware_chain_derivatives(result.local_solute, result.filling, fraction_gradient,
                                    probe_kernel, basis, gradient, laplacian);
}

// Periodic dielectric derivatives (Environ deriv_method 'chain'): the chain
// grad s and lapl s of the density, carried through the filling when it is
// on. The filled response keeps grad n and grad c for its chain surface.
void periodic_dielectric_derivatives(const std::vector<double>& density,
                                     const std::vector<std::complex<double>>& density_g,
                                     const CavityParameters& cavity,
                                     const std::vector<double>& probe_kernel,
                                     const ModulePW::PW_Basis& basis,
                                     const double tpiba,
                                     SccsResponse& result,
                                     std::vector<ModuleBase::Vector3<double>>& gradient,
                                     std::vector<double>& laplacian)
{
    std::vector<ModuleBase::Vector3<double>> density_gradient;
    chain_boundary_gradient(density_g, result.dsolute_drho, basis, tpiba, density_gradient, gradient);
    chain_boundary_laplacian(density, density_g, density_gradient, cavity, basis, tpiba, laplacian);
    if (result.local_solute.empty())
    {
        return;
    }
    result.fraction_gradient = convolve_probe_gradient(probe_kernel, basis, gradient);
    solvent_aware_chain_derivatives(result.local_solute, result.filling, result.fraction_gradient,
                                    probe_kernel, basis, gradient, laplacian);
    result.density_gradient.swap(density_gradient);
}

// PCC dielectric derivatives on the FFT grid (Environ deriv_method 'fft'): of
// s, or of s_sa with the lowpass, or of the local s with the analytic filling
// without it (an empty filter). The filled surface still takes the chain
// gradients of the periodic path.
void open_dielectric_derivatives(const std::vector<std::complex<double>>& density_g,
                                 const std::vector<double>& filter,
                                 const std::vector<double>& probe_kernel,
                                 const ModulePW::PW_Basis& basis,
                                 const double tpiba,
                                 SccsResponse& result,
                                 std::vector<ModuleBase::Vector3<double>>& gradient,
                                 std::vector<double>& laplacian)
{
    const bool filled = !result.local_solute.empty();
    if (filled && filter.empty())
    {
        filled_fft_derivatives(result, filter, probe_kernel, basis, tpiba, gradient, laplacian);
    }
    else
    {
        switching_boundary_derivatives(result.solute, filter, basis, tpiba, gradient, laplacian);
    }
    if (!filled)
    {
        return;
    }
    std::vector<ModuleBase::Vector3<double>> local_gradient;
    chain_boundary_gradient(density_g, result.dsolute_drho, basis, tpiba, result.density_gradient,
                            local_gradient);
    result.fraction_gradient = convolve_probe_gradient(probe_kernel, basis, local_gradient);
}

// Continuum boundary potential -(deps/ds)|grad v|^2/(8 pi) of Environ
// dielectric::de_dboundary, with deps/ds = -L eps and grad v from the solved
// potential.
void continuum_boundary_potential(const CavityParameters& cavity, SccsResponse& result)
{
    const std::size_t size = result.epsilon.size();
    const double log_bulk = std::log(cavity.epsilon_bulk);
    result.boundary_potential.resize(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        const ModuleBase::Vector3<double>& gradient = result.polarization.field.gradient[i];
        const double gradient_square
            = gradient.x * gradient.x + gradient.y * gradient.y + gradient.z * gradient.z;
        result.boundary_potential[i]
            = log_bulk * result.epsilon[i] * gradient_square / (8.0 * ModuleBase::PI);
    }
}

// Exact derivative of the discrete reaction energy through the cavity for the
// lowpass switching-function factsqrt. The sqrt-CG solves A v = q with
// A = sqrt(eps) G^-1 sqrt(eps) + F, which is symmetric, so E = q^T A^-1 q / 2
// changes by dE = -v^T dA v / 2 and needs no adjoint solve. With
// L = ln(eps_bulk), s the switching function and b = eps v^2/(8 pi), the
// sqrt(eps) term and the pointwise part of dF add up to q v / 2, and the
// transposes of the filtered lapl and grad in F give
// dE/ds = L/2 (q v + lapl b + L div(b grad s)).
// Its continuum limit is L eps |grad v|^2/(8 pi). Without the filter the
// discrete derivative is not grid-converged at the cavity edge and its
// grid-scale oscillations drive the electronic SCF to diverge. Both filtered
// transposes are summed in G space, so one inverse transform returns them;
// minus the filtered divergence is the transpose of the filtered gradient.
void switching_boundary_potential(const std::vector<double>& charge,
                                const std::vector<double>& potential,
                                const std::vector<ModuleBase::Vector3<double>>& solute_gradient,
                                const CavityParameters& cavity,
                                const std::vector<double>& filter,
                                const ModulePW::PW_Basis& basis,
                                const double tpiba,
                                SccsResponse& result)
{
    if (filter.size() != static_cast<std::size_t>(basis.npw))
    {
        throw std::invalid_argument("SCCS switching boundary potential requires the lowpass filter");
    }
    const std::size_t size = charge.size();
    const double log_bulk = std::log(cavity.epsilon_bulk);
    std::vector<double> weight(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        weight[i] = result.epsilon[i] * potential[i] * potential[i] / (8.0 * ModuleBase::PI);
    }
    // lapl b + L div(b grad s), filtered, accumulated in G space.
    std::vector<std::complex<double>> transpose_g(basis.npw);
    basis.real2recip(weight.data(), transpose_g.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (int ig = 0; ig < basis.npw; ++ig)
    {
        transpose_g[ig] *= -tpiba * tpiba * basis.gg[ig] * filter[ig];
    }
    std::vector<double> component(size);
    std::vector<std::complex<double>> component_g(basis.npw);
    for (int d = 0; d < 3; ++d)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            component[i] = weight[i] * solute_gradient[i][d];
        }
        basis.real2recip(component.data(), component_g.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            transpose_g[ig] += log_bulk * ModuleBase::IMAG_UNIT * tpiba * basis.gcar[ig][d]
                               * component_g[ig] * filter[ig];
        }
    }
    std::vector<double> transpose(size);
    basis.recip2real(transpose_g.data(), transpose.data());
    result.boundary_potential.resize(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        result.boundary_potential[i]
            = 0.5 * log_bulk * (charge[i] * potential[i] + transpose[i]);
    }
}

// ENVIRON generalized_sqrt warm start: one preconditioned fixed-point step
// v = P(q - f v_old) from the previous potential, whose charge residual is
// f (v_old - v) with f the sqrt-CG coefficient. Keep it in potential and
// residual only when it improves on the cold-start residual of polarization;
// true when kept.
bool warm_start(const std::vector<double>& charge,
                const std::vector<double>& coefficient,
                const std::vector<double>& initial_potential,
                const ModuleSurchem::ChargeReduction& reduction,
                SqrtPreconditioner& preconditioner,
                std::vector<double>& potential,
                std::vector<double>& residual,
                PolarizationResult& polarization)
{
    const std::size_t size = charge.size();
    std::vector<double> guess_residual(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        guess_residual[i] = charge[i] - coefficient[i] * initial_potential[i];
    }
    std::vector<double> guess;
    preconditioner.apply(guess_residual, guess);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        guess_residual[i] = coefficient[i] * (initial_potential[i] - guess[i]);
    }
    double guess_rms = 0.0;
    double guess_max = 0.0;
    reduced_rms_max(guess_residual, reduction, guess_rms, guess_max);
    if (guess_rms >= polarization.residual_rms)
    {
        return false;
    }
    potential.swap(guess);
    residual.swap(guess_residual);
    polarization.warm_started = true;
    return true;
}

// Preconditioned CG on K v = q with K = P^-1 + f, from potential and its
// charge residual. P^-1 z = r for z = P r, so K d follows from the recurrence
// without P^-1. Returns whether both residual tolerances were reached.
bool sqrt_cg(const std::vector<double>& coefficient,
             const PolarizationSolverParameters& solver,
             const ModulePW::PW_Basis& basis,
             const ModuleSurchem::ChargeReduction& reduction,
             SqrtPreconditioner& preconditioner,
             std::vector<double>& potential,
             std::vector<double>& residual,
             PolarizationResult& polarization)
{
    const std::size_t size = potential.size();
    std::vector<double> direction(size, 0.0);
    std::vector<double> image(size, 0.0);
    std::vector<double> z;
    double old_rz = 0.0;
    bool converged = false;
    for (int iteration = 1; !converged && iteration <= solver.max_iterations; ++iteration)
    {
        preconditioner.apply(residual, z);
        const double rz = grid_dot(residual, z, basis, reduction);
        if (!std::isfinite(rz) || std::abs(rz) < 1e-30)
        {
            throw std::runtime_error("CG sqrt null/nonfinite preconditioned residual");
        }
        const double beta = std::abs(old_rz) > 1e-30 ? rz / old_rz : 0.0;
        old_rz = rz;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            direction[i] = z[i] + beta * direction[i];
            image[i] = coefficient[i] * z[i] + residual[i] + beta * image[i];
        }
        const double curvature = grid_dot(direction, image, basis, reduction);
        if (!std::isfinite(curvature) || curvature == 0.0)
        {
            throw std::runtime_error("CG sqrt invalid curvature");
        }
        const double alpha = rz / curvature;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            potential[i] += alpha * direction[i];
            residual[i] -= alpha * image[i];
        }
        polarization.iterations = iteration;
        converged = residual_converged(residual, solver, reduction, polarization);
    }
    return converged;
}

// Check the preconditioned equation v = P(q - K v) independently of the CG
// recurrences; this costs one extra Poisson solve.
void check_fixed_point(const std::vector<double>& charge,
                       const std::vector<double>& coefficient,
                       const std::vector<double>& potential,
                       const ModuleSurchem::ChargeReduction& reduction,
                       SqrtPreconditioner& preconditioner,
                       PolarizationResult& polarization)
{
    const std::size_t size = potential.size();
    std::vector<double> right(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        right[i] = charge[i] - coefficient[i] * potential[i];
    }
    std::vector<double> image;
    preconditioner.apply(right, image);
    std::vector<double> defect(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        defect[i] = potential[i] - image[i];
    }
    reduced_rms_max(defect, reduction,
                    polarization.fixed_point_defect_rms,
                    polarization.fixed_point_defect_max);
    polarization.fixed_point_checked = true;
}

// Shift a periodic potential to zero cell mean (ENVIRON generalized_sqrt).
void remove_mean(const ModulePW::PW_Basis& basis,
                 const ModuleSurchem::ChargeReduction& reduction,
                 std::vector<double>& potential)
{
    const std::size_t size = potential.size();
    const auto add_potential = [&potential](const std::size_t i, std::array<double, 1>& sum) {
        sum[0] += potential[i];
    };
    double mean = ModuleSurchem::thread_sums<1>(size, add_potential)[0];
    reduction.reduce_sum(mean);
    mean /= basis.nxyz;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        potential[i] -= mean;
    }
}

// The CG builds sqrt(eps) v = w = C_PCC(s) with s = (q - f v)/sqrt(eps).
// Report the far-field polarization charge int(s)/sqrt(eps_bulk) - int(q).
void finish_open_boundary_response(const std::vector<double>& charge,
                                   const std::vector<double>& coefficient,
                                   const std::vector<double>& invsqrt,
                                   const CavityParameters& cavity,
                                   const ModulePW::PW_Basis& basis,
                                   const ModuleSurchem::ChargeReduction& reduction,
                                   SccsResponse& result)
{
    const std::size_t size = charge.size();
    const std::vector<double>& potential = result.polarization.field.potential;
    const auto add_source = [&](const std::size_t i, std::array<double, 2>& sums) {
        sums[0] += (charge[i] - coefficient[i] * potential[i]) * invsqrt[i];
        sums[1] += charge[i];
    };
    const std::array<double, 2> sums = ModuleSurchem::thread_sums<2>(size, add_source);
    double source_sum = sums[0];
    double solute_sum = sums[1];
    reduction.reduce_sum(source_sum);
    reduction.reduce_sum(solute_sum);
    const double volume_element = basis.omega / basis.nxyz;
    const double bulk_invsqrt = 1.0 / std::sqrt(cavity.epsilon_bulk);
    result.far_field_polarization_charge
        = (source_sum * bulk_invsqrt - solute_sum) * volume_element;
}

} // namespace

std::vector<double> boundary_to_density_potential(const SccsResponse& response,
                                                  const std::vector<double>& probe_kernel,
                                                  const ModulePW::PW_Basis& basis,
                                                  const std::vector<double>& boundary_potential)
{
    const std::size_t size = response.dsolute_drho.size();
    if (boundary_potential.size() != size)
    {
        throw std::invalid_argument("SCCS boundary potential must match the cavity grid");
    }
    std::vector<double> density_potential = boundary_potential;
    if (!response.local_solute.empty())
    {
        density_potential = solvent_aware_adjoint(response.local_solute, response.filling,
                                                  probe_kernel, basis, boundary_potential);
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        density_potential[i] *= response.dsolute_drho[i];
    }
    return density_potential;
}

SolventAwareSurface solvent_aware_surface_of_density(const std::vector<double>& density,
                                                     const CavityParameters& cavity,
                                                     const SccsResponse& response,
                                                     const std::vector<double>& probe_kernel,
                                                     const ModulePW::PW_Basis& basis,
                                                     const double tpiba,
                                                     const double regularization)
{
    ModuleBase::timer::start("ModuleSccs", "solvent_aware_surface_of_density");
    const std::size_t size = density.size();
    if (response.local_solute.size() != size || response.density_gradient.size() != size
        || response.fraction_gradient.size() != size
        || response.density_reciprocal.size() != static_cast<std::size_t>(basis.npw))
    {
        throw std::invalid_argument(
            "SCCS solvent-aware surface requires the filled boundary and its chain gradients");
    }
    const std::vector<ModuleBase::Vector3<double>>& density_gradient = response.density_gradient;
    const std::vector<std::complex<double>>& density_g = response.density_reciprocal;
    std::vector<double> dsolute(size);
    std::vector<double> d2solute(size);
    long invalid = 0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024) reduction(+ : invalid)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        if (!std::isfinite(density[i]))
        {
            ++invalid;
            continue;
        }
        const CavityPoint point = ModuleSccs::evaluate_cavity(density[i], cavity);
        dsolute[i] = point.dsolute_drho;
        d2solute[i] = point.d2solute_drho2;
    }
    if (invalid > 0)
    {
        throw std::domain_error("SCCS cavity density must be finite");
    }
    std::vector<ModuleBase::Vector3<double>> local_gradient(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        local_gradient[i] = density_gradient[i] * dsolute[i];
    }
    std::vector<std::vector<double>> local_hessian(hessian_component_count);
    std::vector<std::complex<double>> hessian_g(basis.npw);
    const double tpiba_square = tpiba * tpiba;
    for (int component = 0; component < hessian_component_count; ++component)
    {
        int first = 0;
        int second = 0;
        hessian_axes(component, first, second);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            hessian_g[ig] = -tpiba_square * basis.gcar[ig][first] * basis.gcar[ig][second]
                            * density_g[ig];
        }
        std::vector<double>& hessian = local_hessian[component];
        hessian.resize(size);
        basis.recip2real(hessian_g.data(), hessian.data());
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            const double outer = density_gradient[i][first] * density_gradient[i][second];
            hessian[i] = dsolute[i] * hessian[i] + d2solute[i] * outer;
        }
    }
    SolventAwareSurface surface = solvent_aware_surface(response.local_solute,
                                                        response.filling,
                                                        probe_kernel,
                                                        basis,
                                                        local_gradient,
                                                        response.fraction_gradient,
                                                        local_hessian,
                                                        regularization);
    ModuleBase::timer::end("ModuleSccs", "solvent_aware_surface_of_density");
    return surface;
}

SccsResponse solve_sccs_response(
    const std::vector<double>& density,
    const std::vector<double>& charge,
    const ModuleSccs::CavityParameters& cavity,
    const ModuleSccs::SolventAwareParameters& solvent_aware,
    const std::vector<double>& probe_kernel,
    const ModuleSccs::PolarizationSolverParameters& solver,
    const std::vector<double>& initial_potential,
    const ModulePW::PW_Basis& basis,
    const double tpiba,
    const ModuleSccs::CoulombOperator& coulomb,
    const ModuleSurchem::ChargeReduction& reduction)
{
    ModuleBase::timer::start("ModuleSccs", "solve_sccs_response");
    ModuleSccs::validate_cavity_parameters(cavity);
    ModuleSccs::validate_solvent_aware_parameters(solvent_aware);
    SccsResponse result = prepare_cavity(density, cavity);
    const std::size_t size = density.size();
    const bool filled = uses_solvent_aware(solvent_aware);
    if (filled)
    {
        fill_cavity(solvent_aware, probe_kernel, basis, reduction, result);
    }
    const bool open_boundary = coulomb.has_boundary_correction();
    const std::vector<double> filter = switching_filter(cavity, basis);
    // One forward transform of the cavity density serves every chain
    // derivative, including the Hessian of the filled surface.
    std::vector<std::complex<double>> density_g;
    if (!open_boundary || filled)
    {
        density_g.resize(basis.npw);
        basis.real2recip(density.data(), density_g.data());
    }
    std::vector<ModuleBase::Vector3<double>> solute_gradient;
    std::vector<double> solute_laplacian;
    if (open_boundary)
    {
        open_dielectric_derivatives(density_g, filter, probe_kernel, basis, tpiba, result,
                                    solute_gradient, solute_laplacian);
    }
    else
    {
        periodic_dielectric_derivatives(density, density_g, cavity, probe_kernel, basis, tpiba,
                                        result, solute_gradient, solute_laplacian);
    }
    if (filled)
    {
        result.density_reciprocal.swap(density_g);
    }
    std::vector<double> coefficient;
    dielectric_of_boundary(cavity, solute_gradient, solute_laplacian, result, coefficient);
    std::vector<double> invsqrt(size);
#ifdef _OPENMP
#pragma omp parallel for schedule(static, 1024)
#endif
    for (std::size_t i = 0; i < size; ++i)
    {
        invsqrt[i] = 1.0 / std::sqrt(result.epsilon[i]);
    }
    SqrtPreconditioner preconditioner(invsqrt, coulomb);
    std::vector<double> residual = charge;
    std::vector<double> potential(size, 0.0);
    PolarizationResult& polarization = result.polarization;
    bool converged = residual_converged(residual, solver, reduction, polarization);
    if (!converged && initial_potential.size() == size)
    {
        const bool restarted = warm_start(charge, coefficient, initial_potential, reduction,
                                          preconditioner, potential, residual, polarization);
        converged = restarted && residual_converged(residual, solver, reduction, polarization);
    }
    if (!converged)
    {
        converged = sqrt_cg(coefficient, solver, basis, reduction, preconditioner, potential,
                            residual, polarization);
    }
    if (!converged)
    {
        throw std::runtime_error(
            "SCCS sqrt-CG did not reach sccs_tol_rms and sccs_tol_max within sccs_maxiter");
    }
    if (solver.check_fixed_point)
    {
        check_fixed_point(charge, coefficient, potential, reduction, preconditioner, polarization);
    }
    result.restart_potential = potential;
    // A PCC operator in the preconditioner fixes the physical gauge; only the
    // periodic potential is shifted to zero mean.
    if (!open_boundary)
    {
        remove_mean(basis, reduction, potential);
    }
    result.polarization.field.potential = potential;
    // Environ dielectric::de_dboundary differentiates the solved potential on
    // its derivative grid for the continuum cavity potential. The lowpass
    // exact derivative needs no field, so there it is formed only for the
    // diagnostics.
    const bool continuum_cavity = !uses_switching_lowpass(cavity);
    if (continuum_cavity || solver.polarization_diagnostics)
    {
        result.polarization.field.gradient = ModuleSccs::periodic_gradient(potential, basis, tpiba);
    }
    if (open_boundary)
    {
        finish_open_boundary_response(charge, coefficient, invsqrt, cavity, basis, reduction,
                                      result);
    }
    if (open_boundary && solver.polarization_diagnostics)
    {
        // The corrected potential has no periodic Laplacian inverse; use the
        // ENVIRON dielectric_of_potential polarization charge instead. Its
        // integral carries a finite-grid error; the far field fixes the net
        // screening charge.
        result.polarization.polarization_charge = continuum_polarization_charge(charge, result);
    }
    if (uses_switching_lowpass(cavity))
    {
        switching_boundary_potential(charge, potential, solute_gradient, cavity, filter, basis,
                                     tpiba, result);
    }
    else
    {
        continuum_boundary_potential(cavity, result);
    }
    result.cavity_potential
        = boundary_to_density_potential(result, probe_kernel, basis, result.boundary_potential);
    ModuleBase::timer::end("ModuleSccs", "solve_sccs_response");
    return result;
}

} // namespace ModuleSccs
