#include "sccs_sqrt_cg.h"

#include "sccs_poisson.h"
#include "../common/charge_reduction.h"
#include "../common/thread_sum.h"

#include "source_basis/module_pw/pw_basis.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{
namespace
{

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

} // namespace

SqrtPreconditioner::SqrtPreconditioner(const std::vector<double>& invsqrt,
                                       const CoulombOperator& coulomb)
    : invsqrt_(invsqrt), coulomb_(coulomb)
{
    this->weighted_.resize(invsqrt.size());
}

void SqrtPreconditioner::apply(const std::vector<double>& rhs, std::vector<double>& value)
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

bool residual_converged(const std::vector<double>& residual,
                        const PolarizationSolverParameters& solver,
                        const ModuleSurchem::ChargeReduction& reduction,
                        PolarizationResult& polarization)
{
    reduced_rms_max(residual, reduction, polarization.residual_rms, polarization.residual_max);
    return polarization.residual_rms <= solver.tolerance_rms
           && polarization.residual_max <= solver.tolerance_max;
}

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

} // namespace ModuleSccs
