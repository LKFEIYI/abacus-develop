#include "sccs_response.h"
#include "sccs_cavity_derivatives.h"
#include "sccs_lowpass.h"
#include "sccs_parameters.h"
#include "sccs_pw_coulomb.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_hamilt/module_xc/xc_functional.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <utility>

namespace ModuleSccs
{
namespace
{
// sqrt(eps) v = C_PCC(s), s = (q - f v)/sqrt(eps). The far-field
// screening charge is int(s)/sqrt(eps_bulk) - int(q), including its physical sign.
bool finish_open_boundary_response(const std::vector<double>& charge,
                                   const std::vector<double>& coefficient,
                                   const std::vector<double>& invsqrt,
                                   const CavityParameters& cavity,
                                   const ModulePW::PW_Basis& basis,
                                   SccsResponse& result,
                                   std::string& error)
{
    const std::size_t size = charge.size();
    const std::vector<double>& potential = result.polarization.potential;
    double source_sum = 0.0;
    double solute_sum = 0.0;
    for (std::size_t i = 0; i < size; ++i)
    {
        source_sum += (charge[i] - coefficient[i] * potential[i]) * invsqrt[i];
        solute_sum += charge[i];
    }
    Parallel_Reduce::reduce_pool(source_sum);
    Parallel_Reduce::reduce_pool(solute_sum);
    const double volume_element = basis.omega / basis.nxyz;
    const double bulk_invsqrt = 1.0 / std::sqrt(cavity.epsilon_bulk);
    result.far_field_polarization_charge = (source_sum * bulk_invsqrt - solute_sum) * volume_element;
    if (!std::isfinite(result.far_field_polarization_charge))
    {
        error = "SCCS open-boundary screening charge must be finite";
        return false;
    }
    return true;
}

bool residual_norms(const std::vector<double>& values,
                    const ModulePW::PW_Basis& basis,
                    double& rms,
                    double& maximum,
                    std::string& error)
{
    double square = 0.0;
    maximum = 0.0;
    double invalid = 0.0;
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            invalid = 1.0;
        }
        square += value * value;
        const double magnitude = std::abs(value);
        maximum = std::max(maximum, magnitude);
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    Parallel_Reduce::reduce_pool(square);
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, maximum);
    if (invalid != 0.0 || !std::isfinite(square))
    {
        error = "SCCS sqrt-CG residual is not finite";
        return false;
    }
    const double mean_square = square / basis.nxyz;
    rms = std::sqrt(mean_square);
    return true;
}

bool converged(const PolarizationResult& result, const PolarizationSolverParameters& solver)
{
    return result.residual_rms <= solver.tolerance_rms && result.residual_max <= solver.tolerance_max;
}

double grid_dot(const std::vector<double>& left,
                const std::vector<double>& right,
                const ModulePW::PW_Basis& basis)
{
    double value = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        value += left[i] * right[i];
    }
    Parallel_Reduce::reduce_pool(value);
    return value * basis.omega / basis.nxyz;
}

// P r = eps^-1/2 G eps^-1/2 r; only FFT scratch survives an application.
bool apply_preconditioner(const std::vector<double>& rhs,
                          const std::vector<double>& invsqrt,
                          CoulombOperator& coulomb,
                          std::vector<double>& weighted,
                          std::vector<double>& value,
                          std::string& error)
{
    const std::size_t size = rhs.size();
    weighted.resize(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        weighted[i] = rhs[i] * invsqrt[i];
    }
    if (!coulomb.apply_potential(weighted, value, error))
    {
        return false;
    }
    for (std::size_t i = 0; i < size; ++i)
    {
        value[i] *= invsqrt[i];
    }
    return true;
}
} // namespace

bool solve_sccs_response(const std::vector<double>& density,
                         const std::vector<double>& charge,
                         const CavityParameters& cavity,
                         const PolarizationSolverParameters& solver,
                         const std::vector<double>& initial_potential,
                         const ModulePW::PW_Basis& basis,
                         double tpiba,
                         SccsResponse& result,
                         std::string& error)
{
    PeriodicCoulombOperator coulomb(basis, tpiba);
    return solve_sccs_response(density, charge, cavity, solver, initial_potential, basis, tpiba,
                               coulomb, result, error);
}

bool solve_sccs_response(const std::vector<double>& density,
                         const std::vector<double>& charge,
                         const CavityParameters& cavity,
                         const PolarizationSolverParameters& solver,
                         const std::vector<double>& initial_potential,
                         const ModulePW::PW_Basis& basis,
                         double tpiba,
                         CoulombOperator& coulomb,
                         SccsResponse& result,
                         std::string& error)
{
    if (!validate_pw_grid(basis, tpiba, error) || !validate_grid_values(density, basis, error)
        || !validate_grid_values(charge, basis, error))
    {
        return false;
    }
    double invalid = 0.0;
    if (!validate_cavity_parameters(cavity, error) || solver.max_iterations <= 0
        || !std::isfinite(solver.tolerance_rms) || solver.tolerance_rms <= 0.0
        || !std::isfinite(solver.tolerance_max) || solver.tolerance_max <= 0.0)
    {
        invalid = 1.0;
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS requires valid cavity parameters, a positive iteration limit and finite positive tolerances";
        return false;
    }

    // Determine start mode collectively, including ranks with no real-space points.
    double initial_count = initial_potential.size();
    Parallel_Reduce::reduce_pool(initial_count);
    const bool warm_start = initial_count != 0.0;
    if (warm_start && !validate_grid_values(initial_potential, basis, error))
    {
        return false;
    }

    const bool open_boundary = coulomb.has_boundary_correction();
    SccsResponse candidate;
    CavityDerivatives derivatives;
    if (!prepare_cavity_derivatives(density, cavity, basis, tpiba, open_boundary, candidate, derivatives, error))
    {
        return false;
    }
    const std::vector<double>& coefficient = derivatives.coefficient;
    const std::size_t size = density.size();
    std::vector<double> invsqrt(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        invsqrt[i] = 1.0 / std::sqrt(candidate.epsilon[i]);
    }
    std::vector<double> residual = charge;
    std::vector<double> potential(size, 0.0);
    std::vector<double> direction(size, 0.0);
    std::vector<double> image(size, 0.0);
    std::vector<double> weighted;
    std::vector<double> z;
    PolarizationResult& polarization = candidate.polarization;
    if (!residual_norms(residual, basis, polarization.residual_rms, polarization.residual_max, error))
    {
        return false;
    }
    // Retain the old fixed-point warm start only when it reduces the charge residual.
    if (!converged(polarization, solver) && warm_start)
    {
        std::vector<double> guess_residual(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            guess_residual[i] = charge[i] - coefficient[i] * initial_potential[i];
        }
        if (!apply_preconditioner(guess_residual, invsqrt, coulomb, weighted, z, error))
        {
            return false;
        }
        for (std::size_t i = 0; i < size; ++i)
        {
            guess_residual[i] = coefficient[i] * (initial_potential[i] - z[i]);
        }
        double guess_rms = 0.0;
        double guess_max = 0.0;
        if (!residual_norms(guess_residual, basis, guess_rms, guess_max, error))
        {
            return false;
        }
        if (guess_rms < polarization.residual_rms)
        {
            potential.swap(z);
            residual.swap(guess_residual);
            polarization.warm_started = true;
            polarization.residual_rms = guess_rms;
            polarization.residual_max = guess_max;
        }
    }
    double old_rz = 0.0;
    for (int iteration = 1; !converged(polarization, solver) && iteration <= solver.max_iterations; ++iteration)
    {
        if (!apply_preconditioner(residual, invsqrt, coulomb, weighted, z, error))
        {
            return false;
        }
        const double rz = grid_dot(residual, z, basis);
        if (!std::isfinite(rz) || std::abs(rz) < 1e-30)
        {
            error = "SCCS sqrt-CG has a null or nonfinite preconditioned residual";
            return false;
        }
        const double beta = std::abs(old_rz) > 1e-30 ? rz / old_rz : 0.0;
        old_rz = rz;
        for (std::size_t i = 0; i < size; ++i)
        {
            direction[i] = z[i] + beta * direction[i];
            image[i] = coefficient[i] * z[i] + residual[i] + beta * image[i];
        }
        const double curvature = grid_dot(direction, image, basis);
        if (!std::isfinite(curvature) || curvature == 0.0)
        {
            error = "SCCS sqrt-CG has invalid curvature";
            return false;
        }
        const double alpha = rz / curvature;
        for (std::size_t i = 0; i < size; ++i)
        {
            potential[i] += alpha * direction[i];
            residual[i] -= alpha * image[i];
        }
        polarization.iterations = iteration;
        if (!residual_norms(residual, basis, polarization.residual_rms, polarization.residual_max, error))
        {
            return false;
        }
    }
    if (!converged(polarization, solver))
    {
        error = "SCCS sqrt-CG did not reach both residual tolerances within the iteration limit";
        return false;
    }

    if (!validate_grid_values(potential, basis, error))
    {
        return false;
    }
    candidate.restart_potential = potential;
    // Preserve the physical gauge fixed by a boundary correction.
    if (!open_boundary)
    {
        double mean = 0.0;
        for (double value : potential)
        {
            mean += value;
        }
        Parallel_Reduce::reduce_pool(mean);
        mean /= basis.nxyz;
        for (double& value : potential)
        {
            value -= mean;
        }
    }
    const bool lowpass = uses_switching_lowpass(cavity);
    if (!lowpass)
    {
        std::vector<std::complex<double>> potential_g(basis.npw);
        basis.real2recip(potential.data(), potential_g.data());
        polarization.gradient.resize(size);
        XC_Functional::grad_rho(potential_g.data(), polarization.gradient.data(), &basis, tpiba);
    }
    polarization.potential.swap(potential);
    if (open_boundary
        && !finish_open_boundary_response(charge, coefficient, invsqrt, cavity, basis, candidate, error))
    {
        return false;
    }
    if (lowpass)
    {
        if (!evaluate_lowpass_cavity_potential(charge, cavity, derivatives, basis, tpiba, candidate, error))
        {
            return false;
        }
    }
    else
    {
        candidate.cavity_potential.resize(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            const ModuleBase::Vector3<double>& gradient = polarization.gradient[i];
            const double gradient_square = gradient * gradient;
            candidate.cavity_potential[i] = -candidate.depsilon_drho[i] * gradient_square / (8.0 * ModuleBase::PI);
        }
    }
    if (!validate_grid_values(candidate.cavity_potential, basis, error))
    {
        return false;
    }
    result = std::move(candidate);
    return true;
}
} // namespace ModuleSccs
