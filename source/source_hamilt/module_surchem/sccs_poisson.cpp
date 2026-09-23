#include "sccs_poisson.h"

#include "source_base/constants.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{
namespace
{

bool finite_vector(const ModuleBase::Vector3<double>& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

double dot(const ModuleBase::Vector3<double>& left, const ModuleBase::Vector3<double>& right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

void validate_input(const std::vector<double>& solute_charge,
                    const std::vector<double>& epsilon,
                    const std::vector<ModuleBase::Vector3<double>>& grad_log_epsilon,
                    const std::vector<double>& initial_polarization_charge)
{
    const std::size_t size = solute_charge.size();
    if (size == 0)
    {
        throw std::invalid_argument("SCCS polarization solver requires a non-empty grid");
    }
    if (epsilon.size() != size || grad_log_epsilon.size() != size)
    {
        throw std::invalid_argument("SCCS polarization input arrays must have the same size");
    }
    if (!initial_polarization_charge.empty() && initial_polarization_charge.size() != size)
    {
        throw std::invalid_argument("SCCS initial polarization charge has the wrong size");
    }
    for (std::size_t index = 0; index < size; ++index)
    {
        if (!std::isfinite(solute_charge[index]) || !std::isfinite(epsilon[index])
            || epsilon[index] < 1.0 || !finite_vector(grad_log_epsilon[index]))
        {
            throw std::domain_error("SCCS polarization inputs must be finite and epsilon must be at least one");
        }
        if (!initial_polarization_charge.empty()
            && !std::isfinite(initial_polarization_charge[index]))
        {
            throw std::domain_error("SCCS initial polarization charge must be finite");
        }
    }
}

bool valid_field(const ElectrostaticField& field, const std::size_t size)
{
    if (field.potential.size() != size || field.gradient.size() != size)
    {
        throw std::runtime_error("SCCS Coulomb operator returned arrays with the wrong size");
    }
    for (std::size_t index = 0; index < size; ++index)
    {
        if (!std::isfinite(field.potential[index]) || !finite_vector(field.gradient[index]))
        {
            return false;
        }
    }
    return true;
}

void evaluate_field(const std::vector<double>& solute_charge,
                    const std::vector<double>& polarization_charge,
                    const CoulombOperator& coulomb,
                    ElectrostaticField& field)
{
    std::vector<double> total_charge(solute_charge.size(), 0.0);
    for (std::size_t index = 0; index < solute_charge.size(); ++index)
    {
        total_charge[index] = solute_charge[index] + polarization_charge[index];
    }
    coulomb.apply(total_charge, field);
}

} // namespace

void validate_polarization_solver_parameters(const PolarizationSolverParameters& parameters)
{
    if (parameters.max_iterations <= 0)
    {
        throw std::invalid_argument("SCCS polarization maximum iteration count must be positive");
    }
    if (!std::isfinite(parameters.mixing) || parameters.mixing <= 0.0
        || parameters.mixing > 1.0)
    {
        throw std::invalid_argument("SCCS polarization mixing must be in the interval (0, 1]");
    }
    if (!std::isfinite(parameters.tolerance_rms) || parameters.tolerance_rms <= 0.0
        || !std::isfinite(parameters.tolerance_max) || parameters.tolerance_max <= 0.0)
    {
        throw std::invalid_argument("SCCS polarization residual tolerances must be positive and finite");
    }
}

void SerialPolarizationReduction::reduce_residual(double& square_sum,
                                                  double& maximum,
                                                  double& point_count) const
{
    if (!std::isfinite(square_sum) || !std::isfinite(maximum) || !std::isfinite(point_count))
    {
        throw std::domain_error("SCCS residual values must be finite before reduction");
    }
}

PolarizationResult solve_polarization(
    const std::vector<double>& solute_charge,
    const std::vector<double>& epsilon,
    const std::vector<ModuleBase::Vector3<double>>& grad_log_epsilon,
    const std::vector<double>& initial_polarization_charge,
    const PolarizationSolverParameters& parameters,
    const CoulombOperator& coulomb)
{
    const SerialPolarizationReduction reduction;
    return solve_polarization(solute_charge,
                              epsilon,
                              grad_log_epsilon,
                              initial_polarization_charge,
                              parameters,
                              coulomb,
                              reduction);
}

PolarizationResult solve_polarization(
    const std::vector<double>& solute_charge,
    const std::vector<double>& epsilon,
    const std::vector<ModuleBase::Vector3<double>>& grad_log_epsilon,
    const std::vector<double>& initial_polarization_charge,
    const PolarizationSolverParameters& parameters,
    const CoulombOperator& coulomb,
    const PolarizationReduction& reduction)
{
    validate_polarization_solver_parameters(parameters);
    validate_input(solute_charge, epsilon, grad_log_epsilon, initial_polarization_charge);

    const std::size_t size = solute_charge.size();
    PolarizationResult result;
    result.polarization_charge.assign(size, 0.0);
    if (!initial_polarization_charge.empty())
    {
        result.polarization_charge = initial_polarization_charge;
    }

    std::vector<double> trial(size, 0.0);
    for (int iteration = 1; iteration <= parameters.max_iterations; ++iteration)
    {
        evaluate_field(solute_charge, result.polarization_charge, coulomb, result.field);
        result.iterations = iteration;
        if (!valid_field(result.field, size))
        {
            result.status = PolarizationStatus::NonFinite;
            return result;
        }

        double residual_square_sum = 0.0;
        result.residual_max = 0.0;
        for (std::size_t index = 0; index < size; ++index)
        {
            const double dielectric_source
                = dot(grad_log_epsilon[index], result.field.gradient[index])
                  / ModuleBase::FOUR_PI;
            const double screening_source
                = -(epsilon[index] - 1.0) * solute_charge[index] / epsilon[index];
            trial[index] = dielectric_source + screening_source;
            const double residual = trial[index] - result.polarization_charge[index];
            if (!std::isfinite(trial[index]) || !std::isfinite(residual))
            {
                result.status = PolarizationStatus::NonFinite;
                return result;
            }
            residual_square_sum += residual * residual;
            result.residual_max = std::max(result.residual_max, std::abs(residual));
        }
        double point_count = static_cast<double>(size);
        reduction.reduce_residual(residual_square_sum, result.residual_max, point_count);
        if (!std::isfinite(residual_square_sum) || !std::isfinite(result.residual_max)
            || !std::isfinite(point_count) || point_count <= 0.0)
        {
            result.status = PolarizationStatus::NonFinite;
            return result;
        }
        result.residual_rms = std::sqrt(residual_square_sum / point_count);

        if (result.residual_rms <= parameters.tolerance_rms
            && result.residual_max <= parameters.tolerance_max)
        {
            result.polarization_charge = trial;
            result.status = PolarizationStatus::Converged;
            evaluate_field(solute_charge, result.polarization_charge, coulomb, result.field);
            if (!valid_field(result.field, size))
            {
                result.status = PolarizationStatus::NonFinite;
            }
            return result;
        }

        for (std::size_t index = 0; index < size; ++index)
        {
            result.polarization_charge[index]
                += parameters.mixing * (trial[index] - result.polarization_charge[index]);
        }
    }

    result.status = PolarizationStatus::MaxIterations;
    evaluate_field(solute_charge, result.polarization_charge, coulomb, result.field);
    if (!valid_field(result.field, size))
    {
        result.status = PolarizationStatus::NonFinite;
    }
    return result;
}

} // namespace ModuleSccs
