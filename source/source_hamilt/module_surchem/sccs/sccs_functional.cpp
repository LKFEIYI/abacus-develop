#include "sccs_functional.h"

#include "../common/charge_reduction.h"
#include "../common/thread_sum.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{

ElectrostaticFunctionalResult evaluate_electrostatic_functional(
    const std::vector<double>& solute_charge,
    const std::vector<double>& dielectric_potential,
    const std::vector<double>& vacuum_potential,
    const std::vector<double>& cavity_potential,
    const double volume_element,
    const ModuleSurchem::ChargeReduction& reduction)
{
    const std::size_t size = solute_charge.size();
    if (size == 0 || dielectric_potential.size() != size
        || vacuum_potential.size() != size || cavity_potential.size() != size)
    {
        throw std::invalid_argument("SCCS electrostatic functional arrays must have the same non-zero size");
    }
    if (!std::isfinite(volume_element) || volume_element <= 0.0)
    {
        throw std::invalid_argument("SCCS electrostatic functional requires a positive finite volume element");
    }

    ElectrostaticFunctionalResult result;
    result.reaction_potential.resize(size);
    result.electron_potential.resize(size);
    // Slot 1 counts the non-finite inputs, which are not thrown inside the
    // parallel loop.
    const auto add_point = [&](const std::size_t index, std::array<double, 2>& sums) {
        if (!std::isfinite(solute_charge[index])
            || !std::isfinite(dielectric_potential[index])
            || !std::isfinite(vacuum_potential[index])
            || !std::isfinite(cavity_potential[index]))
        {
            sums[1] += 1.0;
            return;
        }
        const double reaction_potential = dielectric_potential[index] - vacuum_potential[index];
        result.reaction_potential[index] = reaction_potential;
        sums[0] += 0.5 * solute_charge[index] * reaction_potential * volume_element;
        result.electron_potential[index] = -reaction_potential + cavity_potential[index];
    };
    const std::array<double, 2> sums = ModuleSurchem::thread_sums<2>(size, add_point);
    if (sums[1] > 0.0)
    {
        throw std::domain_error("SCCS electrostatic functional inputs must be finite");
    }
    result.reaction_energy = sums[0];
    reduction.reduce_sum(result.reaction_energy);
    if (!std::isfinite(result.reaction_energy))
    {
        throw std::domain_error("SCCS reduced reaction energy must be finite");
    }
    return result;
}

} // namespace ModuleSccs
