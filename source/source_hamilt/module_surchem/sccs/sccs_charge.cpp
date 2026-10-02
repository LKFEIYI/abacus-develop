#include "sccs_charge.h"
#include "../common/charge_reduction.h"
#include "../common/thread_sum.h"

#include <array>
#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{

std::vector<double> sum_electron_density(const std::vector<std::vector<double>>& spin_density,
                                         const int charge_channels)
{
    if (spin_density.empty() || charge_channels <= 0
        || charge_channels > static_cast<int>(spin_density.size()))
    {
        throw std::invalid_argument("SCCS electron density requires valid charge-bearing spin channels");
    }
    const std::size_t size = spin_density[0].size();
    if (size == 0)
    {
        throw std::invalid_argument("SCCS electron density grid must not be empty");
    }

    std::vector<double> electron_density(size, 0.0);
    for (int channel = 0; channel < charge_channels; ++channel)
    {
        if (spin_density[channel].size() != size)
        {
            throw std::invalid_argument("SCCS spin-density arrays must have the same size");
        }
        for (std::size_t index = 0; index < size; ++index)
        {
            if (!std::isfinite(spin_density[channel][index]))
            {
                throw std::domain_error("SCCS electron density must be finite");
            }
            electron_density[index] += spin_density[channel][index];
        }
    }
    return electron_density;
}

ChargeDensity assemble_charge_density(const std::vector<double>& electron_density,
                                      const std::vector<double>& ionic_density,
                                      const double volume_element,
                                      const double expected_ionic_charge,
                                      const double normalization_tolerance,
                                      const ModuleSurchem::ChargeReduction& reduction)
{
    if (electron_density.empty() || electron_density.size() != ionic_density.size())
    {
        throw std::invalid_argument("SCCS electron and ionic density arrays must have the same non-zero size");
    }
    if (!std::isfinite(volume_element) || volume_element <= 0.0
        || !std::isfinite(expected_ionic_charge) || expected_ionic_charge < 0.0
        || !std::isfinite(normalization_tolerance) || normalization_tolerance <= 0.0)
    {
        throw std::invalid_argument("SCCS charge normalization inputs must be finite and physically valid");
    }

    ChargeDensity result;
    result.electron = electron_density;
    result.solute.resize(electron_density.size());
    // Slot 2 counts the non-finite densities, which are not thrown inside the
    // parallel loop.
    const auto add_point = [&](const std::size_t index, std::array<double, 3>& sums) {
        if (!std::isfinite(electron_density[index]) || !std::isfinite(ionic_density[index]))
        {
            sums[2] += 1.0;
            return;
        }
        sums[0] += electron_density[index] * volume_element;
        sums[1] += ionic_density[index] * volume_element;
        result.solute[index] = ionic_density[index] - electron_density[index];
    };
    const std::array<double, 3> sums
        = ModuleSurchem::thread_sums<3>(electron_density.size(), add_point);
    if (sums[2] > 0.0)
    {
        throw std::domain_error("SCCS charge densities must be finite");
    }
    result.electron_count = sums[0];
    result.ionic_charge = sums[1];
    reduction.reduce_sum(result.electron_count);
    reduction.reduce_sum(result.ionic_charge);
    if (!std::isfinite(result.electron_count) || !std::isfinite(result.ionic_charge))
    {
        throw std::domain_error("SCCS reduced charge integrals must be finite");
    }
    const double ionic_charge_error = result.ionic_charge - expected_ionic_charge;
    if (std::abs(ionic_charge_error) > normalization_tolerance)
    {
        throw std::runtime_error("SCCS ionic density normalization does not match the valence charge");
    }
    result.net_charge = result.ionic_charge - result.electron_count;
    return result;
}

} // namespace ModuleSccs
