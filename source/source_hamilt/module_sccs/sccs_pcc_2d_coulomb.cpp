#include "sccs_pcc_2d_coulomb.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"

#include <cmath>

namespace ModuleSccs
{
Pcc2dCoulombOperator::Pcc2dCoulombOperator(
    const ModulePW::PW_Basis& basis,
    double tpiba,
    const std::vector<ModuleBase::Vector3<double>>& projected_positions,
    const ModuleBase::Vector3<double>& normal,
    const elecstate::Pcc2dParameters& parameters)
    : basis_(basis), periodic_(basis, tpiba), projected_positions_(projected_positions),
      normal_(normal), parameters_(parameters)
{
}

bool Pcc2dCoulombOperator::has_boundary_correction() const
{
    return true;
}

bool Pcc2dCoulombOperator::apply_potential(const std::vector<double>& charge,
                                         std::vector<double>& potential,
                                         std::string& error)
{
    // Validate collectively before any rank enters a moment reduction or FFT.
    const double volume = parameters_.area * parameters_.length;
    const double volume_difference = volume - basis_.omega;
    const double normal_square = normal_ * normal_;
    const double normalization_error = normal_square - 1.0;
    double invalid = 0.0;
    if (projected_positions_.size() != static_cast<std::size_t>(basis_.nrxx)
        || !std::isfinite(parameters_.area) || parameters_.area <= 0.0
        || !std::isfinite(parameters_.length) || parameters_.length <= 0.0
        || !std::isfinite(volume) || !std::isfinite(basis_.omega) || basis_.omega <= 0.0
        || std::abs(volume_difference) > 1e-10 * basis_.omega
        || !std::isfinite(normal_square) || std::abs(normalization_error) > 1e-10)
    {
        invalid = 1.0;
    }
    for (const ModuleBase::Vector3<double>& position : projected_positions_)
    {
        const double coordinate = position * normal_;
        const ModuleBase::Vector3<double> transverse = position - normal_ * coordinate;
        const double transverse_square = transverse * transverse;
        if (!std::isfinite(coordinate) || !std::isfinite(transverse_square)
            || transverse_square > 1e-20 * (1.0 + coordinate * coordinate))
        {
            invalid = 1.0;
        }
    }
    Parallel_Reduce::reduce_max_pool(basis_.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS PCC 2D requires normal-projected grid positions, a unit normal and parameters matching the PW volume";
        return false;
    }
    if (!validate_grid_values(charge, basis_, error))
    {
        return false;
    }
    const double dv = basis_.omega / basis_.nxyz;
    elecstate::ChargeMoments moments;
    const bool moments_valid = elecstate::charge_moments(charge.data(), projected_positions_.data(),
                                                        basis_.nrxx, dv, moments, error);
    invalid = moments_valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis_.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS PCC 2D requires finite grid-charge moments on every pool rank";
        return false;
    }
    Parallel_Reduce::reduce_pool(moments.charge);
    Parallel_Reduce::reduce_pool(moments.dipole.x);
    Parallel_Reduce::reduce_pool(moments.dipole.y);
    Parallel_Reduce::reduce_pool(moments.dipole.z);
    Parallel_Reduce::reduce_pool(moments.second_moment);
    std::vector<double> candidate;
    if (!periodic_.apply_potential(charge, candidate, error))
    {
        return false;
    }
    for (int ir = 0; ir < basis_.nrxx; ++ir)
    {
        const double coordinate = projected_positions_[ir] * normal_;
        candidate[ir] += elecstate::pcc_2d_potential(moments, coordinate, normal_, parameters_);
    }
    if (!validate_grid_values(candidate, basis_, error))
    {
        return false;
    }
    potential.swap(candidate);
    return true;
}
} // namespace ModuleSccs
