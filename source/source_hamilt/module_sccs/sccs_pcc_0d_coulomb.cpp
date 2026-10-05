#include "sccs_pcc_0d_coulomb.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"

#include <cmath>

namespace ModuleSccs
{
Pcc0dCoulombOperator::Pcc0dCoulombOperator(
    const ModulePW::PW_Basis& basis,
    double tpiba,
    const std::vector<ModuleBase::Vector3<double>>& relative_positions,
    const elecstate::Pcc0dParameters& parameters)
    : basis_(basis), periodic_(basis, tpiba), relative_positions_(relative_positions), parameters_(parameters)
{
}

bool Pcc0dCoulombOperator::has_boundary_correction() const
{
    return true;
}

bool Pcc0dCoulombOperator::apply_potential(const std::vector<double>& charge,
                                         std::vector<double>& potential,
                                         std::string& error)
{
    // Reject rank-local geometry errors before any rank enters the FFTs.
    const double length = parameters_.length;
    const double volume = length * length * length;
    double invalid = 0.0;
    if (relative_positions_.size() != static_cast<std::size_t>(basis_.nrxx)
        || !std::isfinite(length) || length <= 0.0 || !std::isfinite(volume)
        || !std::isfinite(basis_.omega) || basis_.omega <= 0.0
        || std::abs(volume - basis_.omega) > 1e-10 * basis_.omega
        || !std::isfinite(parameters_.madelung))
    {
        invalid = 1.0;
    }
    Parallel_Reduce::reduce_max_pool(basis_.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS PCC 0D requires local grid positions and cubic parameters matching the PW volume";
        return false;
    }
    if (!validate_grid_values(charge, basis_, error))
    {
        return false;
    }
    const double dv = basis_.omega / basis_.nxyz;
    elecstate::ChargeMoments moments;
    const bool moments_valid = elecstate::charge_moments(charge.data(), relative_positions_.data(),
                                                        basis_.nrxx, dv, moments, error);
    invalid = moments_valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis_.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS PCC 0D requires finite grid-charge moments and positions on every pool rank";
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
        candidate[ir] += elecstate::pcc_0d_potential(moments, relative_positions_[ir], parameters_);
    }
    if (!validate_grid_values(candidate, basis_, error))
    {
        return false;
    }
    potential.swap(candidate);
    return true;
}
} // namespace ModuleSccs
