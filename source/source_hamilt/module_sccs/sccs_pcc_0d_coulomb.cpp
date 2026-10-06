#include "sccs_pcc_0d_coulomb.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"

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

void Pcc0dCoulombOperator::apply_potential(const std::vector<double>& charge, std::vector<double>& potential)
{
    const double dv = basis_.omega / basis_.nxyz;
    const double* charge_data = charge.data();
    const ModuleBase::Vector3<double>* position_data = relative_positions_.data();
    elecstate::ChargeMoments moments = elecstate::charge_moments(charge_data, position_data, basis_.nrxx, dv);
    Parallel_Reduce::reduce_pool(moments.charge);
    Parallel_Reduce::reduce_pool(moments.dipole.x);
    Parallel_Reduce::reduce_pool(moments.dipole.y);
    Parallel_Reduce::reduce_pool(moments.dipole.z);
    Parallel_Reduce::reduce_pool(moments.second_moment);
    periodic_.apply_potential(charge, potential);
    for (int ir = 0; ir < basis_.nrxx; ++ir)
    {
        potential[ir] += elecstate::pcc_0d_potential(moments, relative_positions_[ir], parameters_);
    }
}
} // namespace ModuleSccs
