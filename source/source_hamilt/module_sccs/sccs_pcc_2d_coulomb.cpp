#include "sccs_pcc_2d_coulomb.h"

#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"

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

void Pcc2dCoulombOperator::apply_potential(const std::vector<double>& charge, std::vector<double>& potential)
{
    const double dv = basis_.omega / basis_.nxyz;
    const double* charge_data = charge.data();
    const ModuleBase::Vector3<double>* position_data = projected_positions_.data();
    elecstate::ChargeMoments moments = elecstate::charge_moments(charge_data, position_data, basis_.nrxx, dv);
    Parallel_Reduce::reduce_pool(moments.charge);
    Parallel_Reduce::reduce_pool(moments.dipole.x);
    Parallel_Reduce::reduce_pool(moments.dipole.y);
    Parallel_Reduce::reduce_pool(moments.dipole.z);
    Parallel_Reduce::reduce_pool(moments.second_moment);
    periodic_.apply_potential(charge, potential);
    for (int ir = 0; ir < basis_.nrxx; ++ir)
    {
        const double coordinate = projected_positions_[ir] * normal_;
        potential[ir] += elecstate::pcc_2d_potential(moments, coordinate, normal_, parameters_);
    }
}
} // namespace ModuleSccs
