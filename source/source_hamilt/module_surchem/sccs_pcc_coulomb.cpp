#include "sccs_pcc_coulomb.h"

#include "source_basis/module_pw/pw_basis.h"

#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{

PccCoulombOperator::PccCoulombOperator(
    const ModulePW::PW_Basis& basis,
    const double tpiba,
    const std::vector<ModuleBase::Vector3<double>>& positions,
    const double volume_element,
    const ModuleBase::Vector3<double>& origin,
    const PccParameters& parameters,
    const ChargeReduction& reduction)
    : periodic_(basis, tpiba),
      positions_(positions),
      volume_element_(volume_element),
      origin_(origin),
      parameters_(parameters),
      reduction_(reduction)
{
    validate_pcc_parameters(parameters_);
    if (positions_.size() != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument("SCCS PCC positions must match the local PW real-space grid");
    }
    if (!std::isfinite(volume_element_) || volume_element_ <= 0.0
        || !std::isfinite(origin_.x) || !std::isfinite(origin_.y) || !std::isfinite(origin_.z))
    {
        throw std::invalid_argument("SCCS PCC integration geometry must be positive and finite");
    }
}

void PccCoulombOperator::apply(const std::vector<double>& charge,
                               ElectrostaticField& field) const
{
    periodic_.apply(charge, field);
    const MultipoleMoments moments
        = reduced_density_moments(charge, positions_, volume_element_, origin_, reduction_);
    for (std::size_t index = 0; index < charge.size(); ++index)
    {
        const ModuleBase::Vector3<double> relative(positions_[index].x - origin_.x,
                                                    positions_[index].y - origin_.y,
                                                    positions_[index].z - origin_.z);
        field.potential[index] += pcc_potential(moments, relative, parameters_);
        const ModuleBase::Vector3<double> correction
            = pcc_potential_gradient(moments, relative, parameters_);
        field.gradient[index].x += correction.x;
        field.gradient[index].y += correction.y;
        field.gradient[index].z += correction.z;
    }
}

} // namespace ModuleSccs
