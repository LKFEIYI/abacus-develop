#include "sccs_pcc_2d_coulomb.h"

#include "source_basis/module_pw/pw_basis.h"

#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{

Pcc2dMoments reduced_pcc_2d_density_moments(
    const std::vector<double>& density,
    const std::vector<ModuleBase::Vector3<double>>& positions,
    const double volume_element,
    const double origin_y,
    const ChargeReduction& reduction)
{
    Pcc2dMoments moments
        = pcc_2d_density_moments(density, positions, volume_element, origin_y);
    double values[3] = {moments.charge, moments.dipole_y, moments.quadrupole_yy};
    reduction.reduce_sum(values, 3);
    if (!std::isfinite(values[0]) || !std::isfinite(values[1])
        || !std::isfinite(values[2]))
    {
        throw std::domain_error("two-dimensional PCC reduced moments must be finite");
    }
    moments.charge = values[0];
    moments.dipole_y = values[1];
    moments.quadrupole_yy = values[2];
    return moments;
}

std::vector<double> pcc_2d_plane_average(const std::vector<double>& values,
                                         const ModulePW::PW_Basis& basis,
                                         const ChargeReduction& reduction)
{
    if (basis.nx <= 0 || basis.ny <= 0 || basis.nz <= 0 || basis.nplane <= 0
        || basis.startz_current < 0 || basis.startz_current + basis.nplane > basis.nz
        || basis.nrxx != basis.nx * basis.ny * basis.nplane
        || values.size() != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument(
            "two-dimensional PCC plane average requires a valid local PW grid");
    }

    std::vector<double> average(basis.ny, 0.0);
    for (int ix = 0; ix < basis.nx; ++ix)
    {
        for (int iy = 0; iy < basis.ny; ++iy)
        {
            for (int iz_local = 0; iz_local < basis.nplane; ++iz_local)
            {
                const int index = (ix * basis.ny + iy) * basis.nplane + iz_local;
                if (!std::isfinite(values[index]))
                {
                    throw std::domain_error(
                        "two-dimensional PCC plane-average values must be finite");
                }
                average[iy] += values[index];
            }
        }
    }
    reduction.reduce_sum(average.data(), basis.ny);
    const double plane_size = static_cast<double>(basis.nx) * static_cast<double>(basis.nz);
    for (int iy = 0; iy < basis.ny; ++iy)
    {
        average[iy] /= plane_size;
        if (!std::isfinite(average[iy]))
        {
            throw std::domain_error(
                "two-dimensional PCC reduced plane averages must be finite");
        }
    }
    return average;
}

Pcc2dCoulombOperator::Pcc2dCoulombOperator(
    const ModulePW::PW_Basis& basis,
    const double tpiba,
    const std::vector<ModuleBase::Vector3<double>>& positions,
    const double volume_element,
    const Pcc2dGeometry& geometry,
    const ChargeReduction& reduction)
    : periodic_(basis, tpiba),
      positions_(positions),
      volume_element_(volume_element),
      geometry_(geometry),
      reduction_(reduction)
{
    validate_pcc_2d_parameters(geometry_.parameters);
    if (positions_.size() != static_cast<std::size_t>(basis.nrxx))
    {
        throw std::invalid_argument(
            "two-dimensional SCCS PCC positions must match the local PW grid");
    }
    if (!std::isfinite(volume_element_) || volume_element_ <= 0.0
        || !std::isfinite(geometry_.origin_y))
    {
        throw std::invalid_argument(
            "two-dimensional SCCS PCC integration geometry must be finite and positive");
    }
}

void Pcc2dCoulombOperator::apply(const std::vector<double>& charge,
                                 ElectrostaticField& field) const
{
    periodic_.apply(charge, field);
    const Pcc2dMoments moments = reduced_pcc_2d_density_moments(charge,
                                                               positions_,
                                                               volume_element_,
                                                               geometry_.origin_y,
                                                               reduction_);
    for (std::size_t index = 0; index < charge.size(); ++index)
    {
        const double relative_y = positions_[index].y - geometry_.origin_y;
        field.potential[index]
            += pcc_2d_potential(moments, relative_y, geometry_.parameters);
        field.gradient[index].y
            += pcc_2d_potential_gradient(moments, relative_y, geometry_.parameters).y;
    }
}

} // namespace ModuleSccs
