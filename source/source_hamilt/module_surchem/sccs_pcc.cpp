#include "sccs_pcc.h"

#include "source_base/constants.h"

#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{
namespace
{

double dot(const ModuleBase::Vector3<double>& left, const ModuleBase::Vector3<double>& right)
{
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

double norm_squared(const ModuleBase::Vector3<double>& value)
{
    return dot(value, value);
}

} // namespace

void validate_pcc_parameters(const PccParameters& parameters)
{
    if (!std::isfinite(parameters.cube_length) || parameters.cube_length <= 0.0)
    {
        throw std::invalid_argument("zero-dimensional PCC requires a positive finite cube length");
    }
    if (!std::isfinite(parameters.madelung) || parameters.madelung <= 0.0)
    {
        throw std::invalid_argument("zero-dimensional PCC requires a positive finite Madelung constant");
    }
}

MultipoleMoments point_charge_moments(const std::vector<PointCharge>& charges,
                                      const ModuleBase::Vector3<double>& origin)
{
    MultipoleMoments moments;
    for (std::size_t index = 0; index < charges.size(); ++index)
    {
        const PointCharge& point = charges[index];
        if (!std::isfinite(point.charge) || !std::isfinite(point.position.x)
            || !std::isfinite(point.position.y) || !std::isfinite(point.position.z))
        {
            throw std::domain_error("PCC point charges and positions must be finite");
        }
        const ModuleBase::Vector3<double> relative(point.position.x - origin.x,
                                                    point.position.y - origin.y,
                                                    point.position.z - origin.z);
        moments.charge += point.charge;
        moments.dipole.x += point.charge * relative.x;
        moments.dipole.y += point.charge * relative.y;
        moments.dipole.z += point.charge * relative.z;
        moments.quadrupole_trace += point.charge * norm_squared(relative);
    }
    return moments;
}

MultipoleMoments density_moments(const std::vector<double>& density,
                                 const std::vector<ModuleBase::Vector3<double>>& positions,
                                 const double volume_element,
                                 const ModuleBase::Vector3<double>& origin)
{
    if (density.size() != positions.size())
    {
        throw std::invalid_argument("PCC density and position arrays must have the same size");
    }
    if (!std::isfinite(volume_element) || volume_element <= 0.0)
    {
        throw std::invalid_argument("PCC density integration requires a positive finite volume element");
    }

    std::vector<PointCharge> charges(density.size());
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        charges[index].charge = density[index] * volume_element;
        charges[index].position = positions[index];
    }
    return point_charge_moments(charges, origin);
}

double pcc_potential(const MultipoleMoments& moments,
                     const ModuleBase::Vector3<double>& position,
                     const PccParameters& parameters)
{
    validate_pcc_parameters(parameters);
    const double volume = parameters.cube_length * parameters.cube_length * parameters.cube_length;
    const double parabolic = moments.charge * norm_squared(position)
                             - 2.0 * dot(moments.dipole, position)
                             + moments.quadrupole_trace;
    return parameters.madelung * moments.charge / parameters.cube_length
           - 2.0 * ModuleBase::PI * parabolic / (3.0 * volume);
}

ModuleBase::Vector3<double> pcc_potential_gradient(const MultipoleMoments& moments,
                                                   const ModuleBase::Vector3<double>& position,
                                                   const PccParameters& parameters)
{
    validate_pcc_parameters(parameters);
    const double volume = parameters.cube_length * parameters.cube_length * parameters.cube_length;
    const double factor = -4.0 * ModuleBase::PI / (3.0 * volume);
    return ModuleBase::Vector3<double>(factor * (moments.charge * position.x - moments.dipole.x),
                                       factor * (moments.charge * position.y - moments.dipole.y),
                                       factor * (moments.charge * position.z - moments.dipole.z));
}

ModuleBase::Vector3<double> pcc_point_charge_force(
    const MultipoleMoments& total_moments,
    const PointCharge& point,
    const ModuleBase::Vector3<double>& origin,
    const PccParameters& parameters)
{
    if (!std::isfinite(point.charge) || !std::isfinite(point.position.x)
        || !std::isfinite(point.position.y) || !std::isfinite(point.position.z)
        || !std::isfinite(origin.x) || !std::isfinite(origin.y) || !std::isfinite(origin.z))
    {
        throw std::domain_error("PCC point charge force requires finite charge and coordinates");
    }
    const ModuleBase::Vector3<double> relative(point.position.x - origin.x,
                                                point.position.y - origin.y,
                                                point.position.z - origin.z);
    const ModuleBase::Vector3<double> gradient
        = pcc_potential_gradient(total_moments, relative, parameters);
    return ModuleBase::Vector3<double>(-point.charge * gradient.x,
                                       -point.charge * gradient.y,
                                       -point.charge * gradient.z);
}

double pcc_bilinear_energy(const MultipoleMoments& left,
                           const MultipoleMoments& right,
                           const PccParameters& parameters)
{
    validate_pcc_parameters(parameters);
    const double volume = parameters.cube_length * parameters.cube_length * parameters.cube_length;
    const double monopole = parameters.madelung * left.charge * right.charge / parameters.cube_length;
    const double multipole = left.quadrupole_trace * right.charge
                             + left.charge * right.quadrupole_trace
                             - 2.0 * dot(left.dipole, right.dipole);
    return monopole - 2.0 * ModuleBase::PI * multipole / (3.0 * volume);
}

double pcc_self_energy(const MultipoleMoments& moments, const PccParameters& parameters)
{
    return 0.5 * pcc_bilinear_energy(moments, moments, parameters);
}

} // namespace ModuleSccs
