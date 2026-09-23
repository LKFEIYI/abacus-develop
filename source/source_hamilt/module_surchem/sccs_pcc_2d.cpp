#include "sccs_pcc_2d.h"

#include "source_base/constants.h"
#include "source_base/matrix3.h"

#include <cmath>
#include <stdexcept>

namespace ModuleSccs
{
namespace
{

ModuleBase::Vector3<double> lattice_row(const ModuleBase::Matrix3& lattice,
                                        const int row,
                                        const double scale)
{
    if (row == 0)
    {
        return ModuleBase::Vector3<double>(scale * lattice.e11,
                                           scale * lattice.e12,
                                           scale * lattice.e13);
    }
    if (row == 1)
    {
        return ModuleBase::Vector3<double>(scale * lattice.e21,
                                           scale * lattice.e22,
                                           scale * lattice.e23);
    }
    return ModuleBase::Vector3<double>(scale * lattice.e31,
                                       scale * lattice.e32,
                                       scale * lattice.e33);
}

bool finite_position(const ModuleBase::Vector3<double>& position)
{
    return std::isfinite(position.x) && std::isfinite(position.y)
           && std::isfinite(position.z);
}

double norm(const ModuleBase::Vector3<double>& vector)
{
    return std::sqrt(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
}

ModuleBase::Vector3<double> cross(const ModuleBase::Vector3<double>& left,
                                  const ModuleBase::Vector3<double>& right)
{
    return ModuleBase::Vector3<double>(left.y * right.z - left.z * right.y,
                                       left.z * right.x - left.x * right.z,
                                       left.x * right.y - left.y * right.x);
}

double potential_constant(const Pcc2dParameters& parameters)
{
    // Andreussi-Marzari, Phys. Rev. B 90, 245101 (2014), Eq. (88).
    return ModuleBase::PI / (3.0 * parameters.cell_length_y);
}

double inverse_volume_factor(const Pcc2dParameters& parameters)
{
    return 2.0 * ModuleBase::PI
           / (parameters.periodic_area * parameters.cell_length_y);
}

} // namespace

Pcc2dGeometry pcc_2d_geometry(const ModuleBase::Matrix3& lattice_vectors,
                              const double lattice_scale,
                              const double relative_tolerance)
{
    if (!std::isfinite(lattice_scale) || lattice_scale <= 0.0
        || !std::isfinite(relative_tolerance) || relative_tolerance <= 0.0)
    {
        throw std::invalid_argument(
            "two-dimensional PCC geometry requires positive finite scale and tolerance");
    }

    const ModuleBase::Vector3<double> a
        = lattice_row(lattice_vectors, 0, lattice_scale);
    const ModuleBase::Vector3<double> b
        = lattice_row(lattice_vectors, 1, lattice_scale);
    const ModuleBase::Vector3<double> c
        = lattice_row(lattice_vectors, 2, lattice_scale);
    if (!finite_position(a) || !finite_position(b) || !finite_position(c))
    {
        throw std::invalid_argument("two-dimensional PCC lattice vectors must be finite");
    }

    const double length_a = norm(a);
    const double length_b = norm(b);
    const double length_c = norm(c);
    if (!std::isfinite(length_a) || !std::isfinite(length_b) || !std::isfinite(length_c)
        || length_a <= 0.0 || length_b <= 0.0 || length_c <= 0.0)
    {
        throw std::invalid_argument(
            "two-dimensional PCC lattice vectors must have positive lengths");
    }
    if (b.y <= 0.0 || std::abs(a.y) > relative_tolerance * length_a
        || std::abs(c.y) > relative_tolerance * length_c
        || std::abs(b.x) > relative_tolerance * length_b
        || std::abs(b.z) > relative_tolerance * length_b)
    {
        throw std::invalid_argument(
            "two-dimensional PCC requires a and c in the x-z plane and b along +y");
    }

    const double periodic_area = norm(cross(a, c));
    if (!std::isfinite(periodic_area)
        || periodic_area <= relative_tolerance * length_a * length_c)
    {
        throw std::invalid_argument(
            "two-dimensional PCC requires independent in-plane lattice vectors");
    }

    Pcc2dGeometry geometry;
    geometry.parameters.periodic_area = periodic_area;
    geometry.parameters.cell_length_y = length_b;
    geometry.origin_y = 0.5 * (a.y + b.y + c.y);
    validate_pcc_2d_parameters(geometry.parameters);
    return geometry;
}

void validate_pcc_2d_parameters(const Pcc2dParameters& parameters)
{
    if (!std::isfinite(parameters.periodic_area) || parameters.periodic_area <= 0.0)
    {
        throw std::invalid_argument("two-dimensional PCC requires a positive finite periodic area");
    }
    if (!std::isfinite(parameters.cell_length_y) || parameters.cell_length_y <= 0.0)
    {
        throw std::invalid_argument("two-dimensional PCC requires a positive finite y cell length");
    }
}

Pcc2dMoments pcc_2d_point_charge_moments(const std::vector<PointCharge>& charges,
                                          const double origin_y)
{
    if (!std::isfinite(origin_y))
    {
        throw std::invalid_argument("two-dimensional PCC requires a finite y origin");
    }

    Pcc2dMoments moments;
    for (std::size_t index = 0; index < charges.size(); ++index)
    {
        const PointCharge& point = charges[index];
        if (!std::isfinite(point.charge) || !finite_position(point.position))
        {
            throw std::domain_error("two-dimensional PCC point charges must be finite");
        }
        const double relative_y = point.position.y - origin_y;
        moments.charge += point.charge;
        moments.dipole_y += point.charge * relative_y;
        moments.quadrupole_yy += point.charge * relative_y * relative_y;
    }
    return moments;
}

Pcc2dMoments pcc_2d_density_moments(
    const std::vector<double>& density,
    const std::vector<ModuleBase::Vector3<double>>& positions,
    const double volume_element,
    const double origin_y)
{
    if (density.size() != positions.size())
    {
        throw std::invalid_argument("two-dimensional PCC density and positions must have the same size");
    }
    if (!std::isfinite(volume_element) || volume_element <= 0.0)
    {
        throw std::invalid_argument("two-dimensional PCC requires a positive finite volume element");
    }
    if (!std::isfinite(origin_y))
    {
        throw std::invalid_argument("two-dimensional PCC requires a finite y origin");
    }

    Pcc2dMoments moments;
    for (std::size_t index = 0; index < density.size(); ++index)
    {
        if (!std::isfinite(density[index]) || !finite_position(positions[index]))
        {
            throw std::domain_error("two-dimensional PCC density and positions must be finite");
        }
        const double charge = density[index] * volume_element;
        const double relative_y = positions[index].y - origin_y;
        moments.charge += charge;
        moments.dipole_y += charge * relative_y;
        moments.quadrupole_yy += charge * relative_y * relative_y;
    }
    return moments;
}

double pcc_2d_potential(const Pcc2dMoments& moments,
                        const double relative_y,
                        const Pcc2dParameters& parameters)
{
    validate_pcc_2d_parameters(parameters);
    if (!std::isfinite(moments.charge) || !std::isfinite(moments.dipole_y)
        || !std::isfinite(moments.quadrupole_yy) || !std::isfinite(relative_y))
    {
        throw std::domain_error("two-dimensional PCC potential inputs must be finite");
    }

    const double parabolic = moments.charge * relative_y * relative_y
                             - 2.0 * moments.dipole_y * relative_y
                             + moments.quadrupole_yy;
    return potential_constant(parameters) * moments.charge
           - inverse_volume_factor(parameters) * parabolic;
}

ModuleBase::Vector3<double> pcc_2d_potential_gradient(
    const Pcc2dMoments& moments,
    const double relative_y,
    const Pcc2dParameters& parameters)
{
    validate_pcc_2d_parameters(parameters);
    if (!std::isfinite(moments.charge) || !std::isfinite(moments.dipole_y)
        || !std::isfinite(moments.quadrupole_yy) || !std::isfinite(relative_y))
    {
        throw std::domain_error("two-dimensional PCC gradient inputs must be finite");
    }

    const double gradient_y = -2.0 * inverse_volume_factor(parameters)
                              * (moments.charge * relative_y - moments.dipole_y);
    return ModuleBase::Vector3<double>(0.0, gradient_y, 0.0);
}

ModuleBase::Vector3<double> pcc_2d_point_charge_force(
    const Pcc2dMoments& total_moments,
    const PointCharge& point,
    const double origin_y,
    const Pcc2dParameters& parameters)
{
    if (!std::isfinite(point.charge) || !finite_position(point.position)
        || !std::isfinite(origin_y))
    {
        throw std::domain_error("two-dimensional PCC point-charge force inputs must be finite");
    }
    const ModuleBase::Vector3<double> gradient
        = pcc_2d_potential_gradient(total_moments,
                                    point.position.y - origin_y,
                                    parameters);
    return ModuleBase::Vector3<double>(-point.charge * gradient.x,
                                       -point.charge * gradient.y,
                                       -point.charge * gradient.z);
}

double pcc_2d_bilinear_energy(const Pcc2dMoments& left,
                              const Pcc2dMoments& right,
                              const Pcc2dParameters& parameters)
{
    validate_pcc_2d_parameters(parameters);
    if (!std::isfinite(left.charge) || !std::isfinite(left.dipole_y)
        || !std::isfinite(left.quadrupole_yy) || !std::isfinite(right.charge)
        || !std::isfinite(right.dipole_y) || !std::isfinite(right.quadrupole_yy))
    {
        throw std::domain_error("two-dimensional PCC energy moments must be finite");
    }

    const double multipole = left.charge * right.quadrupole_yy
                             + left.quadrupole_yy * right.charge
                             - 2.0 * left.dipole_y * right.dipole_y;
    return potential_constant(parameters) * left.charge * right.charge
           - inverse_volume_factor(parameters) * multipole;
}

double pcc_2d_self_energy(const Pcc2dMoments& moments,
                          const Pcc2dParameters& parameters)
{
    return 0.5 * pcc_2d_bilinear_energy(moments, moments, parameters);
}

double pcc_2d_ionic_shape_energy(const double polarization_charge,
                                 const double smooth_solute_quadrupole_yy,
                                 const double point_solute_quadrupole_yy,
                                 const Pcc2dParameters& parameters)
{
    validate_pcc_2d_parameters(parameters);
    if (!std::isfinite(polarization_charge)
        || !std::isfinite(smooth_solute_quadrupole_yy)
        || !std::isfinite(point_solute_quadrupole_yy))
    {
        throw std::domain_error("two-dimensional PCC ionic-shape energy inputs must be finite");
    }
    // Andreussi-Marzari, Phys. Rev. B 90, 245101 (2014), Eq. (A2).
    return ModuleBase::PI * polarization_charge
           * (smooth_solute_quadrupole_yy - point_solute_quadrupole_yy)
           / (parameters.periodic_area * parameters.cell_length_y);
}

} // namespace ModuleSccs
