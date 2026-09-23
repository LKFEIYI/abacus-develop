#ifndef SCCS_PCC_H
#define SCCS_PCC_H

#include "source_base/vector3.h"

#include <vector>

namespace ModuleSccs
{

struct MultipoleMoments
{
    double charge = 0.0;
    ModuleBase::Vector3<double> dipole;
    double quadrupole_trace = 0.0;
};

struct PointCharge
{
    double charge = 0.0;
    ModuleBase::Vector3<double> position;
};

struct PccParameters
{
    double cube_length = 0.0;
    double madelung = 2.837297479480619;
};

void validate_pcc_parameters(const PccParameters& parameters);

MultipoleMoments point_charge_moments(const std::vector<PointCharge>& charges,
                                      const ModuleBase::Vector3<double>& origin);

MultipoleMoments density_moments(const std::vector<double>& density,
                                 const std::vector<ModuleBase::Vector3<double>>& positions,
                                 double volume_element,
                                 const ModuleBase::Vector3<double>& origin);

double pcc_potential(const MultipoleMoments& moments,
                     const ModuleBase::Vector3<double>& position,
                     const PccParameters& parameters);

ModuleBase::Vector3<double> pcc_potential_gradient(const MultipoleMoments& moments,
                                                   const ModuleBase::Vector3<double>& position,
                                                   const PccParameters& parameters);

ModuleBase::Vector3<double> pcc_point_charge_force(const MultipoleMoments& total_moments,
                                                   const PointCharge& point,
                                                   const ModuleBase::Vector3<double>& origin,
                                                   const PccParameters& parameters);

double pcc_bilinear_energy(const MultipoleMoments& left,
                           const MultipoleMoments& right,
                           const PccParameters& parameters);

double pcc_self_energy(const MultipoleMoments& moments, const PccParameters& parameters);

} // namespace ModuleSccs

#endif
