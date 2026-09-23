#include "../sccs_pcc.h"

#include "source_base/constants.h"
#include "gtest/gtest.h"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

TEST(SccsPcc, AccumulatesPointChargeMomentsAboutTheRequestedOrigin)
{
    std::vector<ModuleSccs::PointCharge> charges(2);
    charges[0].charge = 2.0;
    charges[0].position = ModuleBase::Vector3<double>(2.0, 0.0, 0.0);
    charges[1].charge = -1.0;
    charges[1].position = ModuleBase::Vector3<double>(0.0, 2.0, 0.0);

    const ModuleSccs::MultipoleMoments moments
        = ModuleSccs::point_charge_moments(charges, ModuleBase::Vector3<double>(1.0, 1.0, 0.0));
    EXPECT_DOUBLE_EQ(moments.charge, 1.0);
    EXPECT_DOUBLE_EQ(moments.dipole.x, 3.0);
    EXPECT_DOUBLE_EQ(moments.dipole.y, -3.0);
    EXPECT_DOUBLE_EQ(moments.dipole.z, 0.0);
    EXPECT_DOUBLE_EQ(moments.quadrupole_trace, 2.0);
}

TEST(SccsPcc, PotentialGradientMatchesCentralDifferences)
{
    ModuleSccs::MultipoleMoments moments;
    moments.charge = -1.2;
    moments.dipole = ModuleBase::Vector3<double>(0.3, -0.7, 0.2);
    moments.quadrupole_trace = 4.1;
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 18.0;
    const ModuleBase::Vector3<double> position(0.9, -1.3, 0.4);
    const ModuleBase::Vector3<double> gradient
        = ModuleSccs::pcc_potential_gradient(moments, position, parameters);
    const double step = 1.0e-5;

    for (int direction = 0; direction < 3; ++direction)
    {
        ModuleBase::Vector3<double> plus = position;
        ModuleBase::Vector3<double> minus = position;
        if (direction == 0)
        {
            plus.x += step;
            minus.x -= step;
        }
        else if (direction == 1)
        {
            plus.y += step;
            minus.y -= step;
        }
        else
        {
            plus.z += step;
            minus.z -= step;
        }
        const double finite = (ModuleSccs::pcc_potential(moments, plus, parameters)
                               - ModuleSccs::pcc_potential(moments, minus, parameters))
                              / (2.0 * step);
        const double analytic = direction == 0 ? gradient.x : (direction == 1 ? gradient.y : gradient.z);
        EXPECT_NEAR(analytic, finite, 1.0e-11);
    }
}

TEST(SccsPcc, IntegratesDensityMomentsWithTheVolumeElement)
{
    const std::vector<double> density{0.5, -0.25};
    const std::vector<ModuleBase::Vector3<double>> positions{
        ModuleBase::Vector3<double>(1.0, 0.0, 0.0),
        ModuleBase::Vector3<double>(0.0, 2.0, 0.0)};
    const ModuleSccs::MultipoleMoments moments
        = ModuleSccs::density_moments(density,
                                      positions,
                                      2.0,
                                      ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(moments.charge, 0.5);
    EXPECT_DOUBLE_EQ(moments.dipole.x, 1.0);
    EXPECT_DOUBLE_EQ(moments.dipole.y, -1.0);
    EXPECT_DOUBLE_EQ(moments.quadrupole_trace, -1.0);
}

TEST(SccsPcc, BilinearKernelIsSymmetric)
{
    ModuleSccs::MultipoleMoments left;
    left.charge = 1.0;
    left.dipole = ModuleBase::Vector3<double>(0.1, 0.2, 0.3);
    left.quadrupole_trace = 2.0;
    ModuleSccs::MultipoleMoments right;
    right.charge = -2.0;
    right.dipole = ModuleBase::Vector3<double>(-0.4, 0.7, 0.2);
    right.quadrupole_trace = -1.0;
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 20.0;

    EXPECT_DOUBLE_EQ(ModuleSccs::pcc_bilinear_energy(left, right, parameters),
                     ModuleSccs::pcc_bilinear_energy(right, left, parameters));
}

TEST(SccsPcc, SelfEnergyHasExpectedMakovPayneForm)
{
    ModuleSccs::MultipoleMoments moments;
    moments.charge = 1.5;
    moments.dipole = ModuleBase::Vector3<double>(0.2, -0.4, 0.7);
    moments.quadrupole_trace = 3.2;
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 14.0;
    const double volume = std::pow(parameters.cube_length, 3);
    const double dipole_squared = 0.2 * 0.2 + 0.4 * 0.4 + 0.7 * 0.7;
    const double expected = parameters.madelung * moments.charge * moments.charge
                                / (2.0 * parameters.cube_length)
                            - 2.0 * ModuleBase::PI
                                  * (moments.charge * moments.quadrupole_trace - dipole_squared)
                                  / (3.0 * volume);
    EXPECT_NEAR(ModuleSccs::pcc_self_energy(moments, parameters), expected, 1.0e-15);
}

TEST(SccsPcc, SelfEnergyIsIndependentOfTheMultipoleOrigin)
{
    std::vector<ModuleSccs::PointCharge> charges(2);
    charges[0].charge = 1.7;
    charges[0].position = ModuleBase::Vector3<double>(1.2, -0.5, 0.8);
    charges[1].charge = -0.4;
    charges[1].position = ModuleBase::Vector3<double>(-0.3, 0.6, 1.1);
    const ModuleSccs::MultipoleMoments first
        = ModuleSccs::point_charge_moments(charges, ModuleBase::Vector3<double>(0.0, 0.0, 0.0));
    const ModuleSccs::MultipoleMoments second
        = ModuleSccs::point_charge_moments(charges, ModuleBase::Vector3<double>(0.7, -1.0, 0.2));
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 24.0;

    EXPECT_NEAR(ModuleSccs::pcc_self_energy(first, parameters),
                ModuleSccs::pcc_self_energy(second, parameters),
                1.0e-15);
}

TEST(SccsPcc, PointIonVacuumEnergyDerivativeMatchesElectronPotential)
{
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    const ModuleBase::Vector3<double> electron_position(1.1, -0.8, 0.4);
    const double volume_element = 0.3;
    const double electron_density = 0.7;
    const double step = 1.0e-6;
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 19.0;

    const auto energy = [&](const double density) {
        std::vector<ModuleSccs::PointCharge> charges(2);
        charges[0].charge = 1.0;
        charges[0].position = ModuleBase::Vector3<double>(-0.2, 0.5, -0.1);
        charges[1].charge = -density * volume_element;
        charges[1].position = electron_position;
        return ModuleSccs::pcc_self_energy(ModuleSccs::point_charge_moments(charges, origin),
                                            parameters);
    };

    std::vector<ModuleSccs::PointCharge> center_charges(2);
    center_charges[0].charge = 1.0;
    center_charges[0].position = ModuleBase::Vector3<double>(-0.2, 0.5, -0.1);
    center_charges[1].charge = -electron_density * volume_element;
    center_charges[1].position = electron_position;
    const ModuleSccs::MultipoleMoments center
        = ModuleSccs::point_charge_moments(center_charges, origin);
    const double analytic = -volume_element
                            * ModuleSccs::pcc_potential(center,
                                                       electron_position,
                                                       parameters);
    const double finite = (energy(electron_density + step)
                           - energy(electron_density - step))
                          / (2.0 * step);
    EXPECT_NEAR(analytic, finite, 1.0e-11);
}

TEST(SccsPcc, PointChargeForceMatchesSelfEnergyFiniteDifference)
{
    const ModuleBase::Vector3<double> origin(0.0, 0.0, 0.0);
    std::vector<ModuleSccs::PointCharge> charges(2);
    charges[0].charge = 1.2;
    charges[0].position = ModuleBase::Vector3<double>(0.7, -0.4, 0.2);
    charges[1].charge = -0.3;
    charges[1].position = ModuleBase::Vector3<double>(-0.6, 0.5, 0.9);
    ModuleSccs::PccParameters parameters;
    parameters.cube_length = 17.0;
    const ModuleSccs::MultipoleMoments moments
        = ModuleSccs::point_charge_moments(charges, origin);
    const ModuleBase::Vector3<double> analytic
        = ModuleSccs::pcc_point_charge_force(moments, charges[0], origin, parameters);
    const double step = 1.0e-5;

    for (int direction = 0; direction < 3; ++direction)
    {
        std::vector<ModuleSccs::PointCharge> plus = charges;
        std::vector<ModuleSccs::PointCharge> minus = charges;
        plus[0].position[direction] += step;
        minus[0].position[direction] -= step;
        const double energy_plus
            = ModuleSccs::pcc_self_energy(ModuleSccs::point_charge_moments(plus, origin),
                                           parameters);
        const double energy_minus
            = ModuleSccs::pcc_self_energy(ModuleSccs::point_charge_moments(minus, origin),
                                           parameters);
        const double finite_force = -(energy_plus - energy_minus) / (2.0 * step);
        EXPECT_NEAR(analytic[direction], finite_force, 1.0e-11);
    }
}

TEST(SccsPcc, RejectsInvalidCubeLength)
{
    ModuleSccs::PccParameters parameters;
    EXPECT_THROW(ModuleSccs::pcc_self_energy(ModuleSccs::MultipoleMoments(), parameters),
                 std::invalid_argument);
}

} // namespace
