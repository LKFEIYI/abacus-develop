#ifndef SCCS_LOWPASS_H
#define SCCS_LOWPASS_H

#include <string>
#include <vector>

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSccs
{
struct CavityParameters;
struct CavityDerivatives;
struct SccsResponse;

// Collective validation. Disabled filtering returns an empty vector.
// Failure leaves filter unchanged.
bool make_switching_filter(const CavityParameters& cavity,
                           const ModulePW::PW_Basis& basis,
                           std::vector<double>& filter,
                           std::string& error);

// Exact derivative of the filtered, discrete sqrt-CG reaction energy through
// the electronic-density cavity. Failure leaves response.cavity_potential unchanged.
bool evaluate_lowpass_cavity_potential(const std::vector<double>& charge,
                                      const CavityParameters& cavity,
                                      const CavityDerivatives& derivatives,
                                      const ModulePW::PW_Basis& basis,
                                      double tpiba,
                                      SccsResponse& response,
                                      std::string& error);
} // namespace ModuleSccs

#endif
