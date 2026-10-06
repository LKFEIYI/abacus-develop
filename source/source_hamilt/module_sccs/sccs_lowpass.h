#ifndef SCCS_LOWPASS_H
#define SCCS_LOWPASS_H

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

// Reciprocal-space filter on the local G vectors; empty when disabled.
std::vector<double> make_switching_filter(const CavityParameters& cavity, const ModulePW::PW_Basis& basis);

// Exact derivative of the filtered, discrete sqrt-CG reaction energy through
// the electronic-density cavity; collective over the PW pool.
void evaluate_lowpass_cavity_potential(const std::vector<double>& charge,
                                       const CavityParameters& cavity,
                                       const CavityDerivatives& derivatives,
                                       const ModulePW::PW_Basis& basis,
                                       double tpiba,
                                       SccsResponse& response);
} // namespace ModuleSccs

#endif
