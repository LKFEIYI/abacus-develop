#ifndef SCCS_CAVITY_DERIVATIVES_H
#define SCCS_CAVITY_DERIVATIVES_H

#include <string>
#include <vector>

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSccs
{
struct CavityParameters;
struct SccsResponse;

// Collective over the PW pool. Populate the cavity fields and the periodic
// chain-derivative sqrt-CG coefficient; caller owns the candidate response.
bool prepare_cavity_derivatives(const std::vector<double>& density,
                                const CavityParameters& cavity,
                                const ModulePW::PW_Basis& basis,
                                double tpiba,
                                SccsResponse& response,
                                std::vector<double>& coefficient,
                                std::string& error);
} // namespace ModuleSccs

#endif
