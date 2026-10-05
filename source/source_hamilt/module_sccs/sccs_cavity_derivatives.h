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

// Collective over the PW pool. Periodic cells use density chain derivatives;
// open boundaries differentiate solute on the FFT grid. Caller owns the candidate.
bool prepare_cavity_derivatives(const std::vector<double>& density,
                                const CavityParameters& cavity,
                                const ModulePW::PW_Basis& basis,
                                double tpiba,
                                bool open_boundary,
                                SccsResponse& response,
                                std::vector<double>& coefficient,
                                std::string& error);
} // namespace ModuleSccs

#endif
