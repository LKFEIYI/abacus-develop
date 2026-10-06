#ifndef SCCS_PCC_2D_ADAPTER_H
#define SCCS_PCC_2D_ADAPTER_H

#include <memory>
#include <string>
#include <vector>

class UnitCell;
namespace ModuleBase
{
template <typename T> class Vector3;
}
namespace ModulePW
{
class PW_Basis;
}
namespace ModuleSccs
{
class CoulombOperator;
struct SccsResponse;
struct SccsConfig;
struct PolarizationSolverParameters;
}
namespace unitcell
{
struct AtomData;
}
namespace elecstate
{
// Assemble an SCCS grid-charge operator using the existing cell geometry and
// mass-weighted ionic origin. Failure leaves the output unchanged.
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error);

// Use caller-owned Cartesian grid coordinates, e.g. from a fixed-source cache.
bool make_sccs_pcc_2d_operator(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              int open_axis,
                              const std::vector<ModuleBase::Vector3<double>>& grid_positions,
                              std::unique_ptr<ModuleSccs::CoulombOperator>& coulomb,
                              std::string& error);
// Diagnose slab far-field charge consistency using the original tolerance.
// Finite deviations populate warning; non-finite values fail with error.
// Counts are positive electron/ionic charge integrals; INPUT is not read.
bool validate_sccs_pcc_2d_screening(const ModuleSccs::SccsResponse& response,
                                   const ModuleSccs::SccsConfig& config,
                                   const ModuleSccs::PolarizationSolverParameters& solver,
                                   double electron_count,
                                   double ionic_charge,
                                   double cell_volume,
                                   std::string& warning,
                                   std::string& error);
} // namespace elecstate

#endif
