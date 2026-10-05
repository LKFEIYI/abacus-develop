#ifndef SCCS_FIXED_SOURCES_H
#define SCCS_FIXED_SOURCES_H

#include <memory>
#include <string>
#include <vector>

class UnitCell;
namespace ModuleBase { template <typename T> class Vector3; }
namespace ModulePW { class PW_Basis; }
namespace unitcell { struct AtomData; }
namespace ModuleSccs { struct SccsConfig; }
namespace elecstate
{
// Owns ion/core Gaussian densities and Cartesian grid coordinates for one cell.
// update is the sole cache writer. It decides reuse collectively over the PW
// pool; failure preserves the previous cache and the caller's reuse flag.
class SccsFixedSources
{
public:
    SccsFixedSources();
    ~SccsFixedSources();
    bool update(const UnitCell& cell,
                 const ModulePW::PW_Basis& basis,
                 const std::vector<unitcell::AtomData>& atoms,
                 const ModuleSccs::SccsConfig& config,
                 bool& reused,
                 std::string& error);
    // Empty until the first successful update; views live until the next update.
    const std::vector<double>& ionic_density() const;
    const std::vector<double>& core_density() const;
    const std::vector<ModuleBase::Vector3<double>>& positions() const;
private:
    struct Data;
    std::unique_ptr<Data> data_;
};
} // namespace elecstate
#endif
