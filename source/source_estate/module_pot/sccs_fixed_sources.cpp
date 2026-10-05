#include "sccs_fixed_sources.h"
#include "source_base/parallel_reduce.h"
#include "source_base/timer.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_basis/module_pw/pw_grid_geometry.h"
#include "source_cell/cell_tools.h"
#include "source_cell/unitcell.h"
#include "source_hamilt/module_sccs/sccs_ionic_charge.h"
#include "source_hamilt/module_sccs/sccs_parameters.h"
#include "source_hamilt/module_sccs/sccs_pw_coulomb.h"
#include <array>

namespace
{
struct SourceKey
{
    const ModulePW::PW_Basis* basis = nullptr;
    std::array<int, 15> grid;
    std::array<double, 14> cell;
    bool core_electrons = false;
    std::vector<double> widths;
};
SourceKey source_key(const UnitCell& cell, const ModulePW::PW_Basis& basis,
                     const ModuleSccs::SccsConfig& config)
{
    SourceKey key;
    key.basis = &basis;
    key.grid = {{basis.nx, basis.ny, basis.nz, basis.nrxx, basis.nxyz,
                 basis.nplane, basis.startz_current, basis.npw, basis.npwtot,
                 basis.nst, basis.nstot, basis.poolnproc, basis.poolrank,
                 basis.nmaxgr, basis.ig_gge0}};
    key.cell = {{cell.lat0, cell.tpiba, cell.omega, basis.omega, basis.ggecut,
                 cell.latvec.e11, cell.latvec.e12, cell.latvec.e13,
                 cell.latvec.e21, cell.latvec.e22, cell.latvec.e23,
                 cell.latvec.e31, cell.latvec.e32, cell.latvec.e33}};
    key.core_electrons = config.core_electrons;
    if (config.core_electrons) { key.widths = config.core_spreads; }
    return key;
}
bool same_vector(const ModuleBase::Vector3<double>& left, const ModuleBase::Vector3<double>& right)
{
    return left.x == right.x && left.y == right.y && left.z == right.z;
}
bool same_atoms(const std::vector<unitcell::AtomData>& left, const std::vector<unitcell::AtomData>& right)
{
    if (left.size() != right.size()) { return false; }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        if (!same_vector(left[i].position, right[i].position)
            || left[i].mass != right[i].mass || left[i].valence_charge != right[i].valence_charge
            || left[i].atomic_number != right[i].atomic_number) { return false; }
    }
    return true;
}
}
namespace elecstate
{
struct SccsFixedSources::Data
{
    bool valid = false;
    SourceKey key;
    std::vector<unitcell::AtomData> atoms;
    std::vector<double> gg;
    std::vector<ModuleBase::Vector3<double>> gcar;
    std::vector<int> reciprocal_map;
    std::vector<int> stick_map;
    std::vector<double> ionic_density;
    std::vector<double> core_density;
    std::vector<ModuleBase::Vector3<double>> positions;

    bool matches(const SourceKey& other, const std::vector<unitcell::AtomData>& other_atoms,
                 const ModulePW::PW_Basis& basis) const
    {
        if (!valid || key.basis != other.basis || key.grid != other.grid || key.cell != other.cell
            || key.core_electrons != other.core_electrons || key.widths != other.widths
            || !same_atoms(atoms, other_atoms)) { return false; }
        for (int i = 0; i < basis.npw; ++i)
        {
            if (gg[i] != basis.gg[i] || !same_vector(gcar[i], basis.gcar[i])
                || reciprocal_map[i] != basis.ig2isz[i]) { return false; }
        }
        for (int i = 0; i < basis.nst; ++i)
        {
            if (stick_map[i] != basis.is2fftixy[i]) { return false; }
        }
        return true;
    }
};
SccsFixedSources::SccsFixedSources()
{
    data_.reset(new Data);
}
SccsFixedSources::~SccsFixedSources() = default;

bool SccsFixedSources::update(const UnitCell& cell,
                              const ModulePW::PW_Basis& basis,
                              const std::vector<unitcell::AtomData>& atoms,
                              const ModuleSccs::SccsConfig& config,
                              bool& reused,
                              std::string& error)
{
    ModuleBase::timer::start("SccsFixedSources", "update");
    if (!ModuleSccs::validate_pw_grid(basis, cell.tpiba, error))
    {
        ModuleBase::timer::end("SccsFixedSources", "update");
        return false;
    }
    const bool maps_valid = basis.nst >= 0 && (basis.npw == 0 || basis.ig2isz != nullptr)
                            && (basis.nst == 0 || basis.is2fftixy != nullptr);
    double invalid = maps_valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS fixed sources require initialized local PW distribution maps";
        ModuleBase::timer::end("SccsFixedSources", "update");
        return false;
    }
    const SourceKey key = source_key(cell, basis, config);
    const bool local_reuse = data_->matches(key, atoms, basis);
    double rebuild = local_reuse ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, rebuild);
    if (rebuild == 0.0)
    {
        reused = true;
        ModuleBase::timer::end("SccsFixedSources", "update");
        return true;
    }
    std::unique_ptr<Data> candidate;
    candidate.reset(new Data);
    const bool positions_valid = ModulePW::grid_positions(basis, cell.latvec, cell.lat0, candidate->positions, error);
    invalid = positions_valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        if (error.empty()) { error = "SCCS fixed-source grid invalid on another pool rank"; }
        ModuleBase::timer::end("SccsFixedSources", "update");
        return false;
    }
    bool valid = ModuleSccs::gaussian_ionic_density(atoms, basis, cell.tpiba,
                                                   ModuleSccs::gaussian_ion_spread, candidate->ionic_density, error);
    if (valid && config.core_electrons)
    {
        valid = ModuleSccs::gaussian_core_density(atoms, basis, cell.tpiba, config.core_spreads,
                                                 candidate->core_density, error);
    }
    if (!valid)
    {
        ModuleBase::timer::end("SccsFixedSources", "update");
        return false;
    }
    candidate->key = key;
    candidate->atoms = atoms;
    for (int i = 0; i < basis.npw; ++i)
    {
        candidate->gg.push_back(basis.gg[i]);
        candidate->gcar.push_back(basis.gcar[i]);
        candidate->reciprocal_map.push_back(basis.ig2isz[i]);
    }
    for (int i = 0; i < basis.nst; ++i) { candidate->stick_map.push_back(basis.is2fftixy[i]); }
    candidate->valid = true;
    data_.swap(candidate);
    reused = false;
    ModuleBase::timer::end("SccsFixedSources", "update");
    return true;
}
const std::vector<double>& SccsFixedSources::ionic_density() const { return data_->ionic_density; }
const std::vector<double>& SccsFixedSources::core_density() const { return data_->core_density; }
const std::vector<ModuleBase::Vector3<double>>& SccsFixedSources::positions() const { return data_->positions; }
} // namespace elecstate
