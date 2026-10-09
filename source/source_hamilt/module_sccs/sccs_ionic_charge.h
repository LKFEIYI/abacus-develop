#ifndef SCCS_IONIC_CHARGE_H
#define SCCS_IONIC_CHARGE_H

#include <vector>

namespace ModulePW
{
class PW_Basis;
}
namespace unitcell
{
struct AtomData;
}

namespace ModuleSccs
{
// rho ~ exp(-r^2/spread^2), in Bohr; preserves the original SCCS ionic spread.
const double gaussian_ion_spread = 0.5;

// AtomData is extracted by source_cell; positions are Cartesian Bohr.
// Collective over the PW pool.
void gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            double spread,
                            std::vector<double>& density);

// One positive width per atom; scalar interface remains available.
void gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            const std::vector<double>& spreads,
                            std::vector<double>& density);

// Resolve full-cavity widths without changing the physical ionic charge.
// atom_spreads holds one value or one per atom (checked when SCCS is set up).
void prepare_core_gaussians(const std::vector<unitcell::AtomData>& atoms,
                            const std::vector<double>& atom_spreads,
                            std::vector<unitcell::AtomData>& core_atoms,
                            std::vector<double>& widths);

// Full cavity: a singleton skips known Z == zv atoms; per-atom lists override
// that choice. Nonpositive widths always disable the atom.
void gaussian_core_density(const std::vector<unitcell::AtomData>& atoms,
                           const ModulePW::PW_Basis& basis,
                           double tpiba,
                           const std::vector<double>& atom_spreads,
                           std::vector<double>& density);
} // namespace ModuleSccs

#endif
