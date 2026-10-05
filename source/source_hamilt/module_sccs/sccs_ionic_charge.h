#ifndef SCCS_IONIC_CHARGE_H
#define SCCS_IONIC_CHARGE_H

#include <string>
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
// Collective over the PW pool. Failure leaves density unchanged.
bool gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            double spread,
                            std::vector<double>& density,
                            std::string& error);

// One finite positive width per atom; scalar interface remains available.
bool gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            const std::vector<double>& spreads,
                            std::vector<double>& density,
                            std::string& error);

// Resolve full-cavity widths without changing the physical ionic charge.
bool prepare_core_gaussians(const std::vector<unitcell::AtomData>& atoms,
                            const std::vector<double>& atom_spreads,
                            std::vector<unitcell::AtomData>& core_atoms,
                            std::vector<double>& widths,
                            std::string& error);

// Full cavity: a singleton skips known Z == zv atoms; per-atom lists override
// that choice. Nonpositive widths always disable the atom.
bool gaussian_core_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            const std::vector<double>& atom_spreads,
                            std::vector<double>& density,
                            std::string& error);
} // namespace ModuleSccs

#endif
