#include "sccs_ionic_charge.h"
#include "sccs_pw_coulomb.h"

#include "source_base/constants.h"
#include "source_base/parallel_reduce.h"
#include "source_basis/module_pw/pw_basis.h"
#include "source_cell/cell_tools.h"

#include <cmath>
#include <complex>

namespace ModuleSccs
{
bool gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            double spread,
                            std::vector<double>& density,
                            std::string& error)
{
    double invalid = (!std::isfinite(spread) || spread <= 0.0) ? 1.0 : 0.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS Gaussian width must be finite and positive";
        return false;
    }
    const std::vector<double> spreads(atoms.size(), spread);
    return gaussian_ionic_density(atoms, basis, tpiba, spreads, density, error);
}

bool gaussian_ionic_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            const std::vector<double>& spreads,
                            std::vector<double>& density,
                            std::string& error)
{
    if (!validate_pw_grid(basis, tpiba, error))
    {
        return false;
    }
    double invalid = 0.0;
    if (spreads.size() != atoms.size())
    {
        invalid = 1.0;
    }
    for (const unitcell::AtomData& atom : atoms)
    {
        if (!std::isfinite(atom.valence_charge) || atom.valence_charge < 0.0
            || !std::isfinite(atom.position.x) || !std::isfinite(atom.position.y)
            || !std::isfinite(atom.position.z))
        {
            invalid = 1.0;
        }
    }
    for (double width : spreads)
    {
        if (!std::isfinite(width) || width <= 0.0) { invalid = 1.0; }
    }
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS Gaussian ions require finite positions, nonnegative valence charges and positive finite spread";
        return false;
    }

    std::vector<std::complex<double>> charge(basis.npw, 0.0);
    const double tpiba2 = tpiba * tpiba;
    for (std::size_t ia = 0; ia < atoms.size(); ++ia)
    {
        const unitcell::AtomData& atom = atoms[ia];
        const double spread = spreads[ia];
        for (int ig = 0; ig < basis.npw; ++ig)
        {
            const double exponent = -0.25 * spread * spread * tpiba2 * basis.gg[ig];
            const double phase = tpiba * (basis.gcar[ig] * atom.position);
            const std::complex<double> phase_factor = ModuleBase::NEG_IMAG_UNIT * phase;
            charge[ig] += atom.valence_charge / basis.omega * std::exp(exponent) * std::exp(phase_factor);
        }
    }
    std::vector<double> candidate(basis.nrxx);
    basis.recip2real(charge.data(), candidate.data());
    if (!validate_grid_values(candidate, basis, error))
    {
        return false;
    }
    density.swap(candidate);
    return true;
}
bool prepare_core_gaussians(const std::vector<unitcell::AtomData>& atoms,
                            const std::vector<double>& atom_spreads,
                            std::vector<unitcell::AtomData>& core_atoms,
                            std::vector<double>& widths,
                            std::string& error)
{
    if (atom_spreads.empty() || (atom_spreads.size() != 1 && atom_spreads.size() != atoms.size()))
    {
        error = "SCCS core widths require either one value or one per atom";
        return false;
    }
    for (double width : atom_spreads)
    {
        if (!std::isfinite(width))
        {
            error = "SCCS per-atom core widths must be finite";
            return false;
        }
    }
    std::vector<unitcell::AtomData> selected = atoms;
    std::vector<double> selected_widths(atoms.size(), gaussian_ion_spread);
    for (std::size_t ia = 0; ia < atoms.size(); ++ia)
    {
        const std::size_t index = atom_spreads.size() == 1 ? 0 : ia;
        const double width = atom_spreads[index];
        const double charge_difference = atoms[ia].atomic_number - atoms[ia].valence_charge;
        const bool no_core = atoms[ia].atomic_number > 0 && std::abs(charge_difference) < 1e-8;
        const bool automatic_skip = atom_spreads.size() == 1 && no_core;
        if (width <= 0.0 || automatic_skip) { selected[ia].valence_charge = 0.0; }
        else { selected_widths[ia] = width; }
    }
    core_atoms.swap(selected);
    widths.swap(selected_widths);
    return true;
}

bool gaussian_core_density(const std::vector<unitcell::AtomData>& atoms,
                            const ModulePW::PW_Basis& basis,
                            double tpiba,
                            const std::vector<double>& atom_spreads,
                            std::vector<double>& density,
                            std::string& error)
{
    std::vector<unitcell::AtomData> core_atoms;
    std::vector<double> widths;
    const bool valid = prepare_core_gaussians(atoms, atom_spreads, core_atoms, widths, error);
    double invalid = valid ? 0.0 : 1.0;
    Parallel_Reduce::reduce_max_pool(basis.poolnproc, invalid);
    if (invalid != 0.0)
    {
        error = "SCCS full-cavity widths are invalid on a pool rank";
        return false;
    }
    return gaussian_ionic_density(core_atoms, basis, tpiba, widths, density, error);
}
} // namespace ModuleSccs
