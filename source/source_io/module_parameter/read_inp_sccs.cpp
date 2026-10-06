#include "read_inp_sccs.h"
#include "read_input.h"
#include "read_input_tool.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cctype>
#include <cmath>

namespace ModuleIO
{
bool parse_solvation_model(const std::string& value, int& model, std::string& error)
{
    error.clear();
    std::string normalized = value;
    for (char& c : normalized)
    {
        const unsigned char byte = static_cast<unsigned char>(c);
        const int lower = std::tolower(byte);
        c = static_cast<char>(lower);
    }
    const std::vector<std::string> yes = {"true", "1", "t", "yes", "y", "on", ".true."};
    const std::vector<std::string> no = {"false", "0", "f", "no", "n", "off", ".false."};
    if (normalized == "2") { model = 2; }
    else if (std::find(yes.begin(), yes.end(), normalized) != yes.end()) { model = 1; }
    else if (std::find(no.begin(), no.end(), normalized) != no.end()) { model = 0; }
    else
    {
        error = "imp_sol must be 0, 1, 2 or a legacy Boolean value";
        return false;
    }
    return true;
}

bool validate_sccs_input(const Input_para& input, std::string& error)
{
    error.clear();
    if (input.imp_sol < 0 || input.imp_sol > 2)
    {
        error = "imp_sol must be 0, 1 or 2";
        return false;
    }
    if (input.imp_sol != 2) { return true; }
    if (input.assume_isolated != "none" && input.assume_isolated != "pcc_0d"
        && input.assume_isolated != "pcc_2d")
    { error = "SCCS currently supports assume_isolated none, pcc_0d or pcc_2d"; }
    else if (input.device != "cpu" || input.esolver_type != "ksdft") { error = "SCCS requires CPU KS-DFT"; }
    else if (input.basis_type != "pw" && input.basis_type != "lcao") { error = "SCCS requires basis_type pw or lcao"; }
    else if ((input.calculation != "scf" && input.calculation != "relax") || input.cal_stress)
    { error = "SCCS supports calculation scf or fixed-cell relax without stress"; }
    else if (input.nspin != 1 && input.nspin != 2) { error = "SCCS requires nspin 1 or 2"; }
    else if (input.efield_flag || input.gate_flag || input.dfthalf_type != 0
             || input.deepks_scf || input.deepks_out_labels || input.deepks_bandgap
             || input.deepks_v_delta || input.dm_to_rho)
    { error = "SCCS does not support electric/gate fields, DFT-1/2, DeePKS or dm_to_rho"; }
    else if (!input.vl_in_h || !input.vion_in_h || !input.vh_in_h)
    { error = "SCCS requires the local ionic and Hartree potentials in the Hamiltonian"; }
    if (!error.empty()) { return false; }
    const std::vector<std::string> presets = {"custom", "vacuum", "water-neutral", "water-cation", "water-anion"};
    if (std::find(presets.begin(), presets.end(), input.sccs_preset) == presets.end())
    {
        error = "Unknown sccs_preset";
        return false;
    }
    if (input.sccs_maxiter <= 0 || !std::isfinite(input.sccs_epsilon) || input.sccs_epsilon < 1.0
        || !std::isfinite(input.sccs_rho_min) || !std::isfinite(input.sccs_rho_max)
        || input.sccs_rho_min <= 0.0 || input.sccs_rho_max <= input.sccs_rho_min
        || !std::isfinite(input.sccs_gamma) || !std::isfinite(input.sccs_pressure)
        || !std::isfinite(input.sccs_tol_rms) || input.sccs_tol_rms <= 0.0
        || !std::isfinite(input.sccs_tol_max) || input.sccs_tol_max <= 0.0
        || !std::isfinite(input.sccs_surface_eta) || input.sccs_surface_eta <= 0.0)
    {
        error = "Invalid SCCS numerical parameters";
        return false;
    }
    const bool p1_positive = input.sccs_lowpass_p1 > 0.0;
    const bool p2_positive = input.sccs_lowpass_p2 > 0.0;
    if (!std::isfinite(input.sccs_lowpass_p1) || !std::isfinite(input.sccs_lowpass_p2)
        || p1_positive != p2_positive)
    {
        error = "sccs_lowpass_p1 and sccs_lowpass_p2 must be finite and both positive or both non-positive";
        return false;
    }
    if (p1_positive && input.assume_isolated != "pcc_0d" && input.assume_isolated != "pcc_2d")
    {
        error = "sccs_lowpass_p1 and sccs_lowpass_p2 require assume_isolated pcc_0d or pcc_2d";
        return false;
    }
    if (input.sccs_solvent_mode != "electronic" && input.sccs_solvent_mode != "full")
    {
        error = "sccs_solvent_mode must be electronic or full";
        return false;
    }
    if (input.sccs_corespread.empty())
    {
        error = "sccs_corespread must contain one value or exactly nat values";
        return false;
    }
    for (double spread : input.sccs_corespread)
    {
        if (!std::isfinite(spread))
        {
            error = "sccs_corespread values must be finite";
            return false;
        }
    }
    if (!std::isfinite(input.sccs_start_drho) || input.sccs_start_drho < 0.0)
    {
        error = "sccs_start_drho must be finite and non-negative";
        return false;
    }
    if (input.sccs_start_drho > 0.0 && input.sccs_start_drho <= input.scf_thr)
    {
        error = "sccs_start_drho must be zero or larger than scf_thr";
        return false;
    }
    if (input.sccs_start_nmax <= 0
        || (input.sccs_start_drho > 0.0 && input.sccs_start_nmax >= input.scf_nmax))
    {
        error = "sccs_start_nmax must be positive and smaller than scf_nmax when delayed start is enabled";
        return false;
    }
    return true;
}

void ReadInput::item_sccs()
{
    {
        Input_Item item("imp_sol");
        item.annotation = "implicit solvent model";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "Select 0 for vacuum, 1 for the original ABACUS implicit solvation model, or 2 for SCCS. Legacy Boolean values remain accepted as 0 or 1. SCCS supports CPU KS-DFT SCF and fixed-cell relax calculations with basis_type pw or lcao and nspin 1 or 2: neutral periodic cells (assume_isolated none), neutral/charged molecules in cubic cells (assume_isolated pcc_0d), or neutral/charged slabs (assume_isolated pcc_2d). Forces are supported for periodic SCCS and SCCS with pcc_0d or pcc_2d. Stress, external fields and other correction models are not supported.";
        item.default_value = "0";
        item.read_value = [](const Input_Item& item, Parameter& para) {
            std::string error;
            const bool valid = parse_solvation_model(item.str_values[0], para.input.imp_sol, error);
            if (!valid) { ModuleBase::WARNING_QUIT("ReadInput", error); }
        };
        sync_int(input.imp_sol);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            std::string error;
            const bool valid = validate_sccs_input(para.input, error);
            if (!valid) { ModuleBase::WARNING_QUIT("ReadInput", error); }
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccs_preset");
        item.annotation = "SCCS parameter preset";
        item.category = "Implicit solvation model";
        item.type = "String";
        item.description = "SCCS parameter preset. Allowed values: custom, vacuum, water-neutral, water-cation, water-anion. Non-custom presets override sccs_epsilon, sccs_rho_min, sccs_rho_max, sccs_gamma and sccs_pressure; solver controls and surface regularization remain user-controlled.";
        item.default_value = "custom";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_string(input.sccs_preset);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_epsilon");
        item.annotation = "Bulk dielectric constant >= 1 for the custom preset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Bulk dielectric constant >= 1 for the custom preset. Water presets use 78.3 and vacuum uses 1.";
        item.default_value = "78.3";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_epsilon);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_rho_min");
        item.annotation = "Lower electronic-density cavity threshold for the custom preset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Lower electronic-density cavity threshold for the custom preset. Water-neutral/vacuum: 1e-4; water-cation: 2e-4; water-anion: 2.4e-3.";
        item.default_value = "1e-4";
        item.unit = "e/bohr^3";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_rho_min);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_rho_max");
        item.annotation = "Upper electronic-density cavity threshold for the custom preset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Upper electronic-density cavity threshold for the custom preset. Water-neutral/vacuum: 5e-3; water-cation: 3.5e-3; water-anion: 1.55e-2.";
        item.default_value = "5e-3";
        item.unit = "e/bohr^3";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_rho_max);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_gamma");
        item.annotation = "Surface coefficient for the custom preset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Surface coefficient for the custom preset. Water-neutral: 47.9; water-cation: 5; water-anion/vacuum: 0.";
        item.default_value = "0.0";
        item.unit = "dyn/cm";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_gamma);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_pressure");
        item.annotation = "Volume coefficient for the custom preset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Volume coefficient for the custom preset. Water-neutral: -0.36; water-cation: 0.125; water-anion: 0.45; vacuum: 0.";
        item.default_value = "0.0";
        item.unit = "GPa";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_pressure);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_maxiter");
        item.annotation = "Positive maximum number of SCCS sqrt-CG iterations";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "Positive maximum number of SCCS sqrt-CG iterations. Failure to satisfy both residual tolerances terminates the calculation.";
        item.default_value = "200";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_int(input.sccs_maxiter);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_tol_rms");
        item.annotation = "Positive RMS charge-residual tolerance for the SCCS sqrt-CG solver";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Positive RMS charge-residual tolerance for the SCCS sqrt-CG solver.";
        item.default_value = "1e-10";
        item.unit = "e/bohr^3";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_tol_rms);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_tol_max");
        item.annotation = "Positive maximum charge-residual tolerance for the SCCS sqrt-CG solver";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Positive maximum charge-residual tolerance for the SCCS sqrt-CG solver.";
        item.default_value = "1e-8";
        item.unit = "e/bohr^3";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_tol_max);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_surface_eta");
        item.annotation = "Positive regularization of the SCCS surface gradient norm";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Positive regularization of the SCCS surface gradient norm.";
        item.default_value = "1e-8";
        item.unit = "bohr^-1";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_surface_eta);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_lowpass_p1");
        item.annotation = "SCCS switching-derivative low-pass slope";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Low-pass filter of the SCCS switching-function derivatives, as Environ deriv_lowpass_p1 with deriv_method fft: when sccs_lowpass_p1 and sccs_lowpass_p2 are both positive, every Fourier derivative of the switching function is multiplied by 0.5 erfc(p1 G^2/Gcut^2 - p2), Gcut^2 being the ecutrho sphere. The filter reduces force errors relative to energy finite differences. Only with assume_isolated pcc_0d or pcc_2d. The default -1 turns it off and reproduces Environ deriv_method fft (continuum cavity potential). With lowpass disabled (the default), analytical forces may differ from finite differences of the self-consistent energy. For geometry optimization with PCC, consider enabling lowpass and check force accuracy against finite differences. With lowpass disabled, the cavity potential uses the FFT gradient of the PCC-corrected potential, which oscillates around the potential step at the cell boundary half a cell from the system center; keep the dielectric transition region several bohr away from that boundary. 10 with sccs_lowpass_p2 5 was validated at ecutrho 300-500 Ry; the filter changes the model energy (about 10 meV for H3O+).";
        item.default_value = "-1";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_lowpass_p1);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_lowpass_p2");
        item.annotation = "SCCS switching-derivative low-pass offset";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Offset of the SCCS switching-function low-pass filter, as Environ deriv_lowpass_p2; see sccs_lowpass_p1. Both must be positive or both non-positive. Default -1 (off).";
        item.default_value = "-1";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_lowpass_p2);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_solvent_mode");
        item.annotation = "density that defines the SCCS cavity";
        item.category = "Implicit solvation model";
        item.type = "String";
        item.description = "Allowed values: electronic (default) and full, as Environ solvent_mode. electronic builds the dielectric cavity from the valence electron density. full adds valence-charge Gaussians selected by sccs_corespread. The Gaussians shape only the cavity, not the solute charge; the ionic forces include their cavity term. The published SCCS presets were fitted with electronic mode.";
        item.default_value = "electronic";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_string(input.sccs_solvent_mode);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_corespread");
        item.annotation = "Widths of the full-cavity Gaussians";
        item.category = "Implicit solvation model";
        item.type = "Vector of Real";
        item.description = "Widths of the valence-charge Gaussians used only with sccs_solvent_mode full. One value applies to every atom except those whose known atomic number equals the pseudopotential valence charge (within 1e-8); if the pseudopotential element is unknown, the value is applied. More than one value requires exactly nat values in STRU atom order (grouped by atom type), and overrides this automatic exclusion. A value <= 0 disables the cavity Gaussian on that atom. Values must be finite. These Gaussians do not change the solute charge.";
        item.default_value = "0.5";
        item.unit = "bohr";
        item.set_availability("imp_sol==2");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            std::vector<double> values;
            for (const std::string& value : item.str_values) { values.push_back(std::stod(value)); }
            para.input.sccs_corespread.swap(values);
        };
        sync_doublevec(input.sccs_corespread, para.input.sccs_corespread.size(), 0.0);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_start_drho");
        item.annotation = "SCCS delayed-start density threshold";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Delay SCCS at the start of the run until DRHO is at or below this value. Zero starts SCCS immediately; a positive value must exceed scf_thr so that the SCF cannot converge before SCCS starts, and the SCF does not stop in the iteration that activates SCCS. Once activated, SCCS remains active for all later electronic and ionic steps. PCC remains active during the delay. User-controlled for every sccs_preset, default 0.";
        item.default_value = "0.0";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_start_drho);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_start_nmax");
        item.annotation = "SCCS delayed-start iteration limit";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "Force delayed SCCS activation at this electronic iteration if the SCCS start DRHO threshold has not yet been reached. The value must be positive, and smaller than scf_nmax when delayed start is enabled. User-controlled for every sccs_preset, default 30; inactive when sccs_start_drho=0.";
        item.default_value = "30";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_int(input.sccs_start_nmax);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_debug");
        item.annotation = "detailed SCCS diagnostics";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "SCCS/PCC output level: 0 suppresses per-SCF summaries and diagnostics; 1 prints the iteration count and correction energy; 2 additionally prints residual, warm-start, cavity volume and surface, FFT-count, Gauss-law (PCC), multipole and energy diagnostics, and verifies the sqrt-CG fixed point with one extra Poisson solve per SCCS evaluation. Applies to standalone PCC as well as SCCS. Timings appear in the standard ABACUS timer summary.";
        item.default_value = "0";
        item.unit = "";
        read_sync_int(input.sccs_debug);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            if (para.input.sccs_debug < 0 || para.input.sccs_debug > 2)
            {
                ModuleBase::WARNING_QUIT("ReadInput", "sccs_debug must be 0, 1, or 2");
            }
        };
        this->add_item(item);
    }

}
} // namespace ModuleIO
