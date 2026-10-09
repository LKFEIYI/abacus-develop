#include "read_inp_sccs.h"
#include "read_input.h"
#include "read_input_tool.h"
#include "source_base/tool_quit.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

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

bool parse_core_spreads(const std::vector<std::string>& words, std::vector<double>& spreads, std::string& error)
{
    error.clear();
    std::vector<double> values;
    for (const std::string& word : words)
    {
        const char* begin = word.c_str();
        char* end = nullptr;
        const double value = std::strtod(begin, &end);
        const bool complete = end != begin && *end == '\0';
        if (!complete || !std::isfinite(value))
        {
            error = "sccs_corespread values must be finite real numbers, got '" + word + "'";
            return false;
        }
        values.push_back(value);
    }
    spreads.swap(values);
    return true;
}

namespace
{
// Calculation settings that SCCS supports; empty when all are supported.
std::string sccs_context_error(const Input_para& input)
{
    std::string error;
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
    return error;
}

// Cavity, surface and boundary model parameters, starting with the preset.
std::string sccs_model_error(const Input_para& input)
{
    const std::vector<std::string> presets = {"custom", "vacuum", "water-neutral", "water-cation", "water-anion"};
    if (std::find(presets.begin(), presets.end(), input.sccs_preset) == presets.end())
    {
        return "Unknown sccs_preset";
    }
    const bool p1_positive = input.sccs_lowpass_p1 > 0.0;
    const bool p2_positive = input.sccs_lowpass_p2 > 0.0;
    const bool pcc = input.assume_isolated == "pcc_0d" || input.assume_isolated == "pcc_2d";
    std::string error;
    if (input.sccs_epsilon < 1.0) { error = "sccs_epsilon must be at least 1"; }
    else if (input.sccs_rho_min <= 0.0 || input.sccs_rho_max <= input.sccs_rho_min)
    { error = "sccs_rho_min and sccs_rho_max must satisfy 0 < sccs_rho_min < sccs_rho_max"; }
    else if (input.sccs_surface_eta <= 0.0) { error = "sccs_surface_eta must be positive"; }
    else if (p1_positive != p2_positive)
    { error = "sccs_lowpass_p1 and sccs_lowpass_p2 must be both positive or both non-positive"; }
    else if (p1_positive && !pcc)
    { error = "sccs_lowpass_p1 and sccs_lowpass_p2 require assume_isolated pcc_0d or pcc_2d"; }
    else if (input.sccs_solvent_mode != "electronic" && input.sccs_solvent_mode != "full")
    { error = "sccs_solvent_mode must be electronic or full"; }
    else if (input.sccs_corespread.empty())
    { error = "sccs_corespread must contain one value or exactly nat values"; }
    return error;
}

// Solvent-aware probe and filling; a zero radius turns the filling off.
std::string sccs_solvent_aware_error(const Input_para& input)
{
    std::string error;
    if (input.sccs_solvent_radius < 0.0) { error = "sccs_solvent_radius must be non-negative"; }
    else if (input.sccs_radial_scale < 1.0) { error = "sccs_radial_scale must be at least 1"; }
    else if (input.sccs_radial_spread <= 0.0) { error = "sccs_radial_spread must be positive"; }
    // The probe solute fraction never exceeds one, so a threshold of one or
    // more would fill nothing.
    else if (input.sccs_filling_threshold <= 0.0 || input.sccs_filling_threshold >= 1.0)
    { error = "sccs_filling_threshold must lie between 0 and 1"; }
    else if (input.sccs_filling_spread <= 0.0) { error = "sccs_filling_spread must be positive"; }
    return error;
}

// sqrt-CG solver controls and the delayed start.
std::string sccs_solver_error(const Input_para& input)
{
    const bool delayed_start = input.sccs_start_drho > 0.0;
    std::string error;
    if (input.sccs_maxiter <= 0) { error = "sccs_maxiter must be positive"; }
    else if (input.sccs_tol_rms <= 0.0) { error = "sccs_tol_rms must be positive"; }
    else if (input.sccs_tol_max <= 0.0) { error = "sccs_tol_max must be positive"; }
    else if (input.sccs_start_drho < 0.0) { error = "sccs_start_drho must be non-negative"; }
    else if (delayed_start && input.sccs_start_drho <= input.scf_thr)
    { error = "sccs_start_drho must be zero or larger than scf_thr"; }
    else if (input.sccs_start_nmax <= 0 || (delayed_start && input.sccs_start_nmax >= input.scf_nmax))
    { error = "sccs_start_nmax must be positive and smaller than scf_nmax when delayed start is enabled"; }
    return error;
}
} // namespace

bool validate_sccs_input(const Input_para& input, std::string& error)
{
    error.clear();
    if (input.imp_sol < 0 || input.imp_sol > 2)
    {
        error = "imp_sol must be 0, 1 or 2";
        return false;
    }
    if (input.imp_sol != 2) { return true; }
    error = sccs_context_error(input);
    if (error.empty()) { error = sccs_model_error(input); }
    if (error.empty()) { error = sccs_solvent_aware_error(input); }
    if (error.empty()) { error = sccs_solver_error(input); }
    return error.empty();
}

void ReadInput::item_sccs()
{
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
        item.description = "Bulk dielectric constant >= 1 for the custom preset. The default 1 (no dielectric) follows the Environ env_static_permittivity default for environ_type input. Water presets use 78.3 and vacuum uses 1.";
        item.default_value = "1.0";
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
        item.description = "Low-pass filter of the SCCS switching-function derivatives, as Environ deriv_lowpass_p1 with deriv_method fft. Only with assume_isolated pcc_0d or pcc_2d. The default -1 turns it off and reproduces Environ deriv_method fft (continuum cavity potential). With lowpass disabled (the default), analytical forces may differ from finite differences of the self-consistent energy. For geometry optimization with PCC, consider enabling lowpass and check force accuracy against finite differences. With lowpass disabled, the cavity potential uses the FFT gradient of the PCC-corrected potential, which oscillates around the potential step at the cell boundary half a cell from the system center; keep the dielectric transition region several bohr away from that boundary. The values sccs_lowpass_p1 10 and sccs_lowpass_p2 5 were validated at ecutrho 300-500 Ry; the filter changes the model energy (about 10 meV for H3O+).";
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
        item.description = "Allowed values: electronic (default) and full, as Environ solvent_mode. electronic builds the dielectric cavity from the valence electron density. full adds valence-charge Gaussians selected by sccs_corespread.";
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
        item.description = "Widths of the valence-charge Gaussians used only with sccs_solvent_mode full. One value applies to every atom except those whose known atomic number equals the pseudopotential valence charge (within 1e-8). More than one value requires exactly nat values in STRU atom order (grouped by atom type), and overrides this automatic exclusion. A value <= 0 disables the cavity Gaussian on that atom.";
        item.default_value = "0.5";
        item.unit = "bohr";
        item.set_availability("imp_sol==2");
        item.read_value = [](const Input_Item& item, Parameter& para) {
            std::string error;
            const bool valid = parse_core_spreads(item.str_values, para.input.sccs_corespread, error);
            if (!valid) { ModuleBase::WARNING_QUIT("ReadInput", error); }
        };
        sync_doublevec(input.sccs_corespread, para.input.sccs_corespread.size(), 0.0);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_solvent_radius");
        item.annotation = "solvent radius of the solvent-aware SCCS cavity";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Solvent radius of the solvent-aware SCCS cavity, as Environ solvent_radius (Andreussi et al., J. Chem. Theory Comput. 15, 1996 (2019)). Default 0 (off); Environ's water example uses 3 bohr.";
        item.default_value = "0.0";
        item.unit = "bohr";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_solvent_radius);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_radial_scale");
        item.annotation = "solvent-aware probe radius over sccs_solvent_radius";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Probe radius of the solvent-aware SCCS cavity in units of sccs_solvent_radius, as Environ radial_scale; at least 1, default 2. Used only when sccs_solvent_radius is positive.";
        item.default_value = "2.0";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_radial_scale);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_radial_spread");
        item.annotation = "erfc spread of the solvent-aware probe";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "erfc spread of the solvent-aware probe sphere, as Environ radial_spread; positive, default 0.5 bohr. Used only when sccs_solvent_radius is positive.";
        item.default_value = "0.5";
        item.unit = "bohr";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_radial_spread);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_filling_threshold");
        item.annotation = "probe solute fraction that fills a point";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Solute fraction of the solvent-aware probe sphere above which a point is filled, as Environ filling_threshold; between 0 and 1 (exclusive), default 0.825. Used only when sccs_solvent_radius is positive.";
        item.default_value = "0.825";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_filling_threshold);
        this->add_item(item);
    }
    {
        Input_Item item("sccs_filling_spread");
        item.annotation = "erfc width of the solvent-aware filling";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Width of the erfc step of the solvent-aware filling in the probe solute fraction, as Environ filling_spread; positive, default 0.02. Used only when sccs_solvent_radius is positive.";
        item.default_value = "0.02";
        item.unit = "";
        item.set_availability("imp_sol==2");
        read_sync_double(input.sccs_filling_spread);
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
        item.description = "SCCS/PCC output level, printed to the screen (standard output): 0 suppresses per-SCF summaries and diagnostics; 1 prints the iteration count and correction energy; 2 additionally prints residual, warm-start, cavity volume and surface (with the solvent-aware filled volume), FFT-count, Gauss-law (PCC: far-field polarization charge, its expected value and the comparison tolerance), multipole and energy diagnostics, and verifies the sqrt-CG fixed point with one extra Poisson solve per SCCS evaluation.";
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
