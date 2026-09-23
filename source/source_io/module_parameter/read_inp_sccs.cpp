#include "source_base/tool_quit.h"
#include "read_input.h"
#include "read_input_tool.h"

#include <cmath>

namespace ModuleIO
{
namespace
{
void check_solvation_model(const Input_para& input)
{
    const std::vector<std::string> allowed = {"legacy", "sccs"};
    if (std::find(allowed.begin(), allowed.end(), input.solvation_model) == allowed.end())
    {
        ModuleBase::WARNING_QUIT("ReadInput", nofound_str(allowed, "solvation_model"));
    }
    if (input.solvation_model == "sccs" && !input.imp_sol)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "solvation_model=sccs requires imp_sol=true");
    }
    if (input.solvation_model == "sccs"
        && (input.efield_flag || input.gate_flag
            || input.assume_isolated != "none"))
    {
        ModuleBase::WARNING_QUIT("ReadInput",
                                 "SCCS cannot be combined with electric/gate fields or assume_isolated");
    }
    if (input.solvation_model == "sccs" && input.nspin == 4)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "the first SCCS implementation does not support nspin=4");
    }
    if (input.solvation_model == "sccs"
        && input.calculation != "scf"
        && input.calculation != "relax")
    {
        ModuleBase::WARNING_QUIT(
            "ReadInput",
            "SCCS supports only calculation=scf or fixed-cell calculation=relax");
    }
    if (input.solvation_model == "sccs" && input.device == "gpu")
    {
        ModuleBase::WARNING_QUIT("ReadInput", "the first SCCS implementation supports only device=cpu");
    }
    if (input.solvation_model == "sccs" && input.dfthalf_type != 0)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "SCCS cannot currently be combined with DFT-1/2");
    }
}

void check_sccs_preset(const Input_para& input)
{
    const std::vector<std::string> allowed
        = {"custom", "water-neutral", "water-cation", "water-anion"};
    if (std::find(allowed.begin(), allowed.end(), input.sccs_preset) == allowed.end())
    {
        ModuleBase::WARNING_QUIT("ReadInput", nofound_str(allowed, "sccs_preset"));
    }
}

void check_sccs_start_drho(const Input_para& input)
{
    if (!std::isfinite(input.sccs_start_drho)
        || input.sccs_start_drho < 0.0)
    {
        ModuleBase::WARNING_QUIT(
            "ReadInput",
            "sccs_start_drho must be finite and non-negative");
    }
}

void check_sccs_start_nmax(const Input_para& input)
{
    if (input.sccs_start_nmax <= 0
        || (input.sccs_start_drho > 0.0
            && input.sccs_start_nmax >= input.scf_nmax))
    {
        ModuleBase::WARNING_QUIT(
            "ReadInput",
            "sccs_start_nmax must be positive and smaller than scf_nmax when delayed start is enabled");
    }
}

void check_sccs_boundary(const Input_para& input)
{
    const std::vector<std::string> allowed = {"periodic", "pcc_0d", "pcc_2d"};
    if (std::find(allowed.begin(), allowed.end(), input.sccs_boundary) == allowed.end())
    {
        ModuleBase::WARNING_QUIT("ReadInput", nofound_str(allowed, "sccs_boundary"));
    }
}

void check_sccs_numerical_parameters(const Input_para& input)
{
    if (input.sccs_maxiter <= 0 || !std::isfinite(input.sccs_epsilon)
        || input.sccs_epsilon < 1.0 || !std::isfinite(input.sccs_rho_min)
        || !std::isfinite(input.sccs_rho_max) || input.sccs_rho_min <= 0.0
        || input.sccs_rho_max <= input.sccs_rho_min
        || !std::isfinite(input.sccs_gamma)
        || !std::isfinite(input.sccs_pressure)
        || !std::isfinite(input.sccs_mixing) || input.sccs_mixing <= 0.0
        || input.sccs_mixing > 1.0 || !std::isfinite(input.sccs_tol_rms)
        || input.sccs_tol_rms <= 0.0 || !std::isfinite(input.sccs_tol_max)
        || input.sccs_tol_max <= 0.0
        || !std::isfinite(input.sccs_surface_eta)
        || input.sccs_surface_eta <= 0.0)
    {
        ModuleBase::WARNING_QUIT("ReadInput", "invalid SCCS numerical parameters");
    }
}
} // namespace

void ReadInput::item_sccs()
{
    {
        Input_Item item("solvation_model");
        item.annotation = "implicit-solvent implementation";
        item.category = "Implicit solvation model";
        item.type = "String";
        item.description = "Select legacy or the native SCCS implementation. SCCS is enabled only together with imp_sol=true and supports scf or fixed-cell relax calculations.";
        item.default_value = "legacy";
        item.unit = "";
        read_sync_string(input.solvation_model);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_solvation_model(para.input);
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccs_preset");
        item.annotation = "SCCS parameter preset";
        item.category = "Implicit solvation model";
        item.type = "String";
        item.description = "Select custom, water-neutral, water-cation, or water-anion parameters. Numerical cavity and non-electrostatic inputs are used only by custom.";
        item.default_value = "custom";
        item.unit = "";
        item.set_availability("imp_sol==true and solvation_model==sccs");
        read_sync_string(input.sccs_preset);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_sccs_preset(para.input);
        };
        this->add_item(item);
    }
#define ADD_SCCS_REAL_ITEM(NAME, MEMBER, DESCRIPTION, DEFAULT_VALUE, UNIT) \
    { \
        Input_Item item(NAME); \
        item.annotation = DESCRIPTION; \
        item.category = "Implicit solvation model"; \
        item.type = "Real"; \
        item.description = DESCRIPTION; \
        item.default_value = DEFAULT_VALUE; \
        item.unit = UNIT; \
        item.set_availability("imp_sol==true and solvation_model==sccs"); \
        read_sync_double(input.MEMBER); \
        this->add_item(item); \
    }
    ADD_SCCS_REAL_ITEM("sccs_epsilon", sccs_epsilon, "SCCS bulk relative permittivity", "78.3", "")
    ADD_SCCS_REAL_ITEM("sccs_rho_min", sccs_rho_min, "SCCS lower cavity-density threshold", "1.0e-4", "bohr^-3")
    ADD_SCCS_REAL_ITEM("sccs_rho_max", sccs_rho_max, "SCCS upper cavity-density threshold", "5.0e-3", "bohr^-3")
    ADD_SCCS_REAL_ITEM("sccs_gamma", sccs_gamma, "SCCS effective surface coefficient", "0.0", "dyn/cm")
    ADD_SCCS_REAL_ITEM("sccs_pressure", sccs_pressure, "SCCS effective volume coefficient", "0.0", "GPa")
    ADD_SCCS_REAL_ITEM("sccs_mixing", sccs_mixing, "SCCS polarization linear mixing", "0.5", "")
    ADD_SCCS_REAL_ITEM("sccs_tol_rms", sccs_tol_rms, "SCCS polarization RMS residual tolerance", "1.0e-10", "e/bohr^3")
    ADD_SCCS_REAL_ITEM("sccs_tol_max", sccs_tol_max, "SCCS polarization maximum residual tolerance", "1.0e-8", "e/bohr^3")
    ADD_SCCS_REAL_ITEM("sccs_surface_eta", sccs_surface_eta, "SCCS surface regularization", "1.0e-8", "bohr^-1")
#undef ADD_SCCS_REAL_ITEM
    {
        Input_Item item("sccs_start_drho");
        item.annotation = "SCCS delayed-start density threshold";
        item.category = "Implicit solvation model";
        item.type = "Real";
        item.description = "Delay SCCS on a cold start until DRHO is at or below this value. Zero starts SCCS immediately. Once activated, SCCS remains active for all later electronic and ionic steps.";
        item.default_value = "0.0";
        item.unit = "";
        item.set_availability("imp_sol==true and solvation_model==sccs");
        read_sync_double(input.sccs_start_drho);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_sccs_start_drho(para.input);
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccs_start_nmax");
        item.annotation = "SCCS delayed-start iteration limit";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "Force delayed SCCS activation at this electronic iteration if the SCCS start DRHO threshold has not yet been reached. The value must be smaller than scf_nmax so a later iteration uses the SCCS Hamiltonian.";
        item.default_value = "30";
        item.unit = "";
        item.set_availability("imp_sol==true and solvation_model==sccs");
        read_sync_int(input.sccs_start_nmax);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_sccs_start_nmax(para.input);
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccs_boundary");
        item.annotation = "SCCS electrostatic boundary condition";
        item.category = "Implicit solvation model";
        item.type = "String";
        item.description = "Select periodic electrostatics, cubic zero-dimensional PCC (pcc_0d), or slab PCC (pcc_2d). The pcc_2d boundary fixes the open/vacuum direction to the second lattice vector (+y) and requires that vector to be perpendicular to the x-z periodic plane. Neutral and charged slabs are supported. For a charged slab, the open-boundary field energy grows linearly with the y cell length, so absolute total energies at different y cell lengths are not directly comparable. PCC includes the smooth-source solvent response plus the point-ion/electron vacuum correction in the host energy, electronic potential, and ionic forces.";
        item.default_value = "periodic";
        item.unit = "";
        item.set_availability("imp_sol==true and solvation_model==sccs");
        read_sync_string(input.sccs_boundary);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_sccs_boundary(para.input);
        };
        this->add_item(item);
    }
    {
        Input_Item item("sccs_maxiter");
        item.annotation = "SCCS polarization iteration limit";
        item.category = "Implicit solvation model";
        item.type = "Integer";
        item.description = "Maximum number of inner SCCS polarization iterations.";
        item.default_value = "200";
        item.unit = "";
        item.set_availability("imp_sol==true and solvation_model==sccs");
        read_sync_int(input.sccs_maxiter);
        item.check_value = [](const Input_Item&, const Parameter& para) {
            check_sccs_numerical_parameters(para.input);
        };
        this->add_item(item);
    }
}
} // namespace ModuleIO
