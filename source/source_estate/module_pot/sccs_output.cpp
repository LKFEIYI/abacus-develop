#include "sccs_output.h"
#include <algorithm>
#include <iomanip>
#include <ostream>

namespace
{
class StreamState
{
public:
    explicit StreamState(std::ostream& stream) : stream_(stream)
    {
        flags_ = stream.flags();
        precision_ = stream.precision();
    }
    ~StreamState()
    {
        stream_.flags(flags_);
        stream_.precision(precision_);
    }
private:
    std::ostream& stream_;
    std::ios_base::fmtflags flags_;
    std::streamsize precision_;
};
void write_origin(std::ostream& output, const elecstate::PccOutput& result)
{
    if (result.slab)
    {
        output << " PCC2D_ORIGIN AXIS " << result.axis
               << " COORDINATE/Bohr " << result.coordinate << '\n';
    }
    else
    {
        output << " PCC0D_ORIGIN X/Bohr " << result.origin.x
               << " Y/Bohr " << result.origin.y << " Z/Bohr " << result.origin.z << '\n';
    }
}
void write_moments(std::ostream& output, const char* label, const elecstate::ChargeMoments& moments)
{
    output << " PCC0D_MOMENTS " << label << " Q/e " << moments.charge
           << " PX/eBohr " << moments.dipole.x << " PY/eBohr " << moments.dipole.y
           << " PZ/eBohr " << moments.dipole.z << " QRR/eBohr2 " << moments.second_moment << '\n';
}
}
namespace elecstate
{
double sccs_screening_tolerance(double electron_count, double ionic_charge, double tolerance_max, double volume)
{
    const double system_charge = std::max(electron_count, ionic_charge);
    const double charge_scale = std::max(1.0, system_charge);
    const double relative_tolerance = 1e-4 * charge_scale;
    const double grid_tolerance = tolerance_max * volume;
    const double solver_tolerance = std::max(grid_tolerance, relative_tolerance);
    return std::max(1e-6, solver_tolerance);
}

void write_pcc_output(std::ostream& output, const PccOutput& result, int level, double energy)
{
    if (level == 0) { return; }
    StreamState state(output);
    output << std::setprecision(12) << " E_PCC/Ry " << energy << '\n';
    if (level < 2) { return; }
    write_origin(output, result);
    if (result.slab)
    {
        const double dipole = result.moments[1].dipole * result.normal;
        output << " PCC2D_MOMENTS Q_POINT/e " << result.moments[1].charge
               << " PN_POINT/eBohr " << dipole << " QNN_POINT/eBohr2 " << result.moments[1].second_moment << '\n';
    }
    else { write_moments(output, "POINT", result.moments[1]); }
    const double point_energy = 0.5 * energy;
    output << " PCC_ENERGY PCC_POINT/Ha " << point_energy << " PCC_USED/Ry " << energy << '\n';
}
void write_sccs_output(std::ostream& output, const SccsOutput& result, int level,
                       double solvation_energy, bool pcc, double pcc_energy)
{
    if (level == 0 || !result.valid) { return; }
    StreamState state(output);
    output << " SCCS_ITER " << result.iterations << " E_SOL/Ry " << std::setprecision(8) << solvation_energy;
    if (pcc) { output << " E_PCC/Ry " << pcc_energy; }
    output << '\n';
    if (level < 2) { return; }
    output << " SCCS_RESIDUAL RMS " << result.residual_rms << " MAX " << result.residual_max
           << " WARM_START " << result.warm_started << '\n';
    if (result.fixed_point_checked)
    {
        output << " SCCS_CG_FIXED_POINT_DEFECT RMS " << result.fixed_point_defect_rms
               << " MAX " << result.fixed_point_defect_max << '\n';
    }
    if (pcc)
    {
        output << " SCCS_GAUSS Q_POL_FAR_FIELD/e " << result.far_field_charge
               << " Q_POL_DENSITY/e " << result.pcc.moments[2].charge
               << " Q_POL_EXPECTED/e " << result.expected_charge
               << " TOLERANCE/e " << result.screening_tolerance << '\n';
    }
    output << " SCCS_CAVITY VOLUME/Bohr3 " << result.volume << " SURFACE/Bohr2 " << result.surface << '\n'
           << " SCCS_FFT R2G_CALLS " << result.transforms.forward_calls
           << " G2R_CALLS " << result.transforms.inverse_calls
           << " CACHED_SOURCES " << result.reused_fixed_sources << '\n';
    if (!pcc) { return; }
    output << std::setprecision(12);
    const PccOutput& data = result.pcc;
    if (data.slab)
    {
        const double smooth_dipole = data.moments[0].dipole * data.normal;
        const double point_dipole = data.moments[1].dipole * data.normal;
        output << " PCC2D_MOMENTS Q_SMOOTH/e " << data.moments[0].charge
               << " PN_SMOOTH/eBohr " << smooth_dipole << " QNN_SMOOTH/eBohr2 " << data.moments[0].second_moment
               << " Q_POINT/e " << data.moments[1].charge << " PN_POINT/eBohr " << point_dipole
               << " QNN_POINT/eBohr2 " << data.moments[1].second_moment << '\n';
    }
    else
    {
        write_origin(output, data);
        const char* labels[] = {"SMOOTH", "POINT", "POLARIZATION", "SCREENED"};
        for (int i = 0; i < 4; ++i) { write_moments(output, labels[i], data.moments[i]); }
    }
    const char* label = data.slab ? " PCC2D_ENERGY" : " PCC0D_ENERGY";
    const double point_energy = 0.5 * pcc_energy;
    output << label << " REACTION/Ha " << result.reaction_energy << " PCC_SMOOTH/Ha " << data.smooth_energy
           << " PCC_POINT/Ha " << point_energy << " PCC_USED/Ry " << pcc_energy << '\n';
}
void write_sccs_final_output(std::ostream& output, const SccsOutput& result, bool slab, double pcc_energy)
{
    if (!result.valid) { return; }
    StreamState state(output);
    const double point_energy = 0.5 * pcc_energy;
    const double electrostatic_energy = 2.0 * result.reaction_energy + pcc_energy;
    output << std::setprecision(16)
           << " SCCS_DIAGNOSTIC reaction_energy_hartree " << result.reaction_energy << '\n'
           << " SCCS_DIAGNOSTIC smooth_vacuum_pcc_energy_hartree " << result.pcc.smooth_energy << '\n'
           << " SCCS_DIAGNOSTIC point_vacuum_pcc_energy_hartree " << point_energy << '\n'
           << " SCCS_DIAGNOSTIC electrostatic_energy_rydberg " << electrostatic_energy << '\n';
    if (!slab) { return; }
    const char* labels[] = {"smooth_solute", "point_solute", "polarization", "screened"};
    for (int i = 0; i < 4; ++i)
    {
        const ChargeMoments& moments = result.pcc.moments[i];
        const double dipole = moments.dipole * result.pcc.normal;
        output << " SCCS_DIAGNOSTIC " << labels[i] << "_charge " << moments.charge << '\n'
               << " SCCS_DIAGNOSTIC " << labels[i] << "_dipole_normal " << dipole << '\n'
               << " SCCS_DIAGNOSTIC " << labels[i] << "_quadrupole_normal " << moments.second_moment << '\n';
    }
    output << " SCCS_DIAGNOSTIC far_field_polarization_charge " << result.far_field_charge << '\n';
}
} // namespace elecstate
