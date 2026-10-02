#ifndef SCCS_SQRT_CG_H
#define SCCS_SQRT_CG_H

#include <vector>

namespace ModulePW
{
class PW_Basis;
}

namespace ModuleSurchem
{
class ChargeReduction;
}

namespace ModuleSccs
{

class CoulombOperator;
struct PolarizationSolverParameters;
struct PolarizationResult;

// ENVIRON generalized_sqrt solver of the SCCS response: preconditioned CG on
// K v = q with K = P^-1 + f, P = eps^-1/2 G eps^-1/2 and f the sqrt-CG
// coefficient of the dielectric.

// P r with the periodic or PCC-corrected Coulomb operator G. The cavity is
// fixed during a solve, so the scratch arrays are reused by every
// application; only the scalar potential is needed. invsqrt and coulomb must
// outlive the preconditioner.
class SqrtPreconditioner
{
  public:
    SqrtPreconditioner(const std::vector<double>& invsqrt, const CoulombOperator& coulomb);

    void apply(const std::vector<double>& rhs, std::vector<double>& value);

  private:
    const std::vector<double>& invsqrt_;
    const CoulombOperator& coulomb_;
    std::vector<double> weighted_;
    std::vector<double> potential_;
};

// Update the residual norms of polarization; true when both pass
// sccs_tol_rms and sccs_tol_max.
bool residual_converged(const std::vector<double>& residual,
                        const PolarizationSolverParameters& solver,
                        const ModuleSurchem::ChargeReduction& reduction,
                        PolarizationResult& polarization);

// ENVIRON generalized_sqrt warm start: one preconditioned fixed-point step
// v = P(q - f v_old) from the previous potential, whose charge residual is
// f (v_old - v). Keep it in potential and residual only when it improves on
// the cold-start residual of polarization; true when kept.
bool warm_start(const std::vector<double>& charge,
                const std::vector<double>& coefficient,
                const std::vector<double>& initial_potential,
                const ModuleSurchem::ChargeReduction& reduction,
                SqrtPreconditioner& preconditioner,
                std::vector<double>& potential,
                std::vector<double>& residual,
                PolarizationResult& polarization);

// CG from potential and its charge residual. P^-1 z = r for z = P r, so K d
// follows from the recurrence without P^-1. Returns whether both residual
// tolerances were reached.
bool sqrt_cg(const std::vector<double>& coefficient,
             const PolarizationSolverParameters& solver,
             const ModulePW::PW_Basis& basis,
             const ModuleSurchem::ChargeReduction& reduction,
             SqrtPreconditioner& preconditioner,
             std::vector<double>& potential,
             std::vector<double>& residual,
             PolarizationResult& polarization);

// Check the preconditioned equation v = P(q - K v) independently of the CG
// recurrences; this costs one extra Poisson solve.
void check_fixed_point(const std::vector<double>& charge,
                       const std::vector<double>& coefficient,
                       const std::vector<double>& potential,
                       const ModuleSurchem::ChargeReduction& reduction,
                       SqrtPreconditioner& preconditioner,
                       PolarizationResult& polarization);

// Shift a periodic potential to zero cell mean (ENVIRON generalized_sqrt).
void remove_mean(const ModulePW::PW_Basis& basis,
                 const ModuleSurchem::ChargeReduction& reduction,
                 std::vector<double>& potential);

} // namespace ModuleSccs

#endif
