#ifndef SCCS_COULOMB_H
#define SCCS_COULOMB_H

#include <string>
#include <vector>

namespace ModuleSccs
{
// Potential in Hartree for signed charge density in e/Bohr^3. Applications
// are collective over the PW pool; failure leaves potential unchanged.
class CoulombOperator
{
public:
    virtual ~CoulombOperator() {}
    virtual bool has_boundary_correction() const = 0;
    virtual bool apply_potential(const std::vector<double>& charge,
                                 std::vector<double>& potential,
                                 std::string& error) = 0;
};
} // namespace ModuleSccs

#endif
