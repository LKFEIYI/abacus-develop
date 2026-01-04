#include "surchem.h"
#include <fstream>
#include <iomanip>

double surchem::Acav = 0;
double surchem::Ael = 0;

surchem::surchem()
{
    TOTN_real = nullptr;
    delta_phi = nullptr;
    epspot = nullptr;
    Vcav = ModuleBase::matrix();
    Vel = ModuleBase::matrix();
    qs = 0;
}

void surchem::allocate(const int& nrxx, const int& npw, const int& nspin)
{
    assert(nrxx >= 0);
    assert(nspin > 0);

    delete[] TOTN_real;
    delete[] delta_phi;
    delete[] epspot;
    if (nrxx > 0)
    {
        TOTN_real = new double[nrxx];
        delta_phi = new double[nrxx];
        epspot = new double[nrxx];
    }
    else
    {
        TOTN_real = nullptr;
        delta_phi = nullptr;
        epspot = nullptr;
    }
    Vcav.create(nspin, nrxx);
    Vel.create(nspin, nrxx);

    ModuleBase::GlobalFunc::ZEROS(delta_phi, nrxx);
    ModuleBase::GlobalFunc::ZEROS(TOTN_real, nrxx);
    ModuleBase::GlobalFunc::ZEROS(epspot, nrxx);

    // CG Buffers
    if (cg_resid.size() != npw) cg_resid.resize(npw);
    if (cg_z.size() != npw) cg_z.resize(npw);
    if (cg_lp.size() != npw) cg_lp.resize(npw);
    if (cg_gsqu.size() != npw) cg_gsqu.resize(npw);
    if (cg_d.size() != npw) cg_d.resize(npw);

    // Leps2 Buffers
    if (le_grad_grad_phi_G.size() != npw) le_grad_grad_phi_G.resize(npw);
    
    if (le_grad_phi.size() != nrxx) le_grad_phi.resize(nrxx);
    if (le_tmp_vector3.size() != nrxx) le_tmp_vector3.resize(nrxx);
    if (le_lp_real.size() != nrxx) le_lp_real.resize(nrxx);
    if (le_aux_real.size() != nrxx) le_aux_real.resize(nrxx);
    return;
}

void surchem::clear()
{
    delete[] TOTN_real;
    delete[] delta_phi;
    delete[] epspot;
    this->TOTN_real = nullptr;
    this->delta_phi = nullptr;
    this->epspot = nullptr;
}

surchem::~surchem()
{
    this->clear();
}
