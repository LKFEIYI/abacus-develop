#include "surchem.h"

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

void surchem::set_parameters(const SurchemParameters& parameters)
{
    if (parameters.use_sccs)
    {
        if (parameters.expected_electron_count < 0.0
            || parameters.expected_ionic_charge < 0.0
            || parameters.normalization_tolerance <= 0.0
            || parameters.pool_process_count <= 0)
        {
            throw std::invalid_argument("SCCS system and reduction parameters are invalid");
        }
    }
    this->parameters_ = parameters;
    this->parameters_set_ = true;
    this->sccs_state_ = ModuleSccs::SccsState();
}

bool surchem::uses_sccs() const
{
    return this->parameters_set_ && this->parameters_.use_sccs;
}

const ModuleSccs::SccsResult& surchem::sccs_result() const
{
    if (!this->uses_sccs())
    {
        throw std::logic_error("SCCS result requested while the legacy solvent backend is active");
    }
    return this->sccs_result_;
}

void surchem::allocate(const int &nrxx, const int &nspin)
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

    this->Vcav.create(0, 0); 
    this->Vel.create(0, 0);
    this->sccs_state_ = ModuleSccs::SccsState();
    this->sccs_result_ = ModuleSccs::SccsResult();
}

surchem::~surchem()
{
    this->clear();
}
