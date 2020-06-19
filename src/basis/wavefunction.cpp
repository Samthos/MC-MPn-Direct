#include <vector>
#include "wavefunction.h"

Wavefunction::Wavefunction(std::vector<std::array<double, 3>>* p, const NWChem_Movec_Parser& movecs_in) :
  iocc1(movecs_in.iocc1),
  iocc2(movecs_in.iocc2),
  ivir1(movecs_in.ivir1),
  ivir2(movecs_in.ivir2),
  movecs(movecs_in.movecs),
  electrons(p->size()),
  lda(ivir2),
  psi(lda * electrons, 0.0),
  psiTau(lda * electrons, 0.0),
  pos(p)
{ }

const double *Wavefunction::data() const {
  return psi.data();
}
const double *Wavefunction::occ() const {
  return psi.data() + iocc1;
}
const double *Wavefunction::vir() const {
  return psi.data() + ivir1;
}
double *Wavefunction::dataTau() {
  return psiTau.data();
}
double *Wavefunction::occTau() {
  return psiTau.data() + iocc1;
}
double *Wavefunction::virTau() {
  return psiTau.data() + ivir1;
}
