#include "tau.h"

Tau::Tau(const std::shared_ptr<Movec_Parser> basis) : evals(basis->orbital_energies) {
  iocc1 = basis->iocc1;
  iocc2 = basis->iocc2;
  ivir1 = basis->ivir1;
  ivir2 = basis->ivir2;
  scratch.resize(ivir2);
}

void Tau::copy_p(Tau* other) {
  std::copy(other->p.begin(), other->p.end(), p.begin());
}
