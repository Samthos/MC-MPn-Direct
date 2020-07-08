#ifndef ATOMIC_ORBITAL_H_
#define ATOMIC_ORBITAL_H_
struct BasisMetaData{
  int angular_momentum;
  int contraction_begin;
  int contraction_end;
  int ao_begin;
  double pos[3];
};
#endif  // ATOMIC_ORBITAL_H_
