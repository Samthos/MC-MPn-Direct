#ifndef ATOM_H_
#define ATOM_H_

#include <string>
#include <array>

class Atom {
  public:
  Atom(int, const std::array<double, 3>&);
  Atom(int, const double pos[3]);
  Atom(int, const double pos[3], const std::string&);

  int znum;
  double pos[3];
  bool is_ghost;
  std::string tag;
};

#endif // ATOM_H_
