#ifndef ATOM_H_
#define ATOM_H_

#include <string>
#include "point.h"

class Atom {
  public:
  Atom(int, const std::array<double, 3>&);
  Atom(int, const double pos[3]);
  Atom(int, const double pos[3], const std::string&);

  int znum;
  Point pos;
  bool is_ghost;
  std::string tag;
};

#endif // ATOM_H_
