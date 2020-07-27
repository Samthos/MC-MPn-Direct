#ifndef ATOM_H_
#define ATOM_H_

struct Atom {
  double pos[3];
  int znum;
  std::string tag;
  bool is_ghost;
};

#endif // ATOM_H_
