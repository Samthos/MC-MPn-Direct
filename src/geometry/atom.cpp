#include "atom.h"
#include "atom_tag_parser.h"

Atom::Atom(int z, const double p[3], const std::string& t) : znum(z), pos(p), tag(t) {
  is_ghost = false;
  if (tag.substr(0, 2) == "bq") {
    is_ghost = true;
  } else if (tag[0] == 'x' && tag[1] != 'e') {
    is_ghost = true;
  }
}

Atom::Atom(int z, const double p[3]) : Atom(z, p, Atom_Tag_Parser().symbol(z)) {}

Atom::Atom(int z, const std::array<double, 3>& p) : Atom(z, p.data(), Atom_Tag_Parser().symbol(z)) {}
