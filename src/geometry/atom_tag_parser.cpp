#include <algorithm>
#include <iostream>
#include "atom_tag_parser.h"

Atom_Tag_Record::Atom_Tag_Record(std::string name, std::string symbol, int c) : atom_name(name), atom_symbol(symbol), charge(c) {}
bool Atom_Tag_Record::compare_name(const std::string& tag) {
 if (tag.size() < atom_name.size()) {
   return false;
 }
 return std::equal(atom_name.begin(), atom_name.end(), tag.begin());
}
bool Atom_Tag_Record::compare_symbol(const std::string& tag) {
 if (tag.size() < atom_symbol.size()) {
   return false;
 }
 return std::equal(atom_symbol.begin(), atom_symbol.end(), tag.begin());
}
int Atom_Tag_Record::symbol_length() {
  return atom_symbol.size();
}
int Atom_Tag_Record::get_charge() {
  return charge;
}
std::string Atom_Tag_Record::symbol() {
  return atom_symbol;
}

int Atom_Tag_Parser::parse(std::string tag) {
  /*
   * Parses string into an atom charge. 
   * Reutrns int greater than 0 upon success
   * otherwise return -1
   */

  // make tag lowercase
  std::transform(tag.begin(), tag.end(), tag.begin(), ::tolower);

  // remove leading charcters that would denote dummy atom in nwchem
  if (tag.substr(0, 2) == "bq") {
    tag = tag.substr(2);
  } else if (tag[0] == 'x' && tag[1] != 'e') {
    tag = tag.substr(1);
  }

  // check if tag corresponds to atom name
  for (auto it = atom_tag_records.begin(); it != atom_tag_records.end(); it++) {
    if (it->compare_name(tag)) { 
      return it->get_charge();
    }
  }

  // check if tag corresponds to atom symbol
  for (int symbol_length = 2; symbol_length > 0; symbol_length--) {
    for (auto it = atom_tag_records.begin(); it != atom_tag_records.end(); it++) {
      if (it->symbol_length() == symbol_length) {
        if (it->compare_symbol(tag)) { 
          return it->get_charge();
        }
      }
    }
  }
  return -1;
}
std::string Atom_Tag_Parser::symbol(int znum) {
  return atom_tag_records[znum - 1].symbol();
}
