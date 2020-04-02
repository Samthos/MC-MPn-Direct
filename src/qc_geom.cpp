// Copyright 2017

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

#include "qc_mpi.h"

#include "atom_tag_parser.h"
#include "qc_geom.h"

void Molec::read(const MPI_info& mpi_info, const std::string& filename) {
  /*
   * Reads geometry from an XYZ file
   *
   * Arguments
   *  -MPI_info mpi_info: see qc_mpi.h
   *  -std::string filename: path to xyz file
   */
  const double ang_to_bohr = 1.8897259860;

  int natom;
  int znum;
  double pos[3];

  Atom_Tag_Parser atom_tag_parser;
  std::string atom_tag;

  std::ifstream input;

  if (mpi_info.sys_master) {
    std::cout << "Reading geometry from " << filename << std::endl;
    input.open(filename.c_str());
    if (!input.is_open()) {
      std::cerr << filename << " does not exist" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    input >> natom;
  }

  MPI_info::barrier();
  MPI_info::broadcast_int(&natom, 1);

  atoms.resize(natom);
  if (mpi_info.sys_master) {
    std::cout << "Printing input geometry in angstroms\n";
    std::cout << "-------------------------------------------------------------------------------------------------\n";
    std::cout << std::setw(15) << "Atom-Tag";
    std::cout << std::setw(7) << "Charge";
    std::cout << std::setw(8) << "X";
    std::cout << std::setw(25) << "Y";
    std::cout << std::setw(25) << "Z" << "\n";
    std::cout << "-------------------------------------------------------------------------------------------------\n";
  }
  for (int i = 0; i < natom; i++) {
    if (mpi_info.sys_master) {
      input >> atom_tag >> pos[0] >> pos[1] >> pos[2];

      znum = atom_tag_parser.parse(atom_tag);

      std::cout << std::setw(15) << atom_tag;
      std::cout << std::setw(7) << znum;
      std::cout << std::setw(25) << std::setprecision(16) << std::fixed << pos[0];
      std::cout << std::setw(25) << std::setprecision(16) << std::fixed << pos[1];
      std::cout << std::setw(25) << std::setprecision(16) << std::fixed << pos[2] << "\n";

      std::transform(pos, pos+3, pos, [ang_to_bohr](double x) {return x * ang_to_bohr; });
    }

    MPI_info::barrier();
    MPI_info::broadcast_double(&pos[0], 3);
    MPI_info::broadcast_int(&znum, 1);
    MPI_info::broadcast_string(atom_tag);

    atoms[i].znum = znum;
    atoms[i].pos[0] = pos[0];
    atoms[i].pos[1] = pos[1];
    atoms[i].pos[2] = pos[2];
    atoms[i].tag = atom_tag;
  }

  if (mpi_info.sys_master) {
    std::cout << "-------------------------------------------------------------------------------------------------\n";
    std::cout << std::setprecision(6);
    std::cout << std::endl << std::endl;
  }
}

void Molec::print() {
  std::cout << "-------------------------------------------------------------------------------------------------\n";
  std::cout << std::setw(15) << "Atom-Tag";
  std::cout << std::setw(7) << "Charge";
  std::cout << std::setw(8) << "X";
  std::cout << std::setw(25) << "Y";
  std::cout << std::setw(25) << "Z" << "\n";
  std::cout << "-------------------------------------------------------------------------------------------------\n";

  for (auto &atom : atoms) {
    std::cout << std::setw(15) << atom.tag;
    std::cout << std::setw(7) << atom.znum;
    std::cout << std::setw(25) << std::setprecision(16) << std::fixed << atom.pos[0];
    std::cout << std::setw(25) << std::setprecision(16) << std::fixed << atom.pos[1];
    std::cout << std::setw(25) << std::setprecision(16) << std::fixed << atom.pos[2] << "\n";
  }

  std::cout << "-------------------------------------------------------------------------------------------------\n";
  std::cout << std::setprecision(6);
  std::cout << std::endl << std::endl;
}
