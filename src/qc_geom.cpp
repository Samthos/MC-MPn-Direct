// Copyright 2017

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <algorithm>

#include "qc_mpi.h"

#include "atom_znum.h"
#include "qc_geom.h"

void Molec::read(MPI_info& mpi_info, std::string& filename) {
  /*
   * Reads geometry from an XYZ file
   *
   * Arguments
   *  -MPI_info mpi_info: see qc_mpi.h
   *  -std::string filename: path to xyz file
   */
  const double ang_to_bohr = 1.8897259860;
  int i;
  int znum;
  double pos[3];
  std::string atype;
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

  atom.resize(natom);
  if (mpi_info.sys_master) {
    std::cout << "Printing input geometry in angstroms" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << "\tAtom\t            x                             y                             z" << std::endl;
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
  }
  for (i = 0; i < natom; i++) {
    if (mpi_info.sys_master) {
      input >> atype >> pos[0] >> pos[1] >> pos[2];

      znum = atomic_znum(atype);

      std::cout << "\t " << atype << "\t";
      std::cout << std::setw(30) << std::setprecision(16) << std::fixed << pos[0];
      std::cout << std::setw(30) << std::setprecision(16) << std::fixed << pos[1];
      std::cout << std::setw(30) << std::setprecision(16) << std::fixed << pos[2] << std::endl;

      std::transform(pos, pos+3, pos, [ang_to_bohr](double x) {return x * ang_to_bohr; });
    }

    MPI_info::barrier();
    MPI_info::broadcast_double(&pos[0], 3);
    MPI_info::broadcast_int(&znum, 1);

    atom[i].znum = znum;
    atom[i].pos[0] = pos[0];
    atom[i].pos[1] = pos[1];
    atom[i].pos[2] = pos[2];
  }

  if (mpi_info.sys_master) {
    std::cout << "-----------------------------------------------------------------------------------------------------------" << std::endl;
    std::cout << std::setprecision(6);
    std::cout << std::endl
              << std::endl;
  }
}

void Molec::print(int id) {
  int i;
  std::cout << natom << std::endl;
  for (i = 0; i < natom; i++) {
    std::cout << id << " " << atom[i].znum << " " << atom[i].pos[0] << " " << atom[i].pos[1] << " " << atom[i].pos[2] << std::endl;
  }
}
