#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

#include "nwchem_movec_parser.h"
#include "qc_basis.h"

NWChem_Movec_Parser::NWChem_Movec_Parser(IOPs& iops, MPI_info& mpi_info, Molec& molec, KEYS::KEYS source) {
  if (mpi_info.sys_master) {
    if (0 == iops.iopns[source]) {
      parse_binary_movecs(iops.sopns[source]);
    } else {
      parse_binary_movecs(iops.sopns[source]);
    }

    std::cout << "nw_vectors: nbf " << n_basis_functions << "\n";
    std::cout << "nw_vectors: nmo " << n_molecular_orbitals << "\n";
    n_occupied_orbitals = std::count_if(occupancy.begin(), occupancy.end(), [](double x){return x > 0.0;});
  }

  broadcast();


  //orbital_check();
  iocc1 = 0;
  if (iops.bopns[KEYS::FREEZE_CORE]) {
    for (auto &atom : molec.atoms) {
      if (atom.znum > 3 && atom.znum < 10 && !atom.is_ghost) {
        iocc1 += 1;
      }
    }
  }
  iocc2 = n_occupied_orbitals;
  ivir1 = n_occupied_orbitals;
  ivir2 = n_molecular_orbitals;

  std::cout << iocc1 << "\t";
  std::cout << iocc2 << "\t";
  std::cout << ivir1 << "\t";
  std::cout << ivir2 << "\n";

  if (mpi_info.sys_master) {
    if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GF ||
        iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFDIFF ||
        iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULL ||
        iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFFULLDIFF) {
      log_orbital_energies(iops.sopns[KEYS::JOBNAME]);
    }
  }
}

void NWChem_Movec_Parser::read(std::ifstream& input, char* v, bool set_null=false) {
  int size;
  input.read((char*)&size, 4);
  input.read(v, size);
  if (set_null) {
    v[size] = '\0';
  }
  input.read((char*)&size, 4);
}

void NWChem_Movec_Parser::parse_binary_movecs(std::string filename) {
  long long title_length;
  long long basis_title_length;
  char ignoreChar[256];

  std::ifstream input;

  std::cout << "Reading binary MOVECS from " << filename << std::endl;
  input.open(filename, std::ios::binary);
  if (!input.is_open()) {
    std::cerr << "no movecs file" << std::endl;
    exit(EXIT_FAILURE);
  }

  //get calcaultion info
  read(input, ignoreChar, true);

  //calcualtion type
  read(input, ignoreChar, true);

  //title length
  read(input, (char*) &title_length);

  //title
  read(input, ignoreChar, true);

  //basis name length
  read(input, (char*) &basis_title_length);

  //basis name
  read(input, ignoreChar, true);

  //nwsets
  read(input, (char*) &nw_nsets);

  //nw_nbf
  read(input, (char*) &n_basis_functions);

  //nw_nmo
  if (nw_nsets > 1) {
    std::cerr << "nw_nsets > 1" << std::endl;
    std::cerr << "Code only supports nw_nset==1" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    read(input, (char*) &n_molecular_orbitals);
  }

  resize();

  read(input, (char*) occupancy.data());
  read(input, (char*) orbital_energies.data());
  for (int i = 0; i < n_molecular_orbitals; i++) {
    read(input, (char*) (movecs.data() + i * n_basis_functions));
  }
}

void NWChem_Movec_Parser::parse_ascii_movecs(std::string filename) {
  long long title_length;
  long long basis_title_length;
  std::string scftype20;
  std::string title;
  std::string basis_name;

  std::ifstream input;

  std::cout << "Reading ascii MOVECS from " << filename << std::endl;
  input.open(filename);
  if (!input.is_open()) {
    std::cerr << "no movecs file" << std::endl;
    exit(EXIT_FAILURE);
  }

  input.ignore(1000, '\n');  // #
  input.ignore(1000, '\n');  // skip convergence info
  input.ignore(1000, '\n');  // skip convergence info
  input.ignore(1000, '\n');  // space
  input.ignore(1000, '\n');  // scftype20
  input.ignore(1000, '\n');  // date lentit
  input >> scftype20;
  input >> title_length;
  input.ignore(1000, '\n');
  std::getline(input, title);

  input >> basis_title_length;
  input.ignore(1000, '\n');
  std::getline(input, basis_name);

  input >> nw_nsets >> n_basis_functions;

  if (nw_nsets > 1) {
    std::cerr << "nw_nsets > 1" << std::endl;
    std::cerr << "Code only supports nw_nset==1" << std::endl;
    exit(EXIT_FAILURE);
  } else {
    input >> n_molecular_orbitals;
  }

  resize();

  for (int i = 0; i < n_molecular_orbitals; i++) {
    input >> occupancy[i];
  }
  for (int i = 0; i < n_molecular_orbitals; i++) {
    input >> orbital_energies[i];
  }

  for (int i = 0, index = 0; i < n_molecular_orbitals; i++) {
    for (int j = 0; j < n_basis_functions; j++) {
      input >> movecs[index++];
    }
  }
}
