//
// Created by aedoran on 6/5/18.
//

#include <include/gtest/gtest.h>
#include <qc_mpi.h>
#include <qc_input.h>
#include <weight_function.h>
#include <qc_basis.h>
#include <qc_monte.h>

TEST(MPTests, MP2) {
//  MPI_info mpi_info;
//
//  IOPs iops;
//  iops.read(mpi_info, "input_files/input");
//  iops.print(mpi_info, "input_files/input");
//
//  Molec molec;
//  molec.read(mpi_info, iops.sopns[KEYS::GEOM]);
//
//  Basis basis;
//  basis.read(iops, mpi_info, molec);
//  basis.nw_vectors_read(mpi_info, molec, iops);
//
//  GTO_Weight mc_basis;
//  mc_basis.read(mpi_info, molec, iops.sopns[KEYS::MC_BASIS]);
//
//  MP2 qc_monte(mpi_info, iops, molec, basis, mc_basis);
//  qc_monte.monte_energy();
};
