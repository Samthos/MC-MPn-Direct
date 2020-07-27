#ifdef HAVE_MPI
#include "mpi.h"
#endif

#include "weight_function.h"
#include "basis/basis.h"
#include "qc_monte.h"

int main(int argc, char* argv[]) {
#ifdef HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  MPI_info mpi_info;

  if (argc != 2) {
    if (mpi_info.sys_master) {
      printf("Usage: mcmpN.x <input>\n");
    }
    exit(EXIT_FAILURE);
  } else {
    if (mpi_info.sys_master) {
      printf("MC-GFn program developed by the Hirata lab\n");
      printf("Code compiled from Git-Commit %s\n\n", VERSION);
    }
  }
  mpi_info.print();

  IOPs iops;
  iops.read(mpi_info, argv[1]);
  iops.print(mpi_info, argv[1]);

  Molecule molec(mpi_info, iops.sopns[KEYS::GEOM]);

  Basis_Host basis(std::max(iops.iopns[KEYS::ELECTRON_PAIRS], iops.iopns[KEYS::ELECTRONS]), Basis_Parser(iops.sopns[KEYS::BASIS], iops.bopns[KEYS::SPHERICAL], mpi_info, molec));

  QC_monte<std::vector<double>>* qc_monte;
  if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::ENERGY) {
#ifdef HAVE_CUDA
    qc_monte = new GPU_Energy(mpi_info, iops, molec, basis);
#else
    qc_monte = new Energy<std::vector<double>>(mpi_info, iops, molec, basis);
#endif
  } else if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::DIMER) {
    qc_monte = new Dimer(mpi_info, iops, molec, basis);
  } else if (iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GF || iops.iopns[KEYS::JOBTYPE] == JOBTYPE::GFDIFF) {
    qc_monte = new Diagonal_GF(mpi_info, iops, molec, basis);
  } else {
    if (iops.iopns[KEYS::ORDER] == 2) {
      qc_monte = new GF2(mpi_info, iops, molec, basis);
    } else if (iops.iopns[KEYS::ORDER] == 3) {
      qc_monte = new GF3(mpi_info, iops, molec, basis);
    }
  }
  qc_monte->monte_energy();
  delete qc_monte;

#ifdef HAVE_MPI
  MPI_Finalize();
#endif
}
