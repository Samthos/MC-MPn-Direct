#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>


#include "../qc_mpi.h"
#include "../qc_monte.h"
#include "blas_calls.h"

std::string GF::genFileName(int checkNum, int type, int order, int band, int diff, int block) {
  std::stringstream ss;
  std::string str;
  ss.clear();

  ss << iops.sopns[KEYS::JOBNAME] << ".";
  ss << "2" << order << ".";
  ss << "CHK" << checkNum << ".";
  if (type == 0) {
    ss << "EX1.FULL.";
  } else if (type == 1) {
    ss << "ERR.FULL.";
  }
  ss << "DIFF" << diff << ".";
  ss << "BLOCK" << block << ".";
  if ((band + 1 - offBand) < 0) {
    ss << "HOMO" << band - offBand + 1;
  } else if ((band + 1 - offBand) == 0) {
    ss << "HOMO-" << band - offBand + 1;
  } else {
    ss << "LUMO+" << band - offBand;
  }
  ss >> str;

  return str;
}

void GF::mc_gf_full_print(int band, int steps, int checkNum, int order,
    std::vector<std::vector<double>>& d_ex1,
    std::vector<std::vector<std::vector<double>>>& d_cov) {
  int nDiff = iops.iopns[KEYS::DIFFS];
  // variables for streams
  std::stringstream ss;
  std::string str;
  std::ofstream output;

  // vector to copy data to
  std::vector<double> ex1    (static_cast<unsigned long>((ivir2-iocc1) * (ivir2-iocc1) * nDiff));
  std::vector<double> cov    (static_cast<unsigned long>((ivir2-iocc1) * (ivir2-iocc1) * nDiff * (nDiff+1) / 2));
  std::vector<double> ex1All (static_cast<unsigned long>((ivir2-iocc1) * (ivir2-iocc1) * nDiff));
  std::vector<double> covAll (static_cast<unsigned long>((ivir2-iocc1) * (ivir2-iocc1) * nDiff * (nDiff+1) / 2));

  int blockPower2 = 1;

  double numSteps;
  auto numTasks = static_cast<double>(mpi_info.numtasks);
  mc_gf_copy(ex1, d_ex1[band]);
  MPI_info::barrier();
  MPI_info::reduce_double(ex1.data(), ex1All.data(), ex1.size());

  if (mpi_info.sys_master) {
    numSteps = numTasks * static_cast<double>(steps/blockPower2);
    std::transform(ex1All.begin(), ex1All.end(), ex1All.begin(), [numSteps](double x){return x/numSteps;});

    str = genFileName(checkNum, 0, order, band, 0, 0);
    // open stream, print, close stream
    output.open(str.c_str(), std::ios::binary);
    output.write((char*)&steps, sizeof(int));
    output.write((char*)ex1All.data(), ex1All.size() * sizeof(double));
    output.close();
  }
  for (auto block = 0;  block < iops.iopns[KEYS::NBLOCK]; block++) {
    // copy first and second moments too host
    mc_gf_copy(cov, d_cov[band][block]);
    MPI_info::barrier();
    MPI_info::reduce_double(cov.data(), covAll.data(), cov.size());


    if (mpi_info.sys_master) {
      numSteps = numTasks * static_cast<double>(steps/blockPower2);
      std::transform(covAll.begin(), covAll.end(), covAll.begin(), [numSteps](double x){return x/numSteps;});

      dspr_batched((ivir2-iocc1) * (ivir2-iocc1), nDiff, -1.0, ex1All.data(), covAll.data());

      numSteps = numTasks * static_cast<double>(steps/blockPower2) - 1.0;
      std::transform(covAll.begin(), covAll.end(), covAll.begin(), [numSteps](double x){return x/numSteps;});

      // create file name
      str = genFileName(checkNum, 1, order, band, 0, block);

      // open stream, print, close stream
      output.open(str.c_str());
      output.write((char*)&steps, sizeof(int));
      output.write((char*)covAll.data(), covAll.size() * sizeof(double));
      output.close();
    }
    blockPower2 *= 2;
  }
}
