#include <sstream>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "mpi.h"

#include "qc_monte.h"

std::string QC_monte::genFileName(int checkNum, int type, int order, int band, int diff, int block) {
  std::stringstream ss;
  std::string str;
  ss.clear();

  ss << iops.sopns[KEYS::JOBNAME] <<  ".";
  ss << "2" << order << ".";
  ss << "CHK" << checkNum << ".";
  if (type == 0) {
    ss << "EX1.FULL.";
  } else if (type == 1) {
    ss << "ERR.FULL.";
  }
  ss <<  "DIFF" << diff << ".";
  ss << "BLOCK" << block << ".";
  if((band+1-offBand) < 0) {
    ss << "HOMO" << band-offBand+1;
  } else if((band+1-offBand) == 0) {
    ss << "HOMO-" << band-offBand+1;
  } else {
    ss << "LUMO+" << band-offBand;
  }
  ss >> str;

  return str;
}
void QC_monte::mc_gf2_full_print(int band, int steps, int checkNum) {
  // variables for streams
  std::stringstream ss;
  std::string str;
  std::ofstream output;

  // vector to copy data to
  std::vector<double> ex1 ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex2 ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex1All ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex2All ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> err ((ivir2-iocc1) * (ivir2-iocc1));

  for (auto diff = 0;  diff < iops.iopns[KEYS::DIFFS]; diff++) {
    int blockPower2 = 1;
    for (auto block = 0;  block < iops.iopns[KEYS::NBLOCK]; block++) {
      // copy first and second moments too host
      mc_gf_copy(ex1, ex2, ovps.d_ovps.en2Ex1[band][diff][block], ovps.d_ovps.en2Ex2[band][diff][block]);

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(ex1.data(), ex1All.data(), ex1.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(ex2.data(), ex2All.data(), ex2.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (mpi_info.sys_master) {

        double numTasks = static_cast<double> (mpi_info.numtasks);
        double numSteps = numTasks * static_cast<double>(steps/blockPower2) - 1.0;

        std::transform(ex1All.begin(), ex1All.end(), ex1All.begin(), [numTasks](double x){return x/numTasks;});
        std::transform(ex2All.begin(), ex2All.end(), ex2All.begin(), [numTasks](double x){return x/numTasks;});
        std::transform(ex1All.begin(), ex1All.end(), ex2All.begin(), err.begin(), [numSteps](double x1, double x2){return sqrt((x2-x1*x1)/numSteps);});

        if (block == 0) {
          // create file name
          str = genFileName(checkNum, 0, 2, band, diff, block);

          // open stream, print, close stream
          output.open(str.c_str(), std::ios::binary);
          output.write((char*)&steps, sizeof(int));
          output.write((char*)ex1All.data(), ex1All.size() * sizeof(double));
          // output << steps << std::endl;
          // for(auto it = 0; it < (ivir2-iocc1); it++) {
          //   for(auto jt = 0; jt < (ivir2-iocc1); jt++) {
          //     output << std::fixed << std::showpos << std::setprecision(7) << ex1All[it*(ivir2-iocc1) + jt] << " ";
          //   }
          //   output << std::endl;
          // }
          output.close();
        }

        // create file name
        str = genFileName(checkNum, 1, 2, band, diff, block);

        // open stream, print, close stream
        output.open(str.c_str());
        output.write((char*)&steps, sizeof(int));
        output.write((char*)err.data(), ex1All.size() * sizeof(double));
        // output << steps << std::endl;
        // for(auto it = 0; it < (ivir2-iocc1); it++) {
        //   for(auto jt = 0; jt < (ivir2-iocc1); jt++) {
        //     output << std::fixed << std::showpos << std::setprecision(7) << err[it*(ivir2-iocc1) + jt] << " ";
        //   }
        //   output << std::endl;
        // }
        output.close();
      }
      blockPower2 *= 2;
    }
  }
}
void QC_monte::mc_gf3_full_print(int band, int steps, int checkNum) {
  // variables for streams
  std::stringstream ss;
  std::string str;
  std::ofstream output;

  // vector to copy data to
  std::vector<double> ex1 ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex2 ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex1All ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> ex2All ((ivir2-iocc1) * (ivir2-iocc1));
  std::vector<double> err ((ivir2-iocc1) * (ivir2-iocc1));

  for (auto diff = 0;  diff < iops.iopns[KEYS::DIFFS]; diff++) {
    int blockPower2 = 1;
    for (auto block = 0;  block < iops.iopns[KEYS::NBLOCK]; block++) {
      // copy first and second moments too host
      mc_gf_copy(ex1, ex2, ovps.d_ovps.en3Ex1[band][diff][block], ovps.d_ovps.en3Ex2[band][diff][block]);

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Reduce(ex1.data(), ex1All.data(), ex1.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(ex2.data(), ex2All.data(), ex2.size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

      if (mpi_info.sys_master) {

        double numTasks = static_cast<double> (mpi_info.numtasks);
        double numSteps = numTasks * static_cast<double>(steps/blockPower2) - 1.0;

        std::transform(ex1All.begin(), ex1All.end(), ex1All.begin(), [numTasks](double x){return x/numTasks;});
        std::transform(ex2All.begin(), ex2All.end(), ex2All.begin(), [numTasks](double x){return x/numTasks;});
        std::transform(ex1All.begin(), ex1All.end(), ex2All.begin(), err.begin(), [numSteps](double x1, double x2){return sqrt((x2-x1*x1)/numSteps);});

        if (block == 0) {
          // create file name
          str = genFileName(checkNum, 0, 3, band, diff, block);

          // open stream, print, close stream
          output.open(str.c_str(), std::ios::binary);
          output.write((char*)&steps, sizeof(int));
          output.write((char*)ex1All.data(), ex1All.size() * sizeof(double));
          // output << steps << std::endl;
          // for(auto it = 0; it < (ivir2-iocc1); it++) {
          //   for(auto jt = 0; jt < (ivir2-iocc1); jt++) {
          //     output << std::fixed << std::showpos << std::setprecision(7) << ex1All[it*(ivir2-iocc1) + jt] << " ";
          //   }
          //   output << std::endl;
          // }
          output.close();
        }

        // create file name
        str = genFileName(checkNum, 1, 3, band, diff, block);

        // open stream, print, close stream
        output.open(str.c_str());
        output.write((char*)&steps, sizeof(int));
        output.write((char*)err.data(), ex1All.size() * sizeof(double));
        // output << steps << std::endl;
        // for(auto it = 0; it < (ivir2-iocc1); it++) {
        //   for(auto jt = 0; jt < (ivir2-iocc1); jt++) {
        //     output << std::fixed << std::showpos << std::setprecision(7) << err[it*(ivir2-iocc1) + jt] << " ";
        //   }
        //   output << std::endl;
        // }
        output.close();
      }
      blockPower2 *= 2;
    }
  }
}
