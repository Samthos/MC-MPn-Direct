#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include "mpi.h"

#include "qc_monte.h"

void fillVector3(std::vector<std::vector<std::vector<double>>>& in) {
  for (auto &it : in) {
    for (auto &jt : it) {  
      std::fill(jt.begin(), jt.end(), 0.0);
    }
  }
}

GFStats::GFStats(bool isMaster_, int tasks_, int numBand, int offBand, int nDeriv, int nBlock, const std::string& jobname, int order) : isMaster(isMaster_), tasks(static_cast<double>(tasks_)){
  qeps = std::vector<std::vector<double>>(numBand, std::vector<double>(nDeriv));      //stores energy correction for current step

  qepsBlock = std::vector<std::vector<std::vector<double>>>(numBand, std::vector<std::vector<double>>(nDeriv, std::vector<double>(nBlock)));  // used to accumlated blocked energied
  qepsEx1 = std::vector<std::vector<std::vector<double>>>(numBand, std::vector<std::vector<double>>(nDeriv, std::vector<double>(nBlock)));  // stores first moment of blocked energy correction
  qepsEx2 = std::vector<std::vector<std::vector<double>>>(numBand, std::vector<std::vector<double>>(nDeriv, std::vector<double>(nBlock)));  // stores second moment of blocked energy correction
  qepsAvg = std::vector<std::vector<std::vector<double>>>(numBand, std::vector<std::vector<double>>(nDeriv, std::vector<double>(nBlock)));  // stores energies after MPI reduce
  qepsVar = std::vector<std::vector<std::vector<double>>>(numBand, std::vector<std::vector<double>>(nDeriv, std::vector<double>(nBlock)));  // stores stat error after MPI reduce

  fillVector3(qepsEx1);
  fillVector3(qepsEx2);
  fillVector3(qepsBlock);

  if (isMaster) {
    output_streams.resize(numBand);
    for(int band=0;band<numBand;band++) {
      output_streams[band] = new std::ofstream[nDeriv];
      for(int deriv=0;deriv<nDeriv;deriv++) {
        char file2[256];
        if((band+1-offBand) < 0) {
          sprintf(file2, "%s.2%i.DIFF%i.HOMO%i" , jobname.c_str(), order, deriv, band-offBand+1);
        } else if((band+1-offBand) == 0) {
          sprintf(file2, "%s.2%i.DIFF%i.HOMO-%i", jobname.c_str(), order, deriv, band-offBand+1);
        } else {
          sprintf(file2, "%s.2%i.DIFF%i.LUMO+%i", jobname.c_str(), order, deriv, band-offBand);
        }
        output_streams[band][deriv].open(file2);
      }
    }
  }
}

GFStats::~GFStats() {
  if (isMaster) {
    for (uint32_t band = 0; band < qeps.size(); band++) {
      for (uint32_t deriv = 0; deriv < qeps[band].size(); deriv++) {
        output_streams[band][deriv].close();
      }
      delete[] output_streams[band];
    }
  }
}

void GFStats::blockIt(const int& step) {
  double diff1, diff2;
  uint32_t band, deriv, block;
  uint32_t blockPower2, blockStep;
  for (band = 0; band < qepsBlock.size(); band++) {  // loop over bands
    for (deriv = 0; deriv < qepsBlock[band].size(); deriv++) {  // loop over number of requested derivatives
      for (block = 0, blockPower2=1; block < qepsBlock[band][deriv].size(); block++, blockPower2 *= 2) {  // loop over block size where 2^j == blockSize
        blockStep = (step-1) % blockPower2 + 1;

        diff1 = qeps[band][deriv] - qepsBlock[band][deriv][block];
        qepsBlock[band][deriv][block] += diff1 / static_cast<double>(blockStep);

        if ((step & (blockPower2-1)) == 0) {  // if block is filled -> accumulate
          blockStep = step / blockPower2;

          diff1 = qepsBlock[band][deriv][block] - qepsEx1[band][deriv][block];
          diff2 = qepsBlock[band][deriv][block]*qepsBlock[band][deriv][block] - qepsEx2[band][deriv][block];

          qepsEx1[band][deriv][block] += diff1 / static_cast<double>(blockStep);
          qepsEx2[band][deriv][block] += diff2 / static_cast<double>(blockStep);
          qepsBlock[band][deriv][block] = 0.00;
        }
      }
    }
  }
}

void GFStats::reduce() {
  for (uint it = 0; it < qepsEx1.size(); it++) {
    for (uint jt = 0; jt < qepsEx1[it].size(); jt++) {
      MPI_Reduce(qepsEx1[it][jt].data(), qepsAvg[it][jt].data(), qepsEx1[it][jt].size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
      MPI_Reduce(qepsEx2[it][jt].data(), qepsVar[it][jt].data(), qepsEx2[it][jt].size(), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
  }
}

void GFStats::print(const int& step, const double& time_span) {
  int blockPow2;
  double step_double = static_cast<double>(step);
  double multiplier = 1.0 / (tasks * step_double * ( tasks * step_double - 1.0 ));

  for(uint32_t band=0;band<qepsAvg.size();band++) {
    for(uint32_t deriv=0;deriv<qepsAvg[band].size();deriv++) {
      output_streams[band][deriv] << std::fixed << std::noshowpos << std::setprecision(3) << step_double/1000.0 << "\t"; //debug2
      blockPow2=1;
      for(uint32_t block=0;block<qepsAvg[band][deriv].size();block++) {
        multiplier = 1.0/ (tasks * (static_cast<double>(step/blockPow2)) - 1.0 );
        qepsAvg[band][deriv][block] = qepsAvg[band][deriv][block] / tasks;
        qepsVar[band][deriv][block] = qepsVar[band][deriv][block] / tasks;
        qepsVar[band][deriv][block] = qepsVar[band][deriv][block] - qepsAvg[band][deriv][block] * qepsAvg[band][deriv][block];
        qepsVar[band][deriv][block] = qepsVar[band][deriv][block] * multiplier;
        qepsVar[band][deriv][block] = sqrt(qepsVar[band][deriv][block]);

        output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << qepsAvg[band][deriv][block] << "\t";  //debug2
        output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << qepsVar[band][deriv][block] << "\t";  //debug2
      
        blockPow2*=2;
      }
      output_streams[band][deriv] << std::fixed << std::showpos << std::setprecision(7) << time_span << "\n"; //debug2
      output_streams[band][deriv].flush();
    }
  }
}
