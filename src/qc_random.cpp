#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "qc_random.h"
#include "qc_mpi.h"


Random::Random(int seed_mode, std::string seed_file) :
  uniform_distribution(0.00, 1.00),
  normal_distribution(0.00, 1.00)
{
  int n_tasks, taskid;
  MPI_info::comm_rank(&taskid);
  MPI_info::comm_size(&n_tasks);

  switch (seed_mode) {
    case 0: seed(taskid, n_tasks ,seed_file); break;
    case 1: debug_seed(taskid, n_tasks); break;
    case 2: seed_from_file(taskid, n_tasks ,seed_file); break;
  }
}

void Random::seed(int taskid, int n_tasks, std::string seed_file) {
  if (taskid == 0) {
    std::cout << "Seeding random number generator using random device\n";
  }
  std::random_device r1;

  seed_array seed;
  for (auto &it : seed) {
    it = r1();
  }

  if (!seed_file.empty()) {
    if (taskid == 0) {
      std::cout << "Saving seeds to " << seed_file << "\n";
    }

    std::vector<seed_array> seeds(n_tasks);

    // gather
    MPI_info::gather_vector(&seed, seeds);
    if (0 == taskid) {
      std::ofstream os(seed_file);
      os.write((char*) &n_tasks, sizeof(int));
      os.write((char*) seeds.data(), sizeof(seed_array) * seeds.size());
    }
  }

  std::seed_seq seed1(seed.begin(), seed.end());
  g1.seed(seed1);
}

void Random::debug_seed(int taskid, int n_tasks) { 
  if (taskid == 0) {
    std::cout << "Seeding random number generator using using built in debug seeds\n";
    if (n_tasks > 16) {
      std::cout << "Debug mode 1 only valid for jobs with 16 or fewer tasks.\n";
      exit(0);
    }
  }


  std::string str;
  std::stringstream sstr;
  long long seeds[] = {1793817729, 3227188512, 2530944000, 2295088101,
                       1099163413, 2366715906, 1817017634, 3192454568,
                       2199752324, 1911074146, 2867042420, 3591224665,
                       8321197510, 1781877856, 1785033536, 3983696536};
  sstr << seeds[taskid];
  sstr >> str;
  std::seed_seq seed1(str.begin(), str.end());
  g1.seed(seed1);
}

void Random::seed_from_file(int taskid, int n_tasks, std::string seed_file) {
  if (taskid == 0) {
    std::cout << "Seeding random number generator from file " << seed_file << "\n";
  }

  int n_seeds = n_tasks;
  std::vector<seed_array> seeds(n_tasks);
  if (0 == taskid) {
    if (seed_file.empty()) {
      std::cerr << "SEED_FILE not set\n";
      exit(0);
    }
    std::ifstream is(seed_file);
    if (is.is_open() && is.good()) {
      is.read((char*) &n_seeds, sizeof(int));
      seeds.resize(n_seeds);
      is.read((char*)seeds.data(), n_seeds * sizeof(seed_array));
    } else {
      std::cerr << "SEED_FILE " << seed_file << " failed to open or does not exist\n";
      exit(0);
    }
  }

  if (n_seeds != n_tasks) {
    std::cout << "Seed file " << seed_file << " contains " << n_seeds << ". Current number of tasks is " << n_tasks << ".\n";
    if (n_seeds < n_tasks) {
      exit(0);
    } 
  }

  //broad cast
  seed_array seed;
  MPI_info::scatter_vector(seeds, &seed);

  std::seed_seq seed1(seed.begin(), seed.end());
  g1.seed(seed1);
}

double Random::uniform() {
  return uniform_distribution(g1);
}

double Random::uniform(double low, double high) {
  return (high-low) * uniform() + low;
}

double Random::normal(double mu, double sigma) {
  return sigma * normal_distribution(g1) + mu;
}
