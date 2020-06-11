#ifndef QC_RANDOM_H_
#define QC_RANDOM_H_
#include <array>
#include <random>

class Random {
 public:
  explicit Random(int, std::string);
  double uniform();
  double uniform(double, double);
  double normal(double, double);

 private:
  typedef std::mt19937 generator_type;

  void seed(int, int, std::string);
  void debug_seed(int, int);
  void seed_from_file(int, int, std::string);

  generator_type g1;
  std::uniform_real_distribution<double> uniform_distribution;
  std::normal_distribution<double> normal_distribution;
  int debug;

  static constexpr int random_device_results_type_size = sizeof(std::random_device::result_type);
  static constexpr int state_size = (generator_type::state_size + 7) / 8;
  typedef std::array<std::random_device::result_type, (state_size + random_device_results_type_size - 1) / random_device_results_type_size > seed_array;
};
#endif  // QC_RANDOM_H_
