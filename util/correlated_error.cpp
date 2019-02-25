#include <numeric>
#include <iostream>
#include <fstream>
#include <vector>

std::vector<double> load_data(char* filename) {
  std::vector<double> v;

  std::ifstream is(filename);
  if (is.good() && is.is_open()) {
    is.seekg(0, std::ios_base::end);
    v.resize(is.tellg() / sizeof(double));
    is.seekg(0, std::ios_base::beg);
    is.read((char*)v.data(), sizeof(double) * v.size());
  } 

  return v;
}

double autovariance(const std::vector<double>&data, double mu, int offset) {
  auto autovar = std::inner_product(data.begin() + offset, data.end(), data.begin(), 0.0, std::plus<>(),
      [&mu](double x, double y) {
        return (x - mu) * (y - mu);
      });
  autovar /= static_cast<double>(data.size() - offset);
  return autovar;
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: correleated_error <FILE>>\n";
    std::cerr << "correlated_error calculated the correleate error of a MC trajectory\n";
    std::cerr << "FILE must contain a the values of a MC Trajectory in binary format.\n";
    std::cerr << "FILE is assumed to contain only a single column of data\n";
    exit(0);
  }

  auto data = load_data(argv[1]);
  if (data.size() != 0) {
    auto mu = std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
    auto var = autovariance(data, mu, 0);

    auto last = (1 << 10);
    for (auto offset = 0; offset <= last; offset++) {
      printf("%5i %18.12f\n", offset, autovariance(data, mu, offset) / var);
    }
  }
}
