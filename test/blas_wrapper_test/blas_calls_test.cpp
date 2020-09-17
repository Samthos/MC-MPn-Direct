#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "blas_calls.h"


namespace {
  TEST(blasTest, TransposeTest) {
    constexpr int N = 19;
    std::vector<double> A(N * N);
    std::vector<double> B(N * N, 0.0);

    std::iota(A.begin(), A.end(), 0.0);
    Transpose(A.data(), N, B.data());
    for (int row = 0; row < N; row++) {
      for (int col = 0; col < N; col++) {
        ASSERT_EQ(A[row * N + col], B[col * N + row]) << "(row, col) == (" << row << ", " << col << ")";
      }
    }
  }

  TEST(blasTest, SetUpperFromLowerTest) {
    constexpr int N = 19;
    std::vector<double> A(N * N, 0);

    for (int row = 0; row < N; row++) {
      for (int col = row; col < N; col++) {
        A[row * N + col] = row * N + col;
      }
    }

    set_Upper_from_Lower(A.data(), N);

    for (int row = 0; row < N; row++) {
      for (int col = row; col < N; col++) {
        ASSERT_EQ(A[row * N + col], A[col * N + row]) << "(row, col) == (" << row << ", " << col << ")";
      }
    }
  }
}
