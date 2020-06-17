//
// Created by aedoran on 6/5/18.
#include <thrust/device_vector.h>
#include "gtest/gtest.h"
#include "../../src/ovps_set.h"

namespace {
  class ovpsSetTest : public testing::Test {
   protected:
    void SetUp() override {
      iocc1 = 1;
      iocc2 = 6;
      ivir1 = 6;
      ivir2 = 16;
      lda = ivir2;
      electron_pairs = 10;
      psi1Tau.resize(lda * electron_pairs);
      psi2Tau.resize(lda * electron_pairs);

      for (int col = 0, idx = 0; col < electron_pairs; col++) {
        for (int row = 0; row < lda; row++, idx++) {
          psi1Tau[idx] =  idx;
          psi2Tau[idx] = -idx;
        }
      }
    }

    double occ_value(int row, int col) {
      if (row == col) {
        return 0;
      }
      return 5*(11 + 3*lda*row + col*lda*(3 + lda*row));
    }

    double vir_value(int row, int col) {
      if (row == col) {
        return 0;
      }
      return 5*(col*lda*(21 + 2*lda*row) + 3*(79 + 7*lda*row));
    }

    int iocc1, iocc2, ivir1, ivir2;
    int lda, electron_pairs;
    std::vector<double> psi1Tau;
    std::vector<double> psi2Tau;
  };

  class ovpsSetHostTest : public ovpsSetTest {
   protected:
    void SetUp () override {
      ovpsSetTest::SetUp();
      ovps_set.resize(electron_pairs);
    }
    
    void fill() {
      std::fill(ovps_set.s_11.begin(), ovps_set.s_11.end(), -1.0);
      std::fill(ovps_set.s_12.begin(), ovps_set.s_12.end(), -1.0);
      std::fill(ovps_set.s_21.begin(), ovps_set.s_21.end(), -1.0);
      std::fill(ovps_set.s_22.begin(), ovps_set.s_22.end(), -1.0);
    }
    OVPS_SET ovps_set;
  };

  TEST_F(ovpsSetHostTest, ovpsSetHostOccS11) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(occ_value(row, col), ovps_set.s_11[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostOccS12) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-occ_value(row, col), ovps_set.s_12[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostOccS21) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-occ_value(row, col), ovps_set.s_21[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostOccS22) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(occ_value(row, col), ovps_set.s_22[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }

  TEST_F(ovpsSetHostTest, ovpsSetHostVirS11) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(vir_value(row, col), ovps_set.s_11[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostVirS12) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-vir_value(row, col), ovps_set.s_12[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostVirS21) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-vir_value(row, col), ovps_set.s_21[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetHostTest, ovpsSetHostVirS22) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(vir_value(row, col), ovps_set.s_22[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }

  class ovpsSetDeviceTest : public ovpsSetTest {
   protected:
    void SetUp () override {
      ovpsSetTest::SetUp();
      ovps_set.resize(electron_pairs);
      host_vec.resize(electron_pairs * electron_pairs);
    }
    
    void fill() {
      std::fill(host_vec.begin(), host_vec.end(), 0.0);
      thrust::fill(ovps_set.s_11.begin(), ovps_set.s_11.end(), -1.0);
      thrust::fill(ovps_set.s_12.begin(), ovps_set.s_12.end(), -1.0);
      thrust::fill(ovps_set.s_21.begin(), ovps_set.s_21.end(), -1.0);
      thrust::fill(ovps_set.s_22.begin(), ovps_set.s_22.end(), -1.0);
    }
    std::vector<double> host_vec;
    OVPS_SET_DEVICE ovps_set;
  };

  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceOccS11) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    thrust::copy(ovps_set.s_11.begin(), ovps_set.s_11.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(occ_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceOccS12) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    thrust::copy(ovps_set.s_12.begin(), ovps_set.s_12.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-occ_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceOccS21) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    thrust::copy(ovps_set.s_21.begin(), ovps_set.s_21.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-occ_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceOccS22) {
    fill();
    ovps_set.update(psi1Tau.data() + iocc1, psi2Tau.data() + iocc1, iocc2 - iocc1, lda);
    thrust::copy(ovps_set.s_22.begin(), ovps_set.s_22.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(occ_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }

  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceVirS11) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    thrust::copy(ovps_set.s_11.begin(), ovps_set.s_11.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(vir_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceVirS12) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    thrust::copy(ovps_set.s_12.begin(), ovps_set.s_12.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-vir_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceVirS21) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    thrust::copy(ovps_set.s_21.begin(), ovps_set.s_21.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(-vir_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
  TEST_F(ovpsSetDeviceTest, ovpsSetDeviceVirS22) {
    fill();
    ovps_set.update(psi1Tau.data() + ivir1, psi2Tau.data() + ivir1, ivir2 - ivir1, lda);
    thrust::copy(ovps_set.s_22.begin(), ovps_set.s_22.end(), host_vec.begin());
    for (int row = 0; row < electron_pairs; row++) {
      for (int col = 0; col < electron_pairs; col++) {
        ASSERT_EQ(vir_value(row, col), host_vec[col * electron_pairs + row]) << "row = " << row << ", col = " << col << "\n";
      }
    }
  }
}
