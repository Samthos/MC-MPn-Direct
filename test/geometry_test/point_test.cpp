#include <cmath>

#include "gtest/gtest.h"
#include "point.h"

namespace {
  TEST(PointTest, LengthTest) {
    Point p(2.0, 2.0, 2.0);
    ASSERT_FLOAT_EQ(sqrt(3.0) * 2.0, p.length());
  }

  TEST(PointTest, LengthSquaredTest) {
    Point p(3.0, 3.0, 3.0);
    ASSERT_FLOAT_EQ(27.0, p.length_squared());
  }

  TEST(PointTest, DistanceTest) {
    Point p(-1.0, -1.0, -1.0);
    Point q(1.0, 1.0, 1.0);
    ASSERT_FLOAT_EQ(sqrt(3.0) * 2.0, Point::distance(q, p));
  }

  TEST(PointTest, DistanceSquaredTest) {
    Point p(-1.5, -1.5, -1.5);
    Point q(1.5, 1.5, 1.5);
    ASSERT_FLOAT_EQ(27.0, Point::distance_squared(q, p));
  }

  TEST(PointTest, PlusEqualTest) {
    Point p(1.0, 2.0, 3.0);
    p += p;
    ASSERT_FLOAT_EQ(2.0, p[0]);
    ASSERT_FLOAT_EQ(4.0, p[1]);
    ASSERT_FLOAT_EQ(6.0, p[2]);
  }

  TEST(PointTest, MinusEqualTest) {
    Point p(1.0, 2.0, 3.0);
    Point q(1.0, 2.0, 3.0);
    p -= q;
    p -= q;
    p -= q;
    ASSERT_FLOAT_EQ(-2.0, p[0]);
    ASSERT_FLOAT_EQ(-4.0, p[1]);
    ASSERT_FLOAT_EQ(-6.0, p[2]);
  }

  TEST(PointTest, MulEqualTest) {
    Point p(1.0, 2.0, 3.0);
    p *= 5;
    ASSERT_FLOAT_EQ( 5.0, p[0]);
    ASSERT_FLOAT_EQ(10.0, p[1]);
    ASSERT_FLOAT_EQ(15.0, p[2]);
  }

  TEST(PointTest, DivEqualTest) {
    Point p(5.0, 10.0, 15.0);
    p /= 5;
    ASSERT_FLOAT_EQ(1.0, p[0]);
    ASSERT_FLOAT_EQ(2.0, p[1]);
    ASSERT_FLOAT_EQ(3.0, p[2]);
  }
}
