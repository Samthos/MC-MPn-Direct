#include <stdexcept>
#include <cmath>

#include "point.h"

#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#define HOSTDEVICE __host__ __device__
#else 
#define HOSTDEVICE
#endif

Point::Point(double x_in, double y_in, double z_in) :x(x_in), y(y_in), z(z_in) { }
Point::Point(const std::array<double, 3>& p_in) : Point(p_in[0], p_in[1], p_in[2]) { }
Point::Point(const double p_in[]) : Point(p_in[0], p_in[1], p_in[2])  { }

Point& Point::operator += (const Point& rhs) {
  x += rhs.x;
  y += rhs.y;
  z += rhs.z;
  return *this;
}

Point& Point::operator -= (const Point& rhs) {
  x -= rhs.x;
  y -= rhs.y;
  z -= rhs.z;
  return *this;
}

Point& Point::operator *= (double rhs) {
  x *= rhs;
  y *= rhs;
  z *= rhs;
  return *this;
}

Point& Point::operator /= (double rhs) {
  x /= rhs;
  y /= rhs;
  z /= rhs;
  return *this;
}

double& Point::operator [] (int idx) {
  double& r = x;
  switch (idx) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
  }
  return r;
}

const double& Point::operator [] (int idx) const {
  const double& r = x;
  switch (idx) {
    case 0: return x;
    case 1: return y;
    case 2: return z;
  }
  return r;
}

double Point::length() const {
  return sqrt(length_squared());
}

double Point::length_squared() const {
  double d = 0;
  d += x * x;
  d += y * y;
  d += z * z;
  return d;
}

double Point::distance(const Point& lhs, const Point& rhs) {
  return (lhs - rhs).length();
}

double Point::distance_squared(const Point& lhs, const Point& rhs) {
  return (lhs - rhs).length_squared();
}

Point operator + (Point lhs, const Point& rhs) {
  lhs += rhs;
  return lhs;
}

Point operator - (Point lhs, const Point& rhs) {
  lhs -= rhs;
  return lhs;
}

#undef HOSTDEVICE

//std::array<double, 3>::iterator Point::begin() {
//  return p.begin();
//}
//
//std::array<double, 3>::const_iterator Point::begin() const {
//  return p.begin();
//}
//
//std::array<double, 3>::iterator Point::end() {
//  return p.end();
//}
//
//std::array<double, 3>::const_iterator Point::end() const {
//  return p.end();
//}
//
//std::array<double, 3> Point::data() const {
//  return p;
//}
