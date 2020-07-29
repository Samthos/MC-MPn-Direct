#include <cmath>

#include "point.h"

Point::Point(double x_in, double y_in, double z_in) : p({x_in, y_in, z_in}) { }
Point::Point(const std::array<double, 3>& p_in) : Point(p_in[0], p_in[1], p_in[2]) { }
Point::Point(const double p_in[]) : Point(p_in[0], p_in[1], p_in[2])  { }

Point& Point::operator += (const Point& rhs) {
  p[0] += rhs.p[0];
  p[1] += rhs.p[1];
  p[2] += rhs.p[2];
  return *this;
}

Point& Point::operator -= (const Point& rhs) {
  p[0] -= rhs.p[0];
  p[1] -= rhs.p[1];
  p[2] -= rhs.p[2];
  return *this;
}

Point& Point::operator *= (double rhs) {
  p[0] *= rhs;
  p[1] *= rhs;
  p[2] *= rhs;
  return *this;
}

Point& Point::operator /= (double rhs) {
  p[0] /= rhs;
  p[1] /= rhs;
  p[2] /= rhs;
  return *this;
}

double& Point::operator [] (int idx) {
  return p[idx];
}

const double& Point::operator [] (int idx) const {
  return p[idx];
}

double Point::length() const {
  return sqrt(length_squared());
}

double Point::length_squared() const {
  double d = 0;
  d += p[0] * p[0];
  d += p[1] * p[1];
  d += p[2] * p[2];
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

std::array<double, 3>::iterator Point::begin() {
  return p.begin();
}

std::array<double, 3>::const_iterator Point::begin() const {
  return p.begin();
}

std::array<double, 3>::iterator Point::end() {
  return p.end();
}

std::array<double, 3>::const_iterator Point::end() const {
  return p.end();
}

std::array<double, 3> Point::data() const {
  return p;
}
