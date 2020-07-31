#ifndef POINT_H_
#define POINT_H_

#include <array>

#ifdef HAVE_CUDA
#include "cuda_runtime.h"
#define HOSTDEVICE __host__ __device__
#else 
#define HOSTDEVICE
#endif

class Point {
  public:
    Point() = default;
    Point(const Point&) = default;
    Point(double, double, double);
    Point(const std::array<double, 3>&);
    Point(const double[]);

    HOSTDEVICE Point& operator += (const Point&);
    HOSTDEVICE Point& operator -= (const Point&);
    HOSTDEVICE Point& operator *= (double);
    HOSTDEVICE Point& operator /= (double);
    HOSTDEVICE double& operator [] (int);
    HOSTDEVICE const double& operator [] (int) const;

    HOSTDEVICE double length() const;
    HOSTDEVICE double length_squared() const;

    HOSTDEVICE static double distance(const Point&, const Point&);
    HOSTDEVICE static double distance_squared(const Point&, const Point&);

//   [[deprecated]]
//   std::array<double, 3>::iterator begin();
//   [[deprecated]]
//   std::array<double, 3>::const_iterator begin() const;
//   [[deprecated]]
//   std::array<double, 3>::iterator end();
//   [[deprecated]]
//   std::array<double, 3>::const_iterator end() const;
//   [[deprecated]]
//   std::array<double, 3> data() const;

  private:
    double x, y, z;
};

HOSTDEVICE Point operator + (Point, const Point&);
HOSTDEVICE Point operator - (Point, const Point&); 

#undef HOSTDEVICE

#endif // POINT_H_
