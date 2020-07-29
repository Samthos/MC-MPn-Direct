#ifndef POINT_H_
#define POINT_H_
#include <array>

class Point {
  public:
    Point() = default;
    Point(double, double, double);
    Point(const std::array<double, 3>&);
    Point(const double[]);

    Point& operator += (const Point&);
    Point& operator -= (const Point&);
    Point& operator *= (double);
    Point& operator /= (double);
    double& operator [] (int);
    const double& operator [] (int) const;

    double length() const;
    double length_squared() const;

    static double distance(const Point&, const Point&);
    static double distance_squared(const Point&, const Point&);
    std::array<double, 3>::iterator begin();
    std::array<double, 3>::const_iterator begin() const;
    std::array<double, 3>::iterator end();
    std::array<double, 3>::const_iterator end() const;
    std::array<double, 3>& data();
    const std::array<double, 3>& data() const;

  private:
    std::array<double, 3> p;
};

Point operator + (Point, const Point&);
Point operator - (Point, const Point&);

// std::array<double, 3> operator = (const Point&);

#endif // POINT_H_
