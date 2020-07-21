#ifndef CARTESIAN_POLY_H_
#define CARTESIAN_POLY_H_
namespace Cartesian_Poly {
  enum Cartesian_P {
    X = 0, 
    Y,
    Z
  };
  enum Cartesian_D {
    XX = 0,
    XY,
    XZ,
    YY,
    YZ,
    ZZ
  };
  enum Cartesian_F {
    XXX = 0,
    XXY,
    XXZ,
    XYY,
    XYZ,
    XZZ,
    YYY,
    YYZ,
    YZZ,
    ZZZ
  };
  enum Cartesian_G {
    XXXX = 0,
    XXXY,
    XXXZ,
    XXYY,
    XXYZ,
    XXZZ,
    XYYY,
    XYYZ,
    XYZZ,
    XZZZ,
    YYYY,
    YYYZ,
    YYZZ,
    YZZZ,
    ZZZZ
  };
  enum Cartesian_H {
    XXXXX = 0,
    XXXXY,
    XXXXZ,
    XXXYY,
    XXXYZ,
    XXXZZ,
    XXYYY,
    XXYYZ,
    XXYZZ,
    XXZZZ,
    XYYYY,
    XYYYZ,
    XYYZZ,
    XYZZZ,
    XZZZZ,
    YYYYY,
    YYYYZ,
    YYYZZ,
    YYZZZ,
    YZZZZ,
    ZZZZZ,
  };
}
#endif  // CARTESIAN_POLY_H_
