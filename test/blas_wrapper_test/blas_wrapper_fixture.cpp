#include "blas_wrapper_fixture.h"

template <template <class, class> class Container, template <class> class Allocator>
Blas_Wrapper_Fixture<Container, Allocator>::Blas_Wrapper_Fixture() :
    m(10), n(10), 
    lda(20), ldb(30), ldc(40), 
    inc_x(1), inc_y(2), inc_z(3), 
    offset_a(1), offset_b(2), offset_c(3), 
    offset_x(0), offset_y(1), offset_z(2), 
    A(lda * n, 0.0), B(ldb * n, 0.0), C(ldc * n, 0.0),
    X(inc_x * n, 0.0), Y(inc_y * n, 0.0), Z(inc_z * n, 0.0)
{}
