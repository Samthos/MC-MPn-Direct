#include "blas_wrapper_fixture.h"

template <template <class, class> class Container, template <class> class Allocator>
Blas_Wrapper_Fixture<Container, Allocator>::Blas_Wrapper_Fixture() :
    m(10), n(10), lda(20), ldb(20), ldc(20), offset_a(1), offset_b(2), offset_c(3), A(lda * n, 0.0), B(ldb * n, 0.0), C(ldc * n, 0.0) {}
