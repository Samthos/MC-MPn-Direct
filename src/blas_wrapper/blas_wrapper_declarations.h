//
// Level 3 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgemm(
    bool TransA, bool TransB, 
    size_t m, size_t n, size_t k, 
    double alpha,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& B, size_t offset_b, size_t ldb,
    double beta,
    vector_double& C, size_t offset_c, size_t ldc);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dsyrk(
    BLAS_WRAPPER::Fill_Mode_t fill_mode_t, bool Trans, 
    size_t m, size_t k, 
    double alpha,
    const vector_double& A, size_t offset_a, size_t lda,
    double beta,
    vector_double& B, size_t offset_b, size_t ldb);

//
// Level 2 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::batched_ddot(size_t N, size_t K,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& B, size_t offset_b, size_t ldb,
    vector_double& X, size_t incx);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddgmm(
    bool right_side,
    size_t m, size_t n,
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    vector_double& B, size_t ldb);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgemv(
    bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_double& A, size_t offset_a, size_t lda,
    const vector_double& x, size_t offset_x, size_t incx,
    double beta,
    vector_double& y, size_t offset_y, size_t);

//
// Level 1 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dcopy(
    size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    vector_double& Y, size_t offset_y, size_t incy);

template <>
double Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddot(
    size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddot(size_t N, 
    const vector_double& X, size_t offset_x, size_t incx,
    const vector_double& Y, size_t offset_y, size_t incy, 
    double* result);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dscal(size_t N,
    double alpha,
    vector_double& X, size_t incx);

// 
// Iterators
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::fill(
    iterator first1, iterator last1, double value);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::minus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::multiplies(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::plus(
    const_iterator first1, const_iterator last1,
    const_iterator first2, iterator result);


template <> Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::Blas_Wrapper();
template <> Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::~Blas_Wrapper();
template class Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>;

