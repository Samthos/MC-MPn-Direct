//
// Level 3 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgemm(
    bool TransA, bool TransB, 
    size_t m, size_t n, size_t k, 
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    double beta,
    vector_type& C, size_t offset_c, size_t ldc);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dsyrk(
    BLAS_WRAPPER::Fill_Mode_t fill_mode_t, bool Trans, 
    size_t m, size_t k, 
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    double beta,
    vector_type& B, size_t offset_b, size_t ldb);

//
// Level 2 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::batched_ddot(size_t N, size_t K,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& X, size_t incx);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddgmm(
    BLAS_WRAPPER::Side_t side,
    size_t m, size_t n,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    vector_type& B, size_t offset_b, size_t ldb);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgeam(bool TransA, bool TransB,
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    double beta,
    const vector_type& B, size_t offset_b, size_t ldb,
    vector_type& C, size_t offset_c, size_t ldc);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgekm(bool TransA, bool TransB, 
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& B, size_t offset_b, size_t ldb,
    double beta,
    vector_type& C, size_t offset_c, size_t ldc);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgemv(
    bool Trans, 
    size_t m, size_t n,
    double alpha,
    const vector_type& A, size_t offset_a, size_t lda,
    const vector_type& x, size_t offset_x, size_t incx,
    double beta,
    vector_type& y, size_t offset_y, size_t);

//
// Level 1 blas
//
template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::daxpy(
    size_t N, 
    double alpha,
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dcopy(
    size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    vector_type& Y, size_t offset_y, size_t incy);

template <>
double Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddot(
    size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddot(size_t N, 
    const vector_type& X, size_t offset_x, size_t incx,
    const vector_type& Y, size_t offset_y, size_t incy, 
    double* result);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgekv(size_t m,
    value_type alpha,
    const vector_type& A, size_t offset_a, size_t inc_a,
    const vector_type& B, size_t offset_b, size_t inc_b,
    value_type beta,
    vector_type& C, size_t offset_c, size_t inc_c);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dscal(size_t N,
    double alpha,
    vector_type& X, size_t incx);

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

