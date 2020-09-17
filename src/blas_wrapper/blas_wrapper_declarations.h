template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dgemm(
    bool TransA, bool TransB, 
    size_t m, size_t n, size_t k, 
    double alpha,
    const vector_double& A, size_t lda,
    const vector_double& B, size_t ldb,
    double beta,
    vector_double& C, size_t ldc);

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
    const vector_double& A, size_t lda,
    const vector_double& x, size_t incx,
    double beta,
    vector_double& y, size_t);

template <>
double Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::ddot(
    size_t N, 
    const vector_double& X, size_t incx,
    const vector_double& Y, size_t incy);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::dscal(size_t N,
    double alpha,
    vector_double& X, size_t incx);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::multiplies(
    const vector_double& A, 
    const vector_double& B, 
    vector_double& C);

template <>
void Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::minus(
    const vector_double& A, 
    const vector_double& B, 
    vector_double& C);

template <> Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::Blas_Wrapper();
template <> Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>::~Blas_Wrapper();
template class Blas_Wrapper<VECTOR_TYPE, ALLOCATOR_TYPE>;

