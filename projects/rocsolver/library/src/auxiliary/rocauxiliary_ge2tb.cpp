/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "rocauxiliary_ge2tb.hpp"
#include "exceptions.hpp"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
// Reduces an m-by-n general matrix A (m >= n) to an n-by-n upper triangular
// band matrix with kd superdiagonals, via unitary transformations:
//      Q^H A P = A_band.
//
//  handle      rocblas_handle.
//  m           Number of rows. m >= n >= 0.
//  n           Number of columns. n >= 0.
//  kd          Upper bandwidth. kd >= 1.
//  nb          Block size. nb >= kd and nb is a multiple of kd.
//  A           m-by-n general matrix.
//              On output, Householder vectors for Q overwrite the lower
//              trapezoid of A; Householder vectors for P overwrite A above
//              superdiagonal kd. The rest of A is destroyed.
//  lda         Leading dimension of A. lda >= m.
//  Aband       Band matrix storage, ldab-by-n. On output the main diagonal and
//              kd superdiagonals are set; other entries are destroyed.
//  ldab        Leading dimension of Aband. ldab >= kd + 1.
//  tauQ        Householder tau values for Q, length n.
//  tauP        Householder tau values for P, length n - kd.
//
template <typename T, typename I, typename U>
rocblas_status rocsolver_ge2tb_impl(rocblas_handle handle,
                                    const I m,
                                    const I n,
                                    const I kd,
                                    const I nb,
                                    U A,
                                    const I lda,
                                    T* Aband,
                                    const I ldab,
                                    T* tauQ,
                                    T* tauP)
try
{
    ROCSOLVER_ENTER_TOP("ge2tb", "-m", m, "-n", n, "-kd", kd, "-nb", nb, "--lda", lda, "--ldab",
                        ldab);

    if(!handle)
        return rocblas_status_invalid_handle;

    // argument checking
    rocblas_status st
        = rocsolver_ge2tb_argCheck(handle, m, n, kd, nb, A, lda, Aband, ldab, tauQ, tauP);
    if(st != rocblas_status_continue)
        return st;

    // working with unshifted arrays
    I shiftA = 0;

    // normal (non-batched non-strided) execution
    rocblas_stride strideA = 0;
    rocblas_stride strideAb = 0;
    rocblas_stride strideTauQ = 0;
    rocblas_stride strideTauP = 0;
    I batch_count = 1;

    // memory workspace sizes:
    // size for constants in rocblas calls
    size_t size_scalars;
    // size of arrays of pointers (for batched cases) and re-usable workspace
    size_t size_workArr;
    // extra requirements
    size_t size_D, size_V, size_W, size_X, size_Z, size_work;
    rocsolver_ge2tb_getMemorySize<false, T, I>(m, n, kd, nb, batch_count, &size_scalars, &size_D,
                                               &size_V, &size_W, &size_X, &size_Z, &size_work,
                                               &size_workArr);

    if(rocblas_is_device_memory_size_query(handle))
    {
        return rocblas_set_optimal_device_memory_size(handle, size_scalars, size_D, size_V, size_W,
                                                      size_X, size_Z, size_work, size_workArr);
    }

    // memory workspace allocation
    rocblas_device_malloc mem(handle, size_scalars, size_D, size_V, size_W, size_X, size_Z,
                              size_work, size_workArr);

    if(!mem)
        return rocblas_status_memory_error;

    T* scalars = (T*)mem[0];
    T* D = (T*)mem[1];
    T* V = (T*)mem[2];
    T* W = (T*)mem[3];
    T* X = (T*)mem[4];
    T* Z = (T*)mem[5];
    T* work = (T*)mem[6];
    T** workArr = (T**)mem[7];
    if(size_scalars > 0)
        init_scalars(handle, scalars);

    // execution
    return rocsolver_ge2tb_template<false, false, T, I>(handle, m, n, kd, nb, // opts
                                                        A, shiftA, lda, strideA, // A
                                                        Aband, ldab, strideAb, // Aband
                                                        tauQ, strideTauQ, // tauQ
                                                        tauP, strideTauP, // tauP
                                                        batch_count, scalars, D, V, W, X, Z, work,
                                                        workArr);
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" {

ROCSOLVER_EXPORT rocblas_status rocsolver_sge2tb(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 float* A,
                                                 const rocblas_int lda,
                                                 float* Aband,
                                                 const rocblas_int ldab,
                                                 float* tauQ,
                                                 float* tauP)
{
#ifdef ROCSOLVER_ENABLE_SVD_2STAGE
    return rocsolver::rocsolver_ge2tb_impl<float, rocblas_int>(handle, m, n, kd, nb, A, lda, Aband,
                                                               ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dge2tb(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 double* A,
                                                 const rocblas_int lda,
                                                 double* Aband,
                                                 const rocblas_int ldab,
                                                 double* tauQ,
                                                 double* tauP)
{
#ifdef ROCSOLVER_ENABLE_SVD_2STAGE
    return rocsolver::rocsolver_ge2tb_impl<double, rocblas_int>(handle, m, n, kd, nb, A, lda, Aband,
                                                                ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cge2tb(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 rocblas_float_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_float_complex* Aband,
                                                 const rocblas_int ldab,
                                                 rocblas_float_complex* tauQ,
                                                 rocblas_float_complex* tauP)
{
#ifdef ROCSOLVER_ENABLE_SVD_2STAGE
    return rocsolver::rocsolver_ge2tb_impl<rocblas_float_complex, rocblas_int>(
        handle, m, n, kd, nb, A, lda, Aband, ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zge2tb(rocblas_handle handle,
                                                 const rocblas_int m,
                                                 const rocblas_int n,
                                                 const rocblas_int kd,
                                                 const rocblas_int nb,
                                                 rocblas_double_complex* A,
                                                 const rocblas_int lda,
                                                 rocblas_double_complex* Aband,
                                                 const rocblas_int ldab,
                                                 rocblas_double_complex* tauQ,
                                                 rocblas_double_complex* tauP)
{
#ifdef ROCSOLVER_ENABLE_SVD_2STAGE
    return rocsolver::rocsolver_ge2tb_impl<rocblas_double_complex, rocblas_int>(
        handle, m, n, kd, nb, A, lda, Aband, ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_sge2tb_64(rocblas_handle handle,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    float* A,
                                                    const int64_t lda,
                                                    float* Aband,
                                                    const int64_t ldab,
                                                    float* tauQ,
                                                    float* tauP)
{
#if defined(ROCSOLVER_ENABLE_SVD_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_ge2tb_impl<float, int64_t>(handle, m, n, kd, nb, A, lda, Aband,
                                                           ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_dge2tb_64(rocblas_handle handle,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    double* A,
                                                    const int64_t lda,
                                                    double* Aband,
                                                    const int64_t ldab,
                                                    double* tauQ,
                                                    double* tauP)
{
#if defined(ROCSOLVER_ENABLE_SVD_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_ge2tb_impl<double, int64_t>(handle, m, n, kd, nb, A, lda, Aband,
                                                            ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_cge2tb_64(rocblas_handle handle,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    rocblas_float_complex* A,
                                                    const int64_t lda,
                                                    rocblas_float_complex* Aband,
                                                    const int64_t ldab,
                                                    rocblas_float_complex* tauQ,
                                                    rocblas_float_complex* tauP)
{
#if defined(ROCSOLVER_ENABLE_SVD_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_ge2tb_impl<rocblas_float_complex, int64_t>(
        handle, m, n, kd, nb, A, lda, Aband, ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

ROCSOLVER_EXPORT rocblas_status rocsolver_zge2tb_64(rocblas_handle handle,
                                                    const int64_t m,
                                                    const int64_t n,
                                                    const int64_t kd,
                                                    const int64_t nb,
                                                    rocblas_double_complex* A,
                                                    const int64_t lda,
                                                    rocblas_double_complex* Aband,
                                                    const int64_t ldab,
                                                    rocblas_double_complex* tauQ,
                                                    rocblas_double_complex* tauP)
{
#if defined(ROCSOLVER_ENABLE_SVD_2STAGE) && defined(HAVE_ROCBLAS_64)
    return rocsolver::rocsolver_ge2tb_impl<rocblas_double_complex, int64_t>(
        handle, m, n, kd, nb, A, lda, Aband, ldab, tauQ, tauP);
#else
    return rocblas_status_not_implemented;
#endif
}

} // extern C
