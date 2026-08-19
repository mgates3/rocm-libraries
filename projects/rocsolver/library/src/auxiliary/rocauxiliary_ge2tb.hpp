/************************************************************************
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

#pragma once

#include "lapack/roclapack_geqrf.hpp"
#include "lapack/roclapack_gerqf.hpp"
#include "lapack_device_functions.hpp"
#include "lib_device_helpers.hpp"
#include "rocblas.hpp"
#include "rocsolver/rocsolver.h"

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
template <bool BATCHED, typename T, typename I>
void rocsolver_ge2tb_getMemorySize(const I m,
                                   const I n,
                                   const I kd,
                                   const I nb,
                                   const I batch_count,
                                   size_t* size_scalars,
                                   size_t* size_D,
                                   size_t* size_V,
                                   size_t* size_W,
                                   size_t* size_X,
                                   size_t* size_Z,
                                   size_t* size_work,
                                   size_t* size_workArr)
{
    *size_scalars = 0;
    *size_D = 0;
    *size_V = 0;
    *size_W = 0;
    *size_X = 0;
    *size_Z = 0;
    *size_work = 0;
    *size_workArr = 0;

    // if quick return no workspace needed
    if(n == 0 || m == 0 || batch_count == 0 || kd >= n - 1)
        return;

    // TODO: compute workspace sizes for geqrf/gerqf calls and working arrays
    // modeled after rocsolver_sy2sb_he2hb_getMemorySize.

    *size_workArr = BATCHED ? sizeof(T*) * 2 * batch_count : 0;

    // main working arrays: D (nb x nb), V/W/X/Z (m x nb each)
    *size_D = sizeof(T) * nb * nb * batch_count;
    *size_V = sizeof(T) * m * nb * batch_count;
    *size_W = sizeof(T) * m * nb * batch_count;
    *size_X = sizeof(T) * m * nb * batch_count;
    *size_Z = sizeof(T) * m * nb * batch_count;

    size_t w, wa, s1, s2;

    // extra space for geqrf calls (left reflectors)
    rocsolver_geqrf_getMemorySize<BATCHED, T>(m - kd, kd, batch_count, size_scalars, &w, &s1, &s2,
                                              &wa);
    *size_D = std::max(*size_D, s1);
    *size_Z = std::max(*size_Z, s2);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);

    // extra space for larft calls
    rocsolver_larft_getMemorySize<BATCHED, T>(m - kd, nb, batch_count, size_scalars, &w, &wa);
    *size_work = std::max(*size_work, w);
    *size_workArr = std::max(*size_workArr, wa);
}

//------------------------------------------------------------------------------
template <typename T, typename I, typename U>
rocblas_status rocsolver_ge2tb_argCheck(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        const I kd,
                                        const I nb,
                                        U A,
                                        const I lda,
                                        T* Aband,
                                        const I ldab,
                                        T* tauQ,
                                        T* tauP,
                                        const I batch_count = 1)
{
    // order is important for unit tests:

    // 1. invalid size
    // m >= n is assumed for now; kd >= 1; nb >= kd and multiple of kd;
    // lda >= m; ldab >= kd + 1 (band storage: kd superdiags + diag)
    if(m < 0 || n < 0 || m < n || kd < 1 || nb < kd || nb % kd != 0 || lda < m || ldab < kd + 1
       || batch_count < 0)
        return rocblas_status_invalid_size;

    // skip pointer check if querying memory size
    if(rocblas_is_device_memory_size_query(handle))
        return rocblas_status_continue;

    // skip pointer check if quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_continue;

    // 2. invalid pointers
    if(!A || !Aband || !tauQ || !tauP)
        return rocblas_status_invalid_pointer;

    return rocblas_status_continue;
}

//------------------------------------------------------------------------------
// Implements ge2tb. See rocsolver_ge2tb_impl.
// scalars, D, V, W, X, Z, work, workArr are workspaces.
//
// Reduces an m-by-n general matrix A (m >= n) to an n-by-n upper triangular
// band matrix with kd superdiagonals via unitary transformations:
//      Q^H A P = A_band
// where Q is m-by-m unitary and P is n-by-n unitary.
// The kd-by-kd upper-left portion of P is the Identity.
//
// Householder reflectors for Q are stored in the lower trapezoid of A.
// Householder reflectors for P are stored above superdiagonal kd in A.
// tauQ holds the Householder tau values for Q, length n.
// tauP holds the Householder tau values for P, length n - kd.
//
template <bool BATCHED, bool STRIDED, typename T, typename I, typename U>
rocblas_status rocsolver_ge2tb_template(rocblas_handle handle,
                                        const I m,
                                        const I n,
                                        const I kd,
                                        const I nb,
                                        U A,
                                        const I shiftA,
                                        const I lda,
                                        const rocblas_stride strideA,
                                        T* Aband,
                                        const I ldab,
                                        const rocblas_stride strideAb,
                                        T* tauQ,
                                        const rocblas_stride strideTauQ,
                                        T* tauP,
                                        const rocblas_stride strideTauP,
                                        const I batch_count,
                                        T* scalars,
                                        T* D,
                                        T* V,
                                        T* W,
                                        T* X,
                                        T* Z,
                                        T* work,
                                        T** workArr)
{
    ROCSOLVER_ENTER("ge2tb", "m:", m, "n:", n, "kd:", kd, "nb:", nb, "shiftA:", shiftA, "lda:", lda,
                    "ldab:", ldab, "bc:", batch_count);

    // quick return
    if(m == 0 || n == 0 || batch_count == 0)
        return rocblas_status_success;

    hipStream_t stream;
    rocblas_get_stream(handle, &stream);

    // TODO: implement ge2tb algorithm.
    //
    // Outline (analogous to he2hb, but for a general m-by-n matrix):
    //
    // Zero out Aband.
    // Loop over outer blocks of size nb columns:
    //   Copy panel to working array V.
    //   Inner loop over kd-width sub-panels:
    //     Apply accumulated updates from previous sub-panels to V.
    //     Factor sub-panel with geqrf (left reflectors, stored in V and tauQ).
    //     Apply right reflectors (gerqf or gelqf) to zero entries above
    //       the superdiagonal, updating tauP.
    //     Copy band entries (diagonal tile + R factor) to Aband.
    //     Form compact WY representation T = larft(V, tauQ).
    //     Compute W = V T.
    //     Optionally update W with contributions from previous sub-panels.
    //     Compute X = A W (trailing matrix times W).
    //     Compute D = W^H X.
    //     Compute Z = X - 0.5 V D.
    //   Update trailing matrix: A -= V Z^H + Z V^H.
    //   Copy factored panel back to A.
    // Copy last diagonal block to Aband.

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE
