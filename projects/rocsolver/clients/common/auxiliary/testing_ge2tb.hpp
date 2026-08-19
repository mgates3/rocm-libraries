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

#pragma once

#include "common/misc/client_util.hpp"
#include "common/misc/clientcommon.hpp"
#include "common/misc/generate.hpp"
#include "common/misc/lapack_host_reference.hpp"
#include "common/misc/norm.hpp"
#include "common/misc/rocsolver.hpp"
#include "common/misc/rocsolver_arguments.hpp"
#include "common/misc/rocsolver_test.hpp"
#include "common/misc/rocsolver_timer.hpp"
#include "rocblas_utility.hpp"

//------------------------------------------------------------------------------
template <typename T, typename I>
void ge2tb_checkBadArgs(const rocblas_handle handle,
                        const I m,
                        const I n,
                        const I kl,
                        const I nb,
                        T dA,
                        const I lda,
                        T dAband,
                        const I ldab,
                        T dTauQ,
                        T dTauP)
{
    // handle
    EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(nullptr, m, n, kl, nb, dA, lda, dAband, ldab, dTauQ, dTauP),
                          rocblas_status_invalid_handle);

    // pointers
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ge2tb(handle, m, n, kl, nb, (T) nullptr, lda, dAband, ldab, dTauQ, dTauP),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ge2tb(handle, m, n, kl, nb, dA, lda, (T) nullptr, ldab, dTauQ, dTauP),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ge2tb(handle, m, n, kl, nb, dA, lda, dAband, ldab, (T) nullptr, dTauP),
        rocblas_status_invalid_pointer);
    EXPECT_ROCBLAS_STATUS(
        rocsolver_ge2tb(handle, m, n, kl, nb, dA, lda, dAband, ldab, dTauQ, (T) nullptr),
        rocblas_status_invalid_pointer);

    // quick return with invalid pointers
    EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(handle, I(0), I(0), kl, nb, (T) nullptr, lda, (T) nullptr,
                                          ldab, (T) nullptr, (T) nullptr),
                          rocblas_status_success);
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_ge2tb_bad_arg()
{
    // safe arguments
    rocblas_local_handle handle;
    I m = 1;
    I n = 1;
    I kl = 1;
    I nb = 1;
    I lda = 1;
    I ldab = 3; // 2*kl + 1

    // memory allocations
    device_strided_batch_vector<T> dA(lda, 1, lda, 1);
    device_strided_batch_vector<T> dAband(ldab, 1, ldab, 1);
    device_strided_batch_vector<T> dTauQ(n, 1, n, 1);
    device_strided_batch_vector<T> dTauP(n - kl, 1, n - kl, 1);
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dAband.memcheck());
    CHECK_HIP_ERROR(dTauQ.memcheck());
    CHECK_HIP_ERROR(dTauP.memcheck());

#ifndef ROCSOLVER_ENABLE_SVD_2STAGE
    // ge2tb is gated behind ROCSOLVER_ENABLE_SVD_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(handle, m, n, kl, nb, dA.data(), lda, dAband.data(), ldab,
                                          dTauQ.data(), dTauP.data()),
                          rocblas_status_not_implemented);
    return;
#endif

    // check bad arguments
    ge2tb_checkBadArgs(handle, m, n, kl, nb, dA.data(), lda, dAband.data(), ldab, dTauQ.data(),
                       dTauP.data());
}

//------------------------------------------------------------------------------
template <bool CPU, bool GPU, typename T, typename I, typename Td, typename Th>
void ge2tb_initData(const rocblas_handle handle, const I m, const I n, Td& dA, const I lda, Th& hA)
{
    if(CPU)
    {
        gerand(m, n, hA[0], lda);
    }

    if(GPU)
    {
        // now copy to the GPU
        CHECK_HIP_ERROR(dA.transfer_from(hA));
    }
}

//------------------------------------------------------------------------------
// m        -- number of rows (m >= n)
// n        -- number of columns
// kl       -- desired lower bandwidth
// nb       -- outer blocksize to use
//
// dA       -- matrix on GPU, lda-by-n, lda >= m
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 2*kl + 1
// dTauQ    -- output left  Householder taus on GPU, length n
// dTauP    -- output right Householder taus on GPU, length n - kl
// dQ       -- scratch m-by-n matrix on GPU; holds explicit Q or P (ldq >= m)
// dR       -- scratch n-by-n matrix on GPU for I - Q^H Q or I - P^H P (ldr >= n)
// dnorm    -- scratch length-1 device buffer for norms
//
// hARes    -- output matrix on CPU to copy GPU result, lda-by-n
// hAbandRes-- output band matrix on CPU to copy GPU result, ldab-by-n
// hTauQRes -- output left  taus on CPU to copy GPU result, length n
// hTauPRes -- output right taus on CPU to copy GPU result, length n - kl
//
// hA       -- matrix on CPU, lda-by-n, lda >= m
// hAband   -- output band matrix on CPU, ldab-by-n, ldab >= kl + 1
// hTauQ    -- output left  taus on CPU, length n
// hTauP    -- output right taus on CPU, length n - kl
//
// errors[0] -- || Q^H A_orig P - Aband ||_1 / (n * || A_orig ||_1)
// errors[1] -- || I_n - Q^H Q ||_1 / n   (Q is m-by-n reduced)
// errors[2] -- || I_n - P^H P ||_1 / n   (P is n-by-n)
//
template <typename T, typename I, typename Td, typename Th, typename Sd, typename Sh>
void ge2tb_getError(const rocblas_handle handle,
                    const I m,
                    const I n,
                    const I kl,
                    const I nb,

                    Td& dA,
                    const I lda,
                    Td& dAband,
                    const I ldab,
                    Td& dTauQ,
                    Td& dTauP,
                    Td& dQ,
                    const I ldq,
                    Td& dR,
                    const I ldr,
                    Sd& dnorm,

                    Th& hARes,
                    Th& hAbandRes,
                    Th& hTauQRes,
                    Th& hTauPRes,

                    Th& hA,
                    Th& hAband,
                    Th& hTauQ,
                    Th& hTauP,
                    Sh& hnorm,

                    double errors[3])
{
    using S = decltype(std::real(T{}));
    constexpr bool COMPLEX = rocblas_is_complex<T>;

    const T one = 1;
    const T negone = -1;
    const T zero = 0;
    const rocblas_stride stride = 0;

    const rocblas_operation conjTrans
        = COMPLEX ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose;

    rocblas_int nQ = (rocblas_int)n; // number of left  reflectors (Q)
    rocblas_int nP = std::max((rocblas_int)(n - kl), 0); // number of right reflectors (P)

    // input data initialization: upload A_orig to dA, keep CPU copy in hA
    ge2tb_initData<true, true, T, I>(handle, m, n, dA, lda, hA);

    // Compute 1-norm of A_orig on the GPU before ge2tb overwrites it.
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, rocsolver_norm_type_one, (rocblas_int)m, (rocblas_int)n,
                                        dA.data(), (rocblas_int)lda, dnorm.data()));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    S A_norm = hnorm[0][0];

    // Run GPU ge2tb; results land in dA (reflectors + upper triangle for P),
    // dAband (band entries), dTauQ, dTauP.
    CHECK_ROCBLAS_ERROR(rocsolver_ge2tb(handle, m, n, kl, nb, dA.data(), lda, dAband.data(), ldab,
                                        dTauQ.data(), dTauP.data()));
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hAbandRes.transfer_from(dAband));
    CHECK_HIP_ERROR(hTauQRes.transfer_from(dTauQ));
    CHECK_HIP_ERROR(hTauPRes.transfer_from(dTauP));

    //--------------------
    // Check 0: || Q^H * A_orig * P - Aband ||_1 / (n * || A_orig ||_1)
    //
    // Q is stored as a lower-trapezoidal matrix of Householder vectors below
    // sub-diagonal kl in the output dA (column j: unit entry at row kl+j,
    // nonzero below).  tauQ[j] is the corresponding scalar.
    //
    // P is stored as an upper-triangular matrix of Householder vectors in the
    // upper triangle of dA (LQ storage: row j, unit at column j, nonzero
    // to the right).  tauP[j] is the corresponding scalar.
    //
    // Use dR (m-by-n portion) as scratch for Q^H * A_orig * P:
    //   Step 1: upload A_orig into dR.
    //   Step 2: dR[kl:m, :] <- Q^H * dR[kl:m, :]  (left, ConjTrans)
    //   Step 3: dR <- dR * P                        (right, NoTrans)
    //   Step 4: transfer dR to CPU, subtract Aband, compute 1-norm of residual.

    // Step 1
    CHECK_HIP_ERROR(dR.transfer_from(hA));

    // Step 2: dR[kl:m, :] <- Q^H * dR[kl:m, :]
    CHECK_ROCBLAS_ERROR(rocsolver_ormxr_unmxr(true, // MQR=true -> blocked ormqr
                                              handle, rocblas_side_left, conjTrans,
                                              (rocblas_int)(m - kl), // rows of C
                                              (rocblas_int)n, // cols of C
                                              nQ, // number of reflectors
                                              dA.data() + kl, // Q vectors (shifted past kl rows)
                                              (rocblas_int)lda, dTauQ.data(),
                                              dR.data() + kl, // C = dR[kl:m, :]
                                              (rocblas_int)ldr));

    // Step 3: dR <- dR * P
    CHECK_ROCBLAS_ERROR(rocsolver_ormlx_unmlx(true, // MLQ=true -> blocked ormlq
                                              handle, rocblas_side_right,
                                              rocblas_operation_none,
                                              (rocblas_int)m, // rows
                                              (rocblas_int)n, // cols
                                              nP, // number of reflectors
                                              dA.data(), // P vectors (upper triangle, LQ)
                                              (rocblas_int)lda, dTauP.data(),
                                              dR.data(), // full m x n matrix
                                              (rocblas_int)ldr));

    // Step 4: transfer to CPU, subtract Aband, compute 1-norm manually.
    //
    // Aband_dense[j, i] = hAbandRes[ (kl-(i-j)) + i*ldab ]  for j in [i-kl, i].
    //
    // TODO: perform the subtraction dR -= Aband_dense on the GPU.
    Th hR(ldr * std::max((I)m, n), 1, ldr * std::max((I)m, n), 1);
    CHECK_HIP_ERROR(hR.transfer_from(dR));
    S res_norm_1 = 0;
    for(rocblas_int i = 0; i < (rocblas_int)n; i++)
    {
        S col_sum = 0;
        for(rocblas_int j = 0; j < (rocblas_int)n; j++)
        {
            T xij = hR[0][j + i * (rocblas_int)ldr];
            if(j >= i - (rocblas_int)kl && j <= i)
            {
                rocblas_int br = (rocblas_int)kl - (i - j);
                xij -= hAbandRes[0][br + i * (rocblas_int)ldab];
            }
            col_sum += std::abs(xij);
        }
        res_norm_1 = std::max(res_norm_1, col_sum);
    }
    errors[0] = (double)res_norm_1 / (double)n;
    if(A_norm != 0)
        errors[0] /= (double)A_norm;

    //--------------------
    // Check 1: || I_n - Q^H Q ||_1 / n
    //
    // Q is the reduced (m-by-n) left factor, stored as Householder reflectors in
    // the lower part of dA below sub-diagonal kl.  Generate the explicit (m-kl)-by-n
    // matrix Q2 (the non-trivial rows of Q) in-place:
    //   copy dA into dQ, then call orgxr_ungxr on dQ[kl:, :].
    // The full Q = [ I_kl | 0; 0 | Q2 ] so Q^H Q = I_kl^H I_kl + Q2^H Q2 = I_n
    // iff Q2^H Q2 = I_{n-kl,n} block + I_kl,kl block -- simpler to check Q^H Q - I
    // using just Q2 (k=m-kl rows) in the gemm: Q^H Q = Q2^H Q2 + I_kl (top-left block).
    // We compute the full n-by-n product directly using all m rows of Q:
    //   top kl rows of Q are the first kl standard basis vectors e_0..e_{kl-1},
    //   so Q^H Q[i,j] = delta_{ij} for i<kl plus Q2^H Q2 contribution.
    // Easiest: copy hA into dQ, run orgxr_ungxr on full dQ (m-by-n, kl rows treated
    // as identity via the reflector storage which has zeros there), then gemm.
    //
    // Build a host identity matrix for dR = I_n, transfer to device, then
    // compute dR = I_n - Q^H Q via gemm.
    CHECK_HIP_ERROR(dQ.transfer_from(hA)); // dQ = A_orig (reflectors intact)
    CHECK_ROCBLAS_ERROR(rocsolver_orgxr_ungxr(true, // GQR=true -> blocked ungqr
                                              handle,
                                              (rocblas_int)(m - kl), // rows of the (m-kl)-by-n submatrix
                                              (rocblas_int)n, // cols
                                              nQ, // number of reflectors
                                              dQ.data() + kl, // reflector storage below sub-diag kl
                                              (rocblas_int)ldq, dTauQ.data()));
    // dQ[0:kl, :] still holds original A data; those rows of Q are e_0..e_{kl-1}.
    // Zero them out on CPU scratch to form the full explicit Q, then re-upload.
    // (The first kl rows contribute only delta_{i,j} for i<kl to Q^H Q = I.)
    // Since those rows are just standard basis vectors, we can handle them on CPU:
    Th hQ(ldq * n, 1, ldq * n, 1);
    CHECK_HIP_ERROR(hQ.transfer_from(dQ));
    // Set rows 0..kl-1 of hQ: row i = e_i (zero except column i = 1)
    for(rocblas_int i = 0; i < (rocblas_int)kl; i++)
        for(rocblas_int j = 0; j < (rocblas_int)n; j++)
            hQ[0][i + j * (rocblas_int)ldq] = (j == i) ? one : zero;
    CHECK_HIP_ERROR(dQ.transfer_from(hQ));

    // Build n-by-n identity in dR on the CPU and upload.
    Th hIdentN(ldr * n, 1, ldr * n, 1);
    for(rocblas_int j = 0; j < (rocblas_int)n; j++)
        for(rocblas_int i = 0; i < (rocblas_int)n; i++)
            hIdentN[0][i + j * (rocblas_int)ldr] = (i == j) ? one : zero;
    CHECK_HIP_ERROR(dR.transfer_from(hIdentN));

    // dR = I_n - Q^H Q
    T alpha_neg = negone, beta_one = one;
    CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle,
                                       conjTrans, rocblas_operation_none,
                                       (rocblas_int)n, (rocblas_int)n, (rocblas_int)m,
                                       &alpha_neg, dQ.data(), (rocblas_int)ldq, stride,
                                       dQ.data(), (rocblas_int)ldq, stride,
                                       &beta_one, dR.data(), (rocblas_int)ldr, stride, 1));
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, rocsolver_norm_type_one,
                                        (rocblas_int)n, (rocblas_int)n,
                                        dR.data(), (rocblas_int)ldr, dnorm.data()));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    errors[1] = (double)hnorm[0][0] / (double)n;

    //--------------------
    // Check 2: || I_n - P^H P ||_1 / n
    //
    // P is n-by-n, stored as Householder reflectors in the upper triangle of dA
    // (LQ format).  Generate the explicit n-by-n P via orglx_unglx (blocked unglq):
    //   copy dA (n rows, n cols) into the top n rows of dQ, then call orglx_unglx.
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));
    CHECK_HIP_ERROR(hipMemcpy2DAsync(dQ[0], ldq * sizeof(T), // dst: dQ col-major, ldq rows
                                     dA[0], lda * sizeof(T), // src: dA col-major, lda rows
                                     n * sizeof(T), n, // copy n rows x n cols
                                     hipMemcpyDefault, stream));
    CHECK_ROCBLAS_ERROR(rocsolver_orglx_unglx(true, // GLQ=true -> blocked unglq
                                              handle,
                                              (rocblas_int)n, // rows
                                              (rocblas_int)n, // cols
                                              nP, // number of reflectors
                                              dQ.data(), (rocblas_int)ldq, dTauP.data()));

    // dR = I_n - P^H P
    CHECK_HIP_ERROR(dR.transfer_from(hIdentN));
    CHECK_ROCBLAS_ERROR(rocsolver_gemm(false, handle,
                                       conjTrans, rocblas_operation_none,
                                       (rocblas_int)n, (rocblas_int)n, (rocblas_int)n,
                                       &alpha_neg, dQ.data(), (rocblas_int)ldq, stride,
                                       dQ.data(), (rocblas_int)ldq, stride,
                                       &beta_one, dR.data(), (rocblas_int)ldr, stride, 1));
    CHECK_ROCBLAS_ERROR(rocsolver_lange(handle, rocsolver_norm_type_one,
                                        (rocblas_int)n, (rocblas_int)n,
                                        dR.data(), (rocblas_int)ldr, dnorm.data()));
    CHECK_HIP_ERROR(hnorm.transfer_from(dnorm));
    errors[2] = (double)hnorm[0][0] / (double)n;
}

//------------------------------------------------------------------------------
// m        -- number of rows (m >= n)
// n        -- number of columns
// kl       -- desired lower bandwidth
// nb       -- outer blocksize to use
// dA       -- matrix on GPU, lda-by-n, lda >= m
// dAband   -- output band matrix on GPU, ldab-by-n, ldab >= 2*kl + 1
template <typename T, typename I, typename Td, typename Th>
void ge2tb_getPerfData(const rocblas_handle handle,
                       const I m,
                       const I n,
                       const I kl,
                       const I nb,
                       Td& dA,
                       const I lda,
                       Td& dAband,
                       const I ldab,
                       Td& dTauQ,
                       Td& dTauP,
                       Th& hA,
                       Th& hAband,
                       Th& hTauQ,
                       Th& hTauP,
                       double* gpu_time_used,
                       double* cpu_time_used,
                       const rocblas_int hot_calls,
                       const int profile,
                       const bool profile_kernels,
                       const bool perf)
{
    rocsolver_timer timer;
    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    ge2tb_initData<true, false, T, I>(handle, m, n, dA, lda, hA);

    if(!perf)
    {
        // TODO: cpu_ge2tb reference timing once available.
        *cpu_time_used = nan("");
    }

    // cold calls
    for(int iter = 0; iter < 2; iter++)
    {
        ge2tb_initData<false, true, T, I>(handle, m, n, dA, lda, hA);

        CHECK_ROCBLAS_ERROR(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                            dA.data(), lda, // A
                                            dAband.data(), ldab, // Aband
                                            dTauQ.data(), // tauQ
                                            dTauP.data())); // tauP
    }

    // gpu-lapack performance
    if(profile > 0)
    {
        if(profile_kernels)
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile
                                         | rocblas_layer_mode_ex_log_kernel);
        else
            rocsolver_log_set_layer_mode(rocblas_layer_mode_log_profile);
        rocsolver_log_set_max_levels(profile);
    }

    for(rocblas_int iter = 0; iter < hot_calls; iter++)
    {
        ge2tb_initData<false, true, T, I>(handle, m, n, dA, lda, hA);

        timer.start(stream);
        rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                        dA.data(), lda, // A
                        dAband.data(), ldab, // Aband
                        dTauQ.data(), // tauQ
                        dTauP.data()); // tauP
        timer.end(stream);
    }
    *gpu_time_used = timer.get_combined();
}

//------------------------------------------------------------------------------
template <typename T, typename I>
void testing_ge2tb(Arguments& argus)
{
    // get arguments
    rocblas_local_handle handle;
    I m = argus.get<I>("m");
    I n = argus.get<I>("n", m);
    I kl = argus.get<I>("kl", 1);
    I nb = argus.get<I>("nb", kl);
    I lda = argus.get<I>("lda", m);
    // band storage: kl sub-diagonals + diagonal, with extra for 2nd stage
    I ldab = 2 * kl + 1;

    rocblas_int hot_calls = argus.iters;

#ifndef ROCSOLVER_ENABLE_SVD_2STAGE
    // ge2tb is gated behind ROCSOLVER_ENABLE_SVD_2STAGE; when the flag is off the
    // entry points must report rocblas_status_not_implemented instead of running.
    EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                          (T*)nullptr, lda, // A
                                          (T*)nullptr, ldab, // Aband
                                          (T*)nullptr, // tauQ
                                          (T*)nullptr), // tauP
                          rocblas_status_not_implemented);
    return;
#endif

    // check non-supported values
    // (none currently; all four data types supported)

    // determine sizes
    size_t size_A = lda * n;
    size_t size_Aband = ldab * n;
    size_t size_tauQ = std::max<I>(n, 0);
    size_t size_tauP = std::max<I>(n - kl, 0);
    // dQ: m-by-n for Q orthogonality (ldq = m); also reused as n-by-n for P orthogonality.
    I ldq = m;
    I ldr = std::max(m, n);
    size_t size_Q = ldq * n; // m-by-n covers both Q (m-by-n) and P (n-by-n, since n<=m)
    size_t size_R = ldr * ldr; // max(m,n)-by-max(m,n) scratch
    size_t size_norm = 1;
    double errors[3] = {0, 0, 0}, gpu_time_used = 0, cpu_time_used = 0;

    size_t size_ARes = (argus.unit_check || argus.norm_check) ? size_A : 0;
    size_t size_AbandRes = (argus.unit_check || argus.norm_check) ? size_Aband : 0;
    size_t size_tauQRes = (argus.unit_check || argus.norm_check) ? size_tauQ : 0;
    size_t size_tauPRes = (argus.unit_check || argus.norm_check) ? size_tauP : 0;

    // check invalid sizes
    bool invalid_size = (m < 0 || n < 0 || m < n || kl < 1 || nb < 1 || nb < kl || nb % kl != 0
                         || lda < m || ldab < 2 * kl + 1);
    if(invalid_size)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                              (T*)nullptr, lda, // A
                                              (T*)nullptr, ldab, // Aband
                                              (T*)nullptr, // tauQ
                                              (T*)nullptr), // tauP
                              rocblas_status_invalid_size);

        if(argus.timing)
            rocsolver_bench_inform(inform_invalid_size);

        return;
    }

    // memory size query is necessary
    if(argus.mem_query)
    {
        CHECK_ROCBLAS_ERROR(rocblas_start_device_memory_size_query(handle));
        CHECK_ALLOC_QUERY(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                          (T*)nullptr, lda, // A
                                          (T*)nullptr, ldab, // Aband
                                          (T*)nullptr, // tauQ
                                          (T*)nullptr)); // tauP

        size_t size;
        CHECK_ROCBLAS_ERROR(rocblas_stop_device_memory_size_query(handle, &size));

        rocsolver_bench_inform(inform_mem_query, size);
        return;
    }

    // memory allocations
    using S = decltype(std::real(T{}));
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<T> hTauQ(size_tauQ, 1, size_tauQ, 1);
    host_strided_batch_vector<T> hTauP(size_tauP, 1, size_tauP, 1);
    host_strided_batch_vector<S> hnorm(size_norm, 1, size_norm, 1);

    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAbandRes(size_AbandRes, 1, size_AbandRes, 1);
    host_strided_batch_vector<T> hTauQRes(size_tauQRes, 1, size_tauQRes, 1);
    host_strided_batch_vector<T> hTauPRes(size_tauPRes, 1, size_tauPRes, 1);

    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dTauQ(size_tauQ, 1, size_tauQ, 1);
    device_strided_batch_vector<T> dTauP(size_tauP, 1, size_tauP, 1);
    device_strided_batch_vector<T> dQ(size_Q, 1, size_Q, 1);
    device_strided_batch_vector<T> dR(size_R, 1, size_R, 1);
    device_strided_batch_vector<S> dnorm(size_norm, 1, size_norm, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_tauQ)
        CHECK_HIP_ERROR(dTauQ.memcheck());
    if(size_tauP)
        CHECK_HIP_ERROR(dTauP.memcheck());
    if(size_Q)
        CHECK_HIP_ERROR(dQ.memcheck());
    if(size_R)
        CHECK_HIP_ERROR(dR.memcheck());
    if(size_norm)
        CHECK_HIP_ERROR(dnorm.memcheck());

    // check quick return
    if(m == 0 || n == 0)
    {
        EXPECT_ROCBLAS_STATUS(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                              dA.data(), lda, // A
                                              dAband.data(), ldab, // Aband
                                              dTauQ.data(), // tauQ
                                              dTauP.data()), // tauP
                              rocblas_status_success);
        if(argus.timing)
            rocsolver_bench_inform(inform_quick_return);

        return;
    }

    // check computations
    if(argus.unit_check || argus.norm_check)
    {
        ge2tb_getError<T, I>(handle, m, n, kl, nb,
                             dA, lda, dAband, ldab, dTauQ, dTauP, dQ, ldq, dR, ldr, dnorm,
                             hARes, hAbandRes, hTauQRes, hTauPRes,
                             hA, hAband, hTauQ, hTauP, hnorm,
                             errors);
    }

    // collect performance data
    if(argus.timing && hot_calls > 0)
    {
        ge2tb_getPerfData<T, I>(handle, m, n, kl, nb, dA, lda, dAband, ldab, dTauQ, dTauP, hA,
                                hAband, hTauQ, hTauP, &gpu_time_used, &cpu_time_used, hot_calls,
                                argus.profile, argus.profile_kernels, argus.perf);
    }

    // validate results for rocsolver-test
    // using 10*machine_precision as tolerance
    // errors are already normalized (by n or m).
    if(argus.unit_check)
    {
        ROCSOLVER_TEST_CHECK(T, errors[0], 10);
        ROCSOLVER_TEST_CHECK(T, errors[1], 10);
        ROCSOLVER_TEST_CHECK(T, errors[2], 10);
    }

    // output results for rocsolver-bench
    if(argus.timing)
    {
        if(!argus.perf)
        {
            rocsolver_bench_header("Arguments:");
            rocsolver_bench_output("m", "n", "kl", "nb", "lda");
            rocsolver_bench_output(m, n, kl, nb, lda);
            rocsolver_bench_header("Results:");
            if(argus.norm_check)
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us",
                                       "berror Q^H*A*P-B", "ortho I-Q^HQ", "ortho I-P^HP");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, errors[0], errors[1], errors[2]);
            }
            else
            {
                rocsolver_bench_output("cpu_time_us", "gpu_time_us");
                rocsolver_bench_output(cpu_time_used, gpu_time_used);
            }
            rocsolver_bench_endl();
        }
        else
        {
            if(argus.norm_check)
                rocsolver_bench_output(gpu_time_used, errors[0], errors[1], errors[2]);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_GE2TB(...) extern template void testing_ge2tb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GE2TB, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
