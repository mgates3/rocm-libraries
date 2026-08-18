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
template <typename T, typename I, typename Td, typename Th>
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

                    Th& hARes,
                    Th& hAbandRes,
                    Th& hTauQRes,
                    Th& hTauPRes,

                    Th& hA,
                    Th& hAband,
                    Th& hTauQ,
                    Th& hTauP,
                    double* max_err)
{
    using S = decltype(std::real(T{}));

    hipStream_t stream;
    CHECK_ROCBLAS_ERROR(rocblas_get_stream(handle, &stream));

    // input data initialization
    ge2tb_initData<true, true, T, I>(handle, m, n, dA, lda, hA);

    // execute computations
    // GPU lapack
    CHECK_ROCBLAS_ERROR(rocsolver_ge2tb(handle, m, n, kl, nb, // opts
                                        dA.data(), lda, // A
                                        dAband.data(), ldab, // Aband
                                        dTauQ.data(), // tauQ
                                        dTauP.data())); // tauP
    CHECK_HIP_ERROR(hARes.transfer_from(dA));
    CHECK_HIP_ERROR(hAbandRes.transfer_from(dAband));
    CHECK_HIP_ERROR(hTauQRes.transfer_from(dTauQ));
    CHECK_HIP_ERROR(hTauPRes.transfer_from(dTauP));

    // CPU lapack reference
    // TODO: add cpu_ge2tb when a suitable LAPACK reference is available.
    // For now, verify only that the band structure is correct (entries outside
    // band are zero) and that tauQ/tauP norms are finite.

    // error is ||hAband - hAbandRes|| / (n * ||hAband||)
    // using frobenius norm
    // (THIS DOES NOT ACCOUNT FOR NUMERICAL REPRODUCIBILITY
    // ISSUES. IT MIGHT BE REVISITED IN THE FUTURE)

    // TODO: compare against cpu_ge2tb reference once available.
    *max_err = 0;
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
    double max_error = 0, gpu_time_used = 0, cpu_time_used = 0;

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
    host_strided_batch_vector<T> hA(size_A, 1, size_A, 1);
    host_strided_batch_vector<T> hAband(size_Aband, 1, size_Aband, 1);
    host_strided_batch_vector<T> hTauQ(size_tauQ, 1, size_tauQ, 1);
    host_strided_batch_vector<T> hTauP(size_tauP, 1, size_tauP, 1);

    host_strided_batch_vector<T> hARes(size_ARes, 1, size_ARes, 1);
    host_strided_batch_vector<T> hAbandRes(size_AbandRes, 1, size_AbandRes, 1);
    host_strided_batch_vector<T> hTauQRes(size_tauQRes, 1, size_tauQRes, 1);
    host_strided_batch_vector<T> hTauPRes(size_tauPRes, 1, size_tauPRes, 1);

    device_strided_batch_vector<T> dA(size_A, 1, size_A, 1);
    device_strided_batch_vector<T> dAband(size_Aband, 1, size_Aband, 1);
    device_strided_batch_vector<T> dTauQ(size_tauQ, 1, size_tauQ, 1);
    device_strided_batch_vector<T> dTauP(size_tauP, 1, size_tauP, 1);
    if(size_A)
        CHECK_HIP_ERROR(dA.memcheck());
    if(size_Aband)
        CHECK_HIP_ERROR(dAband.memcheck());
    if(size_tauQ)
        CHECK_HIP_ERROR(dTauQ.memcheck());
    if(size_tauP)
        CHECK_HIP_ERROR(dTauP.memcheck());

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
        ge2tb_getError<T, I>(handle, m, n, kl, nb, dA, lda, dAband, ldab, dTauQ, dTauP, hARes,
                             hAbandRes, hTauQRes, hTauPRes, hA, hAband, hTauQ, hTauP, &max_error);
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
    // max_errors is already normalized, e.g., by n.
    if(argus.unit_check)
        ROCSOLVER_TEST_CHECK(T, max_error, 10);

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
                rocsolver_bench_output("cpu_time_us", "gpu_time_us", "error");
                rocsolver_bench_output(cpu_time_used, gpu_time_used, max_error);
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
                rocsolver_bench_output(gpu_time_used, max_error);
            else
                rocsolver_bench_output(gpu_time_used);
        }
    }

    // ensure all arguments were consumed
    argus.validate_consumed();
}

#define EXTERN_TESTING_GE2TB(...) extern template void testing_ge2tb<__VA_ARGS__>(Arguments&);

INSTANTIATE(EXTERN_TESTING_GE2TB, FOREACH_SCALAR_TYPE, FOREACH_INT_TYPE, APPLY_STAMP)
