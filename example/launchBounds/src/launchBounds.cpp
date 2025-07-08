/* Copyright 2025 Ryan Cross
 * SPDX-License-Identifier: ISC
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/ExecuteForEachAccTag.hpp>

#include <chrono>
#include <iostream>
#include <vector>

//! A kernel that performs some compute-intensive work.
//! We will run this kernel with and without launch bounds to compare performance.
struct ComputeKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(TAcc const& acc, float const* in, float* out, std::size_t size) const -> void
    {
        auto const globalThreadIdx = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0];
        if(globalThreadIdx < size)
        {
            // Do some arbitrary work to keep the GPU busy.
            // The effect of launch bounds is most visible on resource-constrained kernels
            // that can benefit from higher occupancy to hide memory latency.
            float val = 0.f;
            for(int i = 0; i < 32; ++i)
            {
                // Introduce more memory access to make the kernel more sensitive to latency
                auto idx = (globalThreadIdx + i * 1024) % size;
                val += alpaka::math::sin(acc, in[idx]);
            }
            out[globalThreadIdx] = val;
        }
    }
};

//! A separate kernel type that we will use to apply launch bounds.
//! It has the same implementation as ComputeKernel.
//! A kernel for testing maxThreadsPerBlock only.
struct ComputeKernelWithMaxThreadsOnly : ComputeKernel
{
};

//! A kernel for testing maxThreadsPerBlock and minBlocksPerMultiprocessor.
struct ComputeKernelWithMinBlocks : ComputeKernel
{
};

namespace alpaka::trait
{
    // --- Specializations for CUDA ---

    //! maxThreadsPerBlock only
    template<>
    struct KernelLaunchBounds<ComputeKernelWithMaxThreadsOnly, alpaka::TagGpuCudaRt>
    {
        static constexpr std::size_t maxThreadsPerBlock = 256;
    };

    //! maxThreadsPerBlock and minBlocksPerMultiprocessor
    template<>
    struct KernelLaunchBounds<ComputeKernelWithMinBlocks, alpaka::TagGpuCudaRt>
    {
        static constexpr std::size_t maxThreadsPerBlock = 128;
        static constexpr std::size_t minBlocksPerMultiprocessor = 8;
    };

    // --- Specializations for HIP ---

    //! maxThreadsPerBlock only
    template<>
    struct KernelLaunchBounds<ComputeKernelWithMaxThreadsOnly, alpaka::TagGpuHipRt>
    {
        static constexpr std::size_t maxThreadsPerBlock = 256;
    };

    //! maxThreadsPerBlock and minBlocksPerMultiprocessor
    template<>
    struct KernelLaunchBounds<ComputeKernelWithMinBlocks, alpaka::TagGpuHipRt>
    {
        static constexpr std::size_t maxThreadsPerBlock = 128;
        static constexpr std::size_t minBlocksPerMultiprocessor = 4;
    };
} // namespace alpaka::trait

template<alpaka::concepts::Tag TAccTag>
auto example(TAccTag const&) -> int
{
    using Dim = alpaka::DimInt<1>;
    using Idx = std::size_t;
    using Acc = alpaka::TagToAcc<TAccTag, Dim, Idx>;

    std::cout << "\n--- Using alpaka accelerator: " << alpaka::getAccName<Acc>() << " ---" << std::endl;

    // If this isn't CUDA, HIP or SYCL, we don't apply launch bounds, so skip the rest of the example.
    if constexpr(!alpaka::accMatchesTags<Acc, alpaka::TagGpuCudaRt, alpaka::TagGpuHipRt>)
    {
        std::cout << "Launch bounds are only applied to CUDA + HIP accelerators." << std::endl;
        return EXIT_SUCCESS;
    }

    auto const platformAcc = alpaka::Platform<Acc>{};
    if(alpaka::getDevCount(platformAcc) == 0)
    {
        std::cout << "No suitable device found for " << alpaka::getAccName<Acc>() << std::endl;
        return EXIT_SUCCESS;
    }
    auto const devAcc = alpaka::getDevByIdx(platformAcc, 0);
    alpaka::Queue<Acc, alpaka::Blocking> queue(devAcc);

    // Number of runs for averaging performance
    constexpr std::size_t numRuns = 10;

    // Setup data
    std::size_t const problemSize = 1024 * 1024 * 16;
    std::vector<float> hostSrc(problemSize, 1.5f);
    std::vector<float> hostDst(problemSize, 0.0f);

    auto bufSrc = alpaka::allocBuf<float, Idx>(devAcc, problemSize);
    auto bufDst = alpaka::allocBuf<float, Idx>(devAcc, problemSize);
    alpaka::memcpy(queue, bufSrc, hostSrc);

    // Define kernels
    ComputeKernel noBoundsKernel;
    ComputeKernelWithMaxThreadsOnly maxThreadsKernel;
    ComputeKernelWithMinBlocks minBlocksKernel;

    // Define work division configuration
    alpaka::KernelCfg<Acc> const kernelCfg = {problemSize, 1u};

    // Helper lambda to run and time a kernel
    auto runBenchmark = [&](auto const& kernel, std::string const& description) {
        auto const workDiv = alpaka::getValidWorkDiv(
            kernelCfg,
            devAcc,
            kernel,
            alpaka::getPtrNative(bufSrc),
            alpaka::getPtrNative(bufDst),
            problemSize);
        std::cout << "Executing " << description << ". WorkDiv: " << workDiv << std::endl;

        // Warm-up run
        alpaka::exec<Acc>(
            queue,
            workDiv,
            kernel,
            alpaka::getPtrNative(bufSrc),
            alpaka::getPtrNative(bufDst),
            problemSize);

        // Timed runs
        auto start = std::chrono::high_resolution_clock::now();
        for(std::size_t i = 0; i < numRuns; ++i)
        {
            alpaka::exec<Acc>(
                queue,
                workDiv,
                kernel,
                alpaka::getPtrNative(bufSrc),
                alpaka::getPtrNative(bufDst),
                problemSize);
        }
        alpaka::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = (end - start) / numRuns;
        std::cout << "Avg. Duration (" << description << "): " << duration.count() << " ms" << std::endl;
    };

    // --- Run benchmarks ---
    runBenchmark(noBoundsKernel, "without bounds");
    runBenchmark(maxThreadsKernel, "with maxThreads");
    runBenchmark(minBlocksKernel, "with minBlocks");

    return EXIT_SUCCESS;
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
