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
struct ComputeKernelWithBounds : ComputeKernel
{
};

namespace alpaka::trait
{
    // Specialize KernelLaunchBounds for CUDA accelerators.
    // This will apply __launch_bounds__(128, 8) to ComputeKernelWithBounds.
    // These values are a hint to the compiler. They can improve performance by increasing
    // occupancy, but can also hurt performance if they cause register spilling.
    // Finding the optimal values often requires experimentation.
    template<>
    struct KernelLaunchBounds<ComputeKernelWithBounds, alpaka::TagGpuCudaRt>
    {
        static constexpr std::size_t maxThreadsPerBlock = 128;
        static constexpr std::size_t minBlocksPerMultiprocessor = 8;
    };

    //! Specialize KernelLaunchBounds for HIP accelerators.
    template<>
    struct KernelLaunchBounds<ComputeKernelWithBounds, alpaka::TagGpuHipRt>
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
    ComputeKernelWithBounds withBoundsKernel;

    // Define work division configuration
    alpaka::KernelCfg<Acc> const kernelCfg = {problemSize, 1u};

    // --- Run without launch bounds ---
    auto const workDivNoBounds = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        noBoundsKernel,
        alpaka::getPtrNative(bufSrc),
        alpaka::getPtrNative(bufDst),
        problemSize);
    std::cout << "Executing without bounds. WorkDiv: " << workDivNoBounds << std::endl;

    // Warm-up run
    alpaka::exec<Acc>(
        queue,
        workDivNoBounds,
        noBoundsKernel,
        alpaka::getPtrNative(bufSrc),
        alpaka::getPtrNative(bufDst),
        problemSize);

    // Timed runs
    auto startNoBounds = std::chrono::high_resolution_clock::now();
    for(std::size_t i = 0; i < numRuns; ++i)
    {
        alpaka::exec<Acc>(
            queue,
            workDivNoBounds,
            noBoundsKernel,
            alpaka::getPtrNative(bufSrc),
            alpaka::getPtrNative(bufDst),
            problemSize);
    }
    alpaka::wait(queue);
    auto endNoBounds = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationNoBounds = (endNoBounds - startNoBounds) / numRuns;
    std::cout << "Avg. Duration (no bounds): " << durationNoBounds.count() << " ms" << std::endl;

    // --- Run with launch bounds ---
    auto const workDivWithBounds = alpaka::getValidWorkDiv(
        kernelCfg,
        devAcc,
        withBoundsKernel,
        alpaka::getPtrNative(bufSrc),
        alpaka::getPtrNative(bufDst),
        problemSize);
    std::cout << "Executing with bounds.    WorkDiv: " << workDivWithBounds << std::endl;

    // Warm-up run
    alpaka::exec<Acc>(
        queue,
        workDivWithBounds,
        withBoundsKernel,
        alpaka::getPtrNative(bufSrc),
        alpaka::getPtrNative(bufDst),
        problemSize);

    // Timed runs
    auto startWithBounds = std::chrono::high_resolution_clock::now();
    for(std::size_t i = 0; i < numRuns; ++i)
    {
        alpaka::exec<Acc>(
            queue,
            workDivWithBounds,
            withBoundsKernel,
            alpaka::getPtrNative(bufSrc),
            alpaka::getPtrNative(bufDst),
            problemSize);
    }
    alpaka::wait(queue);
    auto endWithBounds = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> durationWithBounds = (endWithBounds - startWithBounds) / numRuns;
    std::cout << "Avg. Duration (with bounds): " << durationWithBounds.count() << " ms" << std::endl;

    return EXIT_SUCCESS;
}

auto main() -> int
{
    std::cout << "Check enabled accelerator tags:" << std::endl;
    alpaka::printTagNames<alpaka::EnabledAccTags>();
    return alpaka::executeForEachAccTag([=](auto const& tag) { return example(tag); });
}
