#ifndef SURCHEM_THREAD_SUM_H
#define SURCHEM_THREAD_SUM_H

#include <array>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace ModuleSurchem
{

// Per-thread partial results of a loop over the grid points [0, size), in
// thread order. Each thread runs accumulate(i, partial) over a fixed static
// share of the points in index order, starting from zeros. Unlike an OpenMP
// reduction clause, whose partial sums meet in run-dependent order, a given
// thread count therefore gives the same partials on every run, and one thread
// gives the serial loop. The sqrt-CG iteration count is sensitive to the last
// bit of its inner products, so the SCCS grid sums use this.
template <std::size_t Count, typename Accumulate>
std::vector<std::array<double, Count>> thread_partials(const std::size_t size,
                                                       const Accumulate& accumulate)
{
#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#else
    const int thread_count = 1;
#endif
    std::array<double, Count> zero;
    zero.fill(0.0);
    std::vector<std::array<double, Count>> partials(thread_count, zero);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_count)
#endif
    {
#ifdef _OPENMP
        const int thread = omp_get_thread_num();
#else
        const int thread = 0;
#endif
        std::array<double, Count> partial = zero;
#ifdef _OPENMP
#pragma omp for schedule(static, 1024)
#endif
        for (std::size_t i = 0; i < size; ++i)
        {
            accumulate(i, partial);
        }
        partials[thread] = partial;
    }
    return partials;
}

// Sums of the thread_partials, added in thread order.
template <std::size_t Count, typename Accumulate>
std::array<double, Count> thread_sums(const std::size_t size, const Accumulate& accumulate)
{
    const std::vector<std::array<double, Count>> partials
        = thread_partials<Count>(size, accumulate);
    std::array<double, Count> sums;
    sums.fill(0.0);
    for (std::size_t thread = 0; thread < partials.size(); ++thread)
    {
        for (std::size_t index = 0; index < Count; ++index)
        {
            sums[index] += partials[thread][index];
        }
    }
    return sums;
}

} // namespace ModuleSurchem

#endif
