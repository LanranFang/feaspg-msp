#include <algorithm>
#include <cstdint>
#include <limits>
#include <omp.h>

// Every stage objective is an integer divided by the same declared denominator.
// Python checks a worst-case bound excluding int64 overflow before calling here.
extern "C" void inner_step(
    int ns, int nq, int initial, const int64_t* offsets,
    const int64_t* next0, const int64_t* next1,
    const int64_t* f0, const int64_t* f1,
    const int64_t* wet, const int64_t* child,
    const int64_t* future, int64_t* output, int threads) {
    const int width = initial < 0 ? ns : 1;
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int q = 0; q < nq; ++q) {
        const int64_t* f = wet[q] ? f1 : f0;
        const int64_t* next = wet[q] ? next1 : next0;
        const int64_t* tail = future + child[q] * ns;
        for (int j = 0; j < width; ++j) {
            const int s = initial < 0 ? j : initial;
            int64_t best = std::numeric_limits<int64_t>::max();
            for (int64_t e = offsets[s]; e < offsets[s + 1]; ++e)
                best = std::min(best, f[e] + tail[next[e]]);
            output[q * width + j] = best;
        }
    }
}
