#ifndef PAR_FOR_EACH_H
#define PAR_FOR_EACH_H

#include "par/range.h"

namespace par {

/// Executes the given computation in parallel over an integral range.
template <typename Executor, typename T, size_t Dim, typename F>
void for_each(Executor& executor, const Range<T, Dim>& range, const F& f) {
    if constexpr (Dim == 1)
        executor.for_each_1d(range, f);
    else if constexpr (Dim == 2)
        executor.for_each_2d(range, f);
    else if constexpr (Dim == 3)
        executor.for_each_3d(range, f);
    else
        executor.for_each_nd(range, f);
}

} // namespace par

#endif
