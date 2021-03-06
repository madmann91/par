#ifndef PAR_TRANSFORM_REDUCE_H
#define PAR_TRANSFORM_REDUCE_H

#include "par/range.h"

namespace par {

/// Executes the given computation in parallel over an integral range.
template <typename Executor, typename T, size_t Dim, typename U, typename BinOp, typename UnOp>
U transform_reduce(Executor& executor, const Range<T, Dim>& range, const U init, const BinOp& bin_op, const UnOp& un_op) {
    if constexpr (Dim == 1)
        return executor.transform_reduce_1d(range, init, bin_op, un_op);
    else if constexpr (Dim == 2)
        return executor.transform_reduce_2d(range, init, bin_op, un_op);
    else if constexpr (Dim == 3)
        return executor.transform_reduce_3d(range, init, bin_op, un_op);
    else
        return executor.transform_reduce_nd(range, init, bin_op, un_op);
}

} // namespace par

#endif
