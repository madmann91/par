#ifndef PAR_SEQUENTIAL_EXECUTOR_H
#define PAR_SEQUENTIAL_EXECUTOR_H

#include "par/executor.h"

namespace par {

struct SequentialExecutor final : Executor {
    template <typename T, typename F>
    static void for_each_1d(const Range1<T>& range, const F& f) {
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            f(i);
    }

    template <typename T, typename F>
    static void for_each_2d(const Range2<T>& range, const F& f) {
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                f(i, j);
        }
    }

    template <typename T, typename F>
    static void for_each_3d(const Range3<T>& range, const F& f) {
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    f(i, j, k);
            }
        }
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_1d(const Range1<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        auto res = init;
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            res = bin_op(res, un_op(i));
        return res;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_2d(const Range2<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        auto res = init;
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                res = bin_op(res, un_op(i, j));
        }
        return res;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_3d(const Range3<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        auto res = init;
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    res = bin_op(res, un_op(i, j, k));
            }
        }
        return res;
    }
};

} // namespace par

#endif
