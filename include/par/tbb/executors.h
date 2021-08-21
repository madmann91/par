#ifndef PAR_TBB_EXECUTORS_H
#define PAR_TBB_EXECUTORS_H

#define TBB_SUPPRESS_DEPRECATED_MESSAGES 1
#include <tbb/tbb.h>

#include "par/range.h"
#include "par/executor.h"

namespace par::tbb {

struct Executor final : par::Executor {
    template <typename T, typename F>
    static void for_each_1d(const Range1<T>& range, const F& f) {
        ::tbb::parallel_for(
            ::tbb::blocked_range<T>(range.begin[0], range.end[0]),
            [&] (const ::tbb::blocked_range<T>& range) {
                for (auto i = range.begin(); i < range.end(); ++i)
                    f(i);
            });
    }

    template <typename T, typename F>
    static void for_each_2d(const Range2<T>& range, const F& f) {
        ::tbb::parallel_for(
            ::tbb::blocked_range2d<T>(
                range.begin[1], range.end[1],
                range.begin[0], range.end[0]),
            [&] (const ::tbb::blocked_range2d<T>& range) {
                for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                    for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                        f(i, j);
                }
            });
    }

    template <typename T, typename F>
    static void for_each_3d(const Range3<T>& range, const F& f) {
        ::tbb::parallel_for(
            ::tbb::blocked_range3d<T>(
                range.begin[2], range.end[2],
                range.begin[1], range.end[1],
                range.begin[0], range.end[0]),
            [&] (const ::tbb::blocked_range3d<T>& range) {
                for (auto k = range.pages().begin(); k < range.pages().end(); ++k) {
                    for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                        for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                            f(i, j, k);
                    }
                }
            });
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_1d(const Range1<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        return ::tbb::parallel_reduce(
            ::tbb::blocked_range<T>(range.begin[0], range.end[0]), init,
            [&] (const ::tbb::blocked_range<T>& range, U res) {
                for (auto i = range.begin(); i < range.end(); ++i)
                    res = bin_op(res, un_op(i));
                return res;
            }, bin_op);
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_2d(const Range2<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        return ::tbb::parallel_reduce(
            ::tbb::blocked_range2d<T>(
                range.begin[1], range.end[1],
                range.begin[0], range.end[0]),
            [&] (const ::tbb::blocked_range2d<T>& range, U res) {
                for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                    for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                        res = bin_op(res, un_op(i, j));
                }
                return res;
            }, bin_op);
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_3d(const Range3<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        return ::tbb::parallel_reduce(
            ::tbb::blocked_range3d<T>(
                range.begin[2], range.end[2],
                range.begin[1], range.end[1],
                range.begin[0], range.end[0]),
            [&] (const ::tbb::blocked_range3d<T>& range, U res) {
                for (auto k = range.pages().begin(); k < range.pages().end(); ++k) {
                    for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                        for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                            res = bin_op(res, un_op(i, j, k));
                    }
                }
                return res;
            }, bin_op);
    }
};

} // namespace par::tbb

#endif
