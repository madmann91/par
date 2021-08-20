#ifndef PAR_TBB_EXECUTORS_H
#define PAR_TBB_EXECUTORS_H

#include "par/range.h"
#include "par/executor.h"

namespace par::tbb {

struct Executor final : par::Executor {
    template <typename T, typename F>
    static void for_each_1d(const range1d<T>& range, const F& f) {
        tbb::parallel_for(
            tbb::blocked_range<T>(range.begin, range.end),
            [&] (const range& range) {
                for (auto i = range.begin(); i < range.end(); ++i)
                    f(i);
            });
    }

    template <typename T, typename F>
    static void for_each_2d(const range2d<T>& range, const F& f) {
        tbb::parallel_for(
            tbb::blocked_range2d<T>(
                range.range2d<T>::begin, range.range2d<T>::end,
                range.range1d<T>::begin, range.range1d<T>::end),
            [&] (const range& range) {
                for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                    for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                        f(i, j);
                }
            });
    }

    template <typename T, typename F>
    static void for_each_3d(const range3d<T>& range, const F& f) {
        tbb::parallel_for(
            tbb::blocked_range3d<T>(
                range.range3d<T>::begin, range.range3d<T>::end,
                range.range2d<T>::begin, range.range2d<T>::end,
                range.range1d<T>::begin, range.range1d<T>::end),
            [&] (const range& range) {
                for (auto k = range.pages().begin(); k < range.pages().end(); ++k) {
                    for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                        for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                            f(i, j, k);
                    }
                }
            });
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_1d(const range1d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        tbb::parallel_reduce(
            tbb::blocked_range<T>(range.begin, range.end), init,
            [&] (const range& range, U res) {
                for (auto i = range.begin(); i < range.end(); ++i)
                    res = bin_op(res, un_op(i));
                return res;
            }, bin_op);
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_2d(const range2d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        tbb::parallel_reduce(
            tbb::blocked_range2d<T>(
                range.range2d<T>::begin, range.range2d<T>::end,
                range.range1d<T>::begin, range.range1d<T>::end),
            [&] (const range& range, U res) {
                for (auto j = range.rows().begin(); j < range.rows().end(); ++j) {
                    for (auto i = range.cols().begin(); i < range.cols().end(); ++i)
                        res = bin_op(res, un_op(i, j, k));
                }
                return res;
            }, bin_op);
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_3d(const range3d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        tbb::parallel_reduce(
            tbb::blocked_range3d<T>(
                range.range3d<T>::begin, range.range3d<T>::end,
                range.range2d<T>::begin, range.range2d<T>::end,
                range.range1d<T>::begin, range.range1d<T>::end),
            [&] (const range& range, U res) {
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
