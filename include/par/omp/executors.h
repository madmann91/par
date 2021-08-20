#ifndef PAR_OMP_EXECUTORS_H
#define PAR_OMP_EXECUTORS_H

#include "par/range.h"
#include "par/executor.h"

namespace par::omp {

struct StaticExecutor final : Executor {
    template <typename T, typename F>
    static void for_each_1d(const range1d<T>& range, const F& f) {
        #pragma omp parallel for
        for (auto i = range.begin; i < range.end; ++i)
            f(i);
    }

    template <typename T, typename F>
    static void for_each_2d(const range2d<T>& range, const F& f) {
        #pragma omp parallel for collapse(2)
        for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
            for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                f(i, j);
        }
    }

    template <typename T, typename F>
    static void for_each_3d(const range3d<T>& range, const F& f) {
        #pragma omp parallel for collapse(3)
        for (auto k = range.range3d<T>::begin; k < range.range3d<T>::end; ++k) {
            for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
                for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                    f(i, j, k);
            }
        }
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_1d(const range1d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for reduction(ReduceOp: res)
        for (auto i = range.begin; i < range.end; ++i)
            res.val = res.bin_op(res.val, un_op(i));
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_2d(const range2d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(2) reduction(ReduceOp: res)
        for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
            for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                res.val = res.bin_op(res.val, un_op(i, j));
        }
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_3d(const range3d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(3) reduction(ReduceOp: res)
        for (auto k = range.range3d<T>::begin; k < range.range3d<T>::end; ++k) {
            for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
                for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                    res.val = res.bin_op(res.val, un_op(i, j, k));
            }
        }
        return res.val;
    }
};

struct DynamicExecutor final : Executor {
    template <typename T, typename F>
    static void for_each_1d(const range1d<T>& range, const F& f) {
        #pragma omp parallel for schedule(dynamic)
        for (auto i = range.begin; i < range.end; ++i)
            f(i);
    }

    template <typename T, typename F>
    static void for_each_2d(const range2d<T>& range, const F& f) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
            for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                f(i, j);
        }
    }

    template <typename T, typename F>
    static void for_each_3d(const range3d<T>& range, const F& f) {
        #pragma omp parallel for collapse(3) schedule(dynamic)
        for (auto k = range.range3d<T>::begin; k < range.range3d<T>::end; ++k) {
            for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
                for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                    f(i, j, k);
            }
        }
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_1d(const range1d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for reduction(ReduceOp: res) schedule(dynamic)
        for (auto i = range.begin; i < range.end; ++i)
            res.val = res.bin_op(res.val, un_op(i));
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_2d(const range2d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(2) reduction(ReduceOp: res) schedule(dynamic)
        for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
            for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                res.val = res.bin_op(res.val, un_op(i, j));
        }
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static void transform_reduce_3d(const range3d<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            T val;
            BinOp bin_op;

            Custom(T val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in, omp_out)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(3) reduction(ReduceOp: res) schedule(dynamic)
        for (auto k = range.range3d<T>::begin; k < range.range3d<T>::end; ++k) {
            for (auto j = range.range2d<T>::begin; j < range.range2d<T>::end; ++j) {
                for (auto i = range.range1d<T>::begin; i < range.range1d<T>::end; ++i)
                    res.val = res.bin_op(res.val, un_op(i, j, k));
            }
        }
        return res.val;
    }
};

} // namespace par::omp

#endif
