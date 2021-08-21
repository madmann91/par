#ifndef PAR_OMP_EXECUTORS_H
#define PAR_OMP_EXECUTORS_H

#include "par/range.h"
#include "par/executor.h"

namespace par::omp {

struct StaticExecutor final : Executor {
    template <typename T, typename F>
    static void for_each_1d(const Range1<T>& range, const F& f) {
        #pragma omp parallel for
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            f(i);
    }

    template <typename T, typename F>
    static void for_each_2d(const Range2<T>& range, const F& f) {
        #pragma omp parallel for collapse(2)
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                f(i, j);
        }
    }

    template <typename T, typename F>
    static void for_each_3d(const Range3<T>& range, const F& f) {
        #pragma omp parallel for collapse(3)
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    f(i, j, k);
            }
        }
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_1d(const Range1<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for reduction(Red: res)
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            res.val = res.bin_op(res.val, un_op(i));
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_2d(const Range2<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(2) reduction(Red: res)
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                res.val = res.bin_op(res.val, un_op(i, j));
        }
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_3d(const Range3<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(3) reduction(Red: res)
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    res.val = res.bin_op(res.val, un_op(i, j, k));
            }
        }
        return res.val;
    }
};

struct DynamicExecutor final : Executor {
    template <typename T, typename F>
    static void for_each_1d(const Range1<T>& range, const F& f) {
        #pragma omp parallel for schedule(dynamic)
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            f(i);
    }

    template <typename T, typename F>
    static void for_each_2d(const Range2<T>& range, const F& f) {
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                f(i, j);
        }
    }

    template <typename T, typename F>
    static void for_each_3d(const Range3<T>& range, const F& f) {
        #pragma omp parallel for collapse(3) schedule(dynamic)
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    f(i, j, k);
            }
        }
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_1d(const Range1<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for reduction(Red: res) schedule(dynamic)
        for (auto i = range.begin[0]; i < range.end[0]; ++i)
            res.val = res.bin_op(res.val, un_op(i));
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_2d(const Range2<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(2) reduction(Red: res) schedule(dynamic)
        for (auto j = range.begin[1]; j < range.end[1]; ++j) {
            for (auto i = range.begin[0]; i < range.end[0]; ++i)
                res.val = res.bin_op(res.val, un_op(i, j));
        }
        return res.val;
    }

    template <typename T, typename U, typename UnOp, typename BinOp>
    static U transform_reduce_3d(const Range3<T>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) {
        struct Custom {
            U val;
            BinOp bin_op;

            Custom(U val, const BinOp& bin_op) : val(val), bin_op(bin_op) {}
            Custom(const Custom&) = default;
            void combine(const Custom& other) { val = bin_op(val, other.val); }
        };
        Custom res(init, bin_op);
        #pragma omp declare reduction(Red:Custom:omp_out.combine(omp_in)) initializer (omp_priv=omp_orig)
        #pragma omp parallel for collapse(3) reduction(Red: res) schedule(dynamic)
        for (auto k = range.begin[2]; k < range.end[2]; ++k) {
            for (auto j = range.begin[1]; j < range.end[1]; ++j) {
                for (auto i = range.begin[0]; i < range.end[0]; ++i)
                    res.val = res.bin_op(res.val, un_op(i, j, k));
            }
        }
        return res.val;
    }
};

} // namespace par::omp

#endif
