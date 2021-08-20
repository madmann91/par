#ifndef PAR_EXECUTOR_H
#define PAR_EXECUTOR_H

#include <functional>

#include "par/range.h"
#include "par/for_each.h"
#include "par/transform_reduce.h"

namespace par {

struct Executor {
    template <typename T, size_t Dim, typename F>
    void for_each_nd(const range<T, Dim>& range, const F& f) const {
        for (auto i = range.begin; i < range.end; ++i)
            for_each(*this, static_cast<range<T, Dim - 1>>(range), std::bind_front(f, i));
    }

    template <typename T, size_t Dim, typename U, typename UnOp, typename BinOp>
    void transform_reduce_nd(const range<T, Dim>& range, const U& init, const BinOp& bin_op, const UnOp& un_op) const {
        auto res = init;
        for (auto i = range.begin; i < range.end; ++i)
            res = transform_reduce(*this, static_cast<range<T, Dim - 1>>(range), res, bin_op, std::bind_front(f, i));
        return res;
    }
};

} // namespace par

#endif
