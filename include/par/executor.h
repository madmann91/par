#ifndef PAR_EXECUTOR_H
#define PAR_EXECUTOR_H

#include "par/range.h"
#include "par/for_each.h"
#include "par/transform_reduce.h"

namespace par {

struct Executor {
protected:
    template <size_t N, typename T, typename F>
    void for_each_nd(const Range<T, N>& range, const F& f) const {
        auto sub_range = range.remove_dim(N - 1);
        for (auto i = range.begins[N - 1]; i < range.ends[N - 1]; ++i)
            for_each(*this, [&] (auto&&... args) { f(std::forward<decltype(args)>(args)..., i); });
    }

    template <size_t N, typename T, typename U, typename UnOp, typename BinOp>
    void transform_reduce_nd(const Range<T, N>& range, U res, const BinOp& bin_op, const UnOp& un_op) const {
        auto sub_range = range.remove_dim(N - 1);
        for (auto i = range.begins[N - 1]; i < range.ends[N - 1]; ++i)
            res = transform_reduce(*this, sub_range, res, bin_op, [&] (auto&&... args) { un_op(std::forward<decltype(args)>(args)..., i); });
        return res;
    }
};

} // namespace par

#endif
