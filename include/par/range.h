#ifndef PAR_RANGE_H
#define PAR_RANGE_H

#include <type_traits>
#include <cstddef>
#include <array>

namespace par {

template <typename T, size_t N, std::enable_if_t<std::is_integral_v<T>, int> = 0>
struct Range {
    static_assert(N != 0);

    std::array<T, N> begin, end;

    template <size_t M>
    Range<T, N + M> product(const Range<T, N>& other) const {
        Range<T, N + M> res;
        std::copy_n(begin.begin(),  N, res.begin.begin());
        std::copy_n(end.begin(),    N, res.end.begin());
        std::copy_n(other.begin.begin(), M, res.begin.begin() + N);
        std::copy_n(other.end.begin(),   M, res.end.begin()   + N);
        return res;
    }

    Range<T, N - 1> remove_dim(size_t dim) const {
        Range<T, N - 1> res;
        std::copy_n(begin.begin(), dim, res.begin.begin());
        std::copy_n(begin.begin() + dim + 1, N - dim - 1, res.begin.begin() + dim);
        std::copy_n(end.begin(), dim, res.end.begin());
        std::copy_n(end.begin() + dim + 1, N - dim - 1, res.end.begin() + dim);
        return res;
    }
};

template <typename T> using Range1 = Range<T, 1>;
template <typename T> using Range2 = Range<T, 2>;
template <typename T> using Range3 = Range<T, 3>;

template <typename T>
inline Range1<T> range_1d(T begin, T end) {
    return Range1<T> { { begin }, { end } };
}

template <typename T>
inline Range2<T> range_2d(T begin_x, T end_x, T begin_y, T end_y) {
    return Range2<T> { { begin_x, begin_y }, { end_x, end_y } };
}

template <typename T>
inline Range3<T> range_3d(T begin_x, T end_x, T begin_y, T end_y, T begin_z, T end_z) {
    return Range2<T> { { begin_x, begin_y, begin_z }, { end_x, end_y, end_z } };
}

} // namespace par

#endif
