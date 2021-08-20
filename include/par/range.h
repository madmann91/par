#ifndef PAR_RANGE_H
#define PAR_RANGE_H

#include <type_traits>
#include <cstddef>

namespace par {

template <typename T, size_t Dim, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
struct range : range<T, Dim - 1> {
    T begin, end;
};

template <typename T>
struct range<T, 0> {};

template <typename T> using range1d = range<T, 1>;
template <typename T> using range2d = range<T, 2>;
template <typename T> using range3d = range<T, 3>;

} // namespace par

#endif
