// Fixed-size dense linear algebra for the Kalman filter.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace cattrack {

template <std::size_t R, std::size_t C>
struct Mat {
    std::array<double, R * C> d{};

    double& operator()(std::size_t r, std::size_t c) { return d[r * C + c]; }
    double operator()(std::size_t r, std::size_t c) const { return d[r * C + c]; }

    static Mat<R, C> zeros() { return Mat<R, C>{}; }

    static Mat<R, C> identity() {
        Mat<R, C> m{};
        for (std::size_t i = 0; i < (R < C ? R : C); ++i) m(i, i) = 1.0;
        return m;
    }
};

template <std::size_t R, std::size_t K, std::size_t C>
Mat<R, C> operator*(const Mat<R, K>& a, const Mat<K, C>& b) {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t k = 0; k < K; ++k) {
            const double aik = a(i, k);
            if (aik == 0.0) continue;
            for (std::size_t j = 0; j < C; ++j) out(i, j) += aik * b(k, j);
        }
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator+(const Mat<R, C>& a, const Mat<R, C>& b) {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.d[i] = a.d[i] + b.d[i];
    return out;
}

template <std::size_t R, std::size_t C>
Mat<R, C> operator-(const Mat<R, C>& a, const Mat<R, C>& b) {
    Mat<R, C> out{};
    for (std::size_t i = 0; i < R * C; ++i) out.d[i] = a.d[i] - b.d[i];
    return out;
}

template <std::size_t R, std::size_t C>
Mat<C, R> transpose(const Mat<R, C>& a) {
    Mat<C, R> out{};
    for (std::size_t i = 0; i < R; ++i)
        for (std::size_t j = 0; j < C; ++j) out(j, i) = a(i, j);
    return out;
}

// Gauss-Jordan inverse with partial pivoting. N is small (4) so this is fine.
template <std::size_t N>
Mat<N, N> inverse(const Mat<N, N>& in) {
    Mat<N, N> a = in;
    Mat<N, N> inv = Mat<N, N>::identity();

    for (std::size_t col = 0; col < N; ++col) {
        std::size_t pivot = col;
        double best = std::fabs(a(col, col));
        for (std::size_t r = col + 1; r < N; ++r) {
            const double v = std::fabs(a(r, col));
            if (v > best) { best = v; pivot = r; }
        }
        if (best == 0.0) throw std::runtime_error("cattrack::inverse: singular matrix");

        if (pivot != col)
            for (std::size_t j = 0; j < N; ++j) {
                std::swap(a(col, j), a(pivot, j));
                std::swap(inv(col, j), inv(pivot, j));
            }

        const double diag = a(col, col);
        for (std::size_t j = 0; j < N; ++j) {
            a(col, j) /= diag;
            inv(col, j) /= diag;
        }

        for (std::size_t r = 0; r < N; ++r) {
            if (r == col) continue;
            const double factor = a(r, col);
            if (factor == 0.0) continue;
            for (std::size_t j = 0; j < N; ++j) {
                a(r, j) -= factor * a(col, j);
                inv(r, j) -= factor * inv(col, j);
            }
        }
    }
    return inv;
}

}  // namespace cattrack
