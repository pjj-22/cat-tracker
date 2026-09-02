#include "cattrack/hungarian.h"

#include <algorithm>
#include <limits>

namespace cattrack {

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

// Hungarian algorithm with potentials, O(n^2 * m), for n rows <= m cols.
// `get(i, j)` yields the cost of row i (0-based, 0..n-1) against col j
// (0-based, 0..m-1). Returns col_for_row[i] = assigned column (0-based).
template <typename Get>
std::vector<int> solve_le(std::size_t n, std::size_t m, Get get) {
    // 1-indexed working arrays; column 0 is the sentinel.
    std::vector<double> u(n + 1, 0.0), v(m + 1, 0.0);
    std::vector<int> p(m + 1, 0), way(m + 1, 0);

    for (std::size_t i = 1; i <= n; ++i) {
        p[0] = static_cast<int>(i);
        std::size_t j0 = 0;
        std::vector<double> minv(m + 1, kInf);
        std::vector<char> used(m + 1, false);

        do {
            used[j0] = true;
            const int i0 = p[j0];
            double delta = kInf;
            std::size_t j1 = 0;

            for (std::size_t j = 1; j <= m; ++j) {
                if (used[j]) continue;
                const double cur = get(i0 - 1, j - 1) - u[i0] - v[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = static_cast<int>(j0);
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            for (std::size_t j = 0; j <= m; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        do {
            const int j1 = way[j0];
            p[j0] = p[j1];
            j0 = static_cast<std::size_t>(j1);
        } while (j0 != 0);
    }

    std::vector<int> col_for_row(n, -1);
    for (std::size_t j = 1; j <= m; ++j) {
        if (p[j] != 0) col_for_row[p[j] - 1] = static_cast<int>(j - 1);
    }
    return col_for_row;
}
}  // namespace

Assignment linear_sum_assignment(const std::vector<double>& cost,
                                 std::size_t rows, std::size_t cols) {
    Assignment out;
    if (rows == 0 || cols == 0) return out;

    if (rows <= cols) {
        auto col_for_row = solve_le(rows, cols, [&](std::size_t i, std::size_t j) {
            return cost[i * cols + j];
        });
        for (std::size_t i = 0; i < rows; ++i) {
            out.row_ind.push_back(static_cast<int>(i));
            out.col_ind.push_back(col_for_row[i]);
        }
    } else {
        // Transpose: solve with cols as the "rows" dimension, then flip back.
        auto row_for_col = solve_le(cols, rows, [&](std::size_t i, std::size_t j) {
            return cost[j * cols + i];
        });
        std::vector<std::pair<int, int>> pairs;
        for (std::size_t j = 0; j < cols; ++j) {
            pairs.emplace_back(row_for_col[j], static_cast<int>(j));
        }
        std::sort(pairs.begin(), pairs.end());
        for (auto& pr : pairs) {
            out.row_ind.push_back(pr.first);
            out.col_ind.push_back(pr.second);
        }
    }
    return out;
}

}  // namespace cattrack
