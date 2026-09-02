// Rectangular linear sum assignment (minimum cost matching).
// Drop-in replacement for scipy.optimize.linear_sum_assignment used by
// MultiTracker._match(). Returns a matching of size min(rows, cols), with
// row indices ascending (matching scipy's output ordering).
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace cattrack {

struct Assignment {
    std::vector<int> row_ind;
    std::vector<int> col_ind;
};

// cost is row-major, shape (rows x cols). Minimizes the total assigned cost.
Assignment linear_sum_assignment(const std::vector<double>& cost,
                                 std::size_t rows, std::size_t cols);

}  // namespace cattrack
