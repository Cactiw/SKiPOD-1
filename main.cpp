#include <iostream>
#include <vector>

const double EPS = 1E-9;


int rank(std::vector<std::vector<int>> &a) {
    if (a.empty()) {
        return 0;
    }
    int n = a.size(), m = a[0].size();

    int rank = std::max(n, m);
    std::vector<bool> line_used(n);
    for (int i = 0; i < m; ++i) {
        int j;
        for (j = 0; j < n; ++j)
            if (!line_used[j] && abs(a[j][i]) > EPS)
                break;
        if (j == n)
            --rank;
        else {
            line_used[j] = true;
            for (int p = i + 1; p < m; ++p)
                a[j][p] /= a[j][i];
            for (int k = 0; k < n; ++k)
                if (k != j && abs(a[k][i]) > EPS)
                    for (int p = i + 1; p < m; ++p)
                        a[k][p] -= a[j][p] * a[k][i];
        }
    }
    return rank;
}


int main() {
    std::vector<std::vector<int>> a;
    a = {{1, 2, 3, 4, 5,}, {2, 3, 4, 5, 1}, {1, 2, 3, 4, 5,}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
    std::cout << rank(a);

    return 0;
}
