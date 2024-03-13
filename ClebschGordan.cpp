#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <vector>

// Declaration of LAPACK subroutines
// Make sure the data types match your version of LAPACK

extern "C" void dgesvd_(char const* JOBU,
                        char const* JOBVT,
                        int const* M,
                        int const* N,
                        double* A,
                        int const* LDA,
                        double* S,
                        double* U,
                        int const* LDU,
                        double* VT,
                        int const* LDVT,
                        double* WORK,
                        int const* LWORK,
                        int *INFO);

extern "C" void dgels_(char const* TRANS,
                       int const* M,
                       int const* N,
                       int const* NRHS,
                       double* A,
                       int const* LDA,
                       double* B,
                       int const* LDB,
                       double* WORK,
                       int const* LWORK,
                       int *INFO);

namespace clebsch {
    const double EPS = 1e-12;

    // binomial coefficients
    class binomial_t {
        std::vector<int> cache;
        int N;

    public:
        int operator()(int n, int k);
    } binomial;

    // Eq. (19) and (25)
    class weight {
        std::vector<int> elem;

    public:
        // the N in "SU(N)"
        const int N;

        // create a non-initialized weight
        weight(int N);

        // create irrep weight of given index
        // Eq. (C2)
        weight(int N, int index);

        // assign from another instance
        clebsch::weight &operator=(const clebsch::weight &w);

        // access elements of this weight (k = 1, ..., N)
        int &operator()(int k);
        const int &operator()(int k) const;

        // compare weights
        // Eq. (C1)
        bool operator<(const weight &w) const;
        bool operator==(const weight &w) const;

        // element-wise sum of weights
        clebsch::weight operator+(const weight &w) const;

        // returns the index of this irrep weight (index = 0, 1, ...)
        // Eq. (C2)
        int index() const;

        // returns the dimension of this irrep weight
        // Eq. (22)
        long long dimension() const;
    };

    // Eq. (20)
    class pattern {
        std::vector<int> elem;

    public:
        // the N in "SU(N)"
        const int N;

        // copy constructor
        pattern(const pattern &pat);

        // create pattern of given index from irrep weight
        // Eq. (C7)
        pattern(const weight &irrep, int index = 0);

        // access elements of this pattern (l = 1, ..., N; k = 1, ..., l)
        int &operator()(int k, int l);
        const int &operator()(int k, int l) const;

        // find succeeding/preceding pattern, return false if not possible
        // Eq. (C9)
        bool operator++();
        bool operator--();

        // returns the pattern index (index = 0, ..., dimension - 1)
        // Eq. (C7)
        int index() const;

        // returns the pattern weight
        // Eq. (25)
        clebsch::weight get_weight() const;

        // returns matrix element of lowering operator J^(l)_-
        // between this pattern minus M^(k,l) and this pattern
        // (l = 1, ..., N; k = 1, ..., l)
        // Eq. (28)
        double lowering_coeff(int k, int l) const;

        // returns matrix element of raising operator J^(l)_+
        // between this pattern plus M^(k,l) and this pattern
        // (l = 1, ..., N; k = 1, ..., l)
        // Eq. (29)
        double raising_coeff(int k, int l) const;
    };

    class decomposition {
        std::vector<clebsch::weight> weights;
        std::vector<int> multiplicities;

    public:
        // the N in "SU(N)"
        const int N;

        // save given irreps for later use
        const weight factor1, factor2;

        // construct the decomposition of factor1 times factor2 into irreps
        // Eq. (31)
        decomposition(const weight &factor1, const weight &factor2);

        // return the number of occurring irreps
        int size() const;

        // access the occurring irreps
        // j = 0, ..., size() - 1
        const clebsch::weight &operator()(int j) const;

        // return the outer multiplicity of irrep in this decomposition
        int multiplicity(const weight &irrep) const;
    };

    class index_adapter {
        std::vector<int> indices;
        std::vector<int> multiplicities;

    public:
        // the N in "SU(N)"
        const int N;

        // save given irreps for later use
        const int factor1, factor2;

        // construct this index_adapter from a given decomposition
        index_adapter(const clebsch::decomposition &decomp);

        // return the number of occurring irreps
        int size() const;

        // access the occurring irreps
        int operator()(int j) const;

        // return the outer multiplicity of irrep in this decomposition
        int multiplicity(int irrep) const;
    };

    class coefficients {
        std::map<std::vector<int>, double> clzx;

        // access Clebsch-Gordan coefficients in convenient manner
        void set(int factor1_state,
                 int factor2_state,
                 int multiplicity_index,
                 int irrep_state,
                 double value);

        // internal functions, doing most of the work
        void highest_weight_normal_form(); // Eq. (37)
        void compute_highest_weight_coeffs(); // Eq. (36)
        void compute_lower_weight_coeffs(int multip_index, int state, std::vector<char> &done); // Eq. (40)

    public:
        // the N in "SU(N)"
        const int N;

        // save irreps and their dimensions for later use
        const weight factor1, factor2, irrep;
        const int factor1_dimension, factor2_dimension, irrep_dimension;

        // outer multiplicity of irrep in this decomposition
        const int multiplicity;

        // construct all Clebsch-Gordan coefficients of this decomposition
        coefficients(const weight &irrep, const weight &factor1, const weight &factor2);

        // access Clebsch-Gordan coefficients (read-only)
        // multiplicity_index = 0, ..., multiplicity - 1
        // factor1_state = 0, ..., factor1_dimension - 1
        // factor2_state = 0, ..., factor2_dimension - 1
        // irrep_state = 0, ..., irrep_dimension
        double operator()(int factor1_state,
                          int factor2_state,
                          int multiplicity_index,
                          int irrep_state) const;
    };
};

// implementation of "binomial_t" starts here

int clebsch::binomial_t::operator()(int n, int k) {
    if (N <= n) {
        for (cache.resize((n + 1) * (n + 2) / 2); N <= n; ++N) {
            cache[N * (N + 1) / 2] = cache[N * (N + 1) / 2 + N] = 1;
            for (int k = 1; k < N; ++k) {
                cache[N * (N + 1) / 2 + k] = cache[(N - 1) * N / 2 + k]
                                           + cache[(N - 1) * N / 2 + k - 1];
            }
        }
    }

    return cache[n * (n + 1) / 2 + k];
}

// implementation of "weight" starts here

clebsch::weight::weight(int N) : elem(N), N(N) {}

clebsch::weight::weight(int N, int index) : elem(N, 0), N(N) {
    for (int i = 0; index > 0 && i < N; ++i) {
        for (int j = 1; binomial(N - i - 1 + j, N - i - 1) <= index; j <<= 1) {
            elem[i] = j;
        }

        for (int j = elem[i] >> 1; j > 0; j >>= 1) {
            if (binomial(N - i - 1 + (elem[i] | j), N - i - 1) <= index) {
                elem[i] |= j;
            }
        }

        index -= binomial(N - i - 1 + elem[i]++, N - i - 1);
    }
}

clebsch::weight &clebsch::weight::operator=(const clebsch::weight &w) {
    int &n = const_cast<int &>(N);
    elem = w.elem;
    n = w.N;
    return *this;
}

int &clebsch::weight::operator()(int k) {
    assert(1 <= k && k <= N);
    return elem[k - 1];
}

const int &clebsch::weight::operator()(int k) const {
    assert(1 <= k && k <= N);
    return elem[k - 1];
}

bool clebsch::weight::operator<(const weight &w) const {
    assert(w.N == N);
    for (int i = 0; i < N; ++i) {
        if (elem[i] - elem[N - 1] != w.elem[i] - w.elem[N - 1]) {
            return elem[i] - elem[N - 1] < w.elem[i] - w.elem[N - 1];
        }
    }
    return false;
}

bool clebsch::weight::operator==(const weight &w) const {
    assert(w.N == N);

    for (int i = 1; i < N; ++i) {
        if (w.elem[i] - w.elem[i - 1] != elem[i] - elem[i - 1]) {
            return false;
        }
    }

    return true;
}

clebsch::weight clebsch::weight::operator+(const weight &w) const {
    weight result(N);

    transform(elem.begin(), elem.end(), w.elem.begin(), result.elem.begin(), std::plus<int>());

    return result;
}

int clebsch::weight::index() const {
    int result = 0;

    for (int i = 0; elem[i] > elem[N - 1]; ++i) {
        result += binomial(N - i - 1 + elem[i] - elem[N - 1] - 1, N - i - 1);
    }

    return result;
}

long long clebsch::weight::dimension() const {
    long long numerator = 1, denominator = 1;

    for (int i = 1; i < N; ++i) {
        for (int j = 0; i + j < N; ++j) {
            numerator *= elem[j] - elem[i + j] + i;
            denominator *= i;
        }
    }

    return numerator / denominator;
}

// implementation of "pattern" starts here

clebsch::pattern::pattern(const pattern &p) : elem(p.elem), N(p.N) {}

clebsch::pattern::pattern(const weight &irrep, int index) :
        elem((irrep.N * (irrep.N + 1)) / 2), N(irrep.N) {
    for (int i = 1; i <= N; ++i) {
        (*this)(i, N) = irrep(i);
    }

    for (int l = N - 1; l >= 1; --l) {
        for (int k = 1; k <= l; ++k) {
            (*this)(k, l) = (*this)(k + 1, l + 1);
        }
    }

    while (index-- > 0) {
        bool b = ++(*this);

        assert(b);
    }
}

int &clebsch::pattern::operator()(int k, int l) {
    return elem[(N * (N + 1) - l * (l + 1)) / 2 + k - 1];
}

const int &clebsch::pattern::operator()(int k, int l) const {
    return elem[(N * (N + 1) - l * (l + 1)) / 2 + k - 1];
}

bool clebsch::pattern::operator++() {
    int k = 1, l = 1;

    while (l < N && (*this)(k, l) == (*this)(k, l + 1)) {
        if (--k == 0) {
            k = ++l;
        }
    }

    if (l == N) {
        return false;
    }

    ++(*this)(k, l);

    while (k != 1 || l != 1) {
        if (++k > l) {
            k = 1;
            --l;
        }

        (*this)(k, l) = (*this)(k + 1, l + 1);
    }

    return true;
}

bool clebsch::pattern::operator--() {
    int k = 1, l = 1;

    while (l < N && (*this)(k, l) == (*this)(k + 1, l + 1)) {
        if (--k == 0) {
            k = ++l;
        }
    }

    if (l == N) {
        return false;
    }

    --(*this)(k, l);

    while (k != 1 || l != 1) {
        if (++k > l) {
            k = 1;
            --l;
        }

        (*this)(k, l) = (*this)(k, l + 1);
    }

    return true;
}

int clebsch::pattern::index() const {
    int result = 0;

    for (pattern p(*this); --p; ++result) {}

    return result;
}

clebsch::weight clebsch::pattern::get_weight() const {
    clebsch::weight result(N);

    for (int prev = 0, l = 1; l <= N; ++l) {
        int now = 0;

        for (int k = 1; k <= l; ++k) {
            now += (*this)(k, l);
        }

        result(l) = now - prev;
        prev = now;
    }

    return result;
}

double clebsch::pattern::lowering_coeff(int k, int l) const {
    double result = 1.0;

    for (int i = 1; i <= l + 1; ++i) {
        result *= (*this)(i, l + 1) - (*this)(k, l) + k - i + 1;
    }
    
    for (int i = 1; i <= l - 1; ++i) {
        result *= (*this)(i, l - 1) - (*this)(k, l) + k - i;
    }

    for (int i = 1; i <= l; ++i) {
        if (i == k) continue;
        result /= (*this)(i, l) - (*this)(k, l) + k - i + 1;
        result /= (*this)(i, l) - (*this)(k, l) + k - i;
    }

    return std::sqrt(-result);
}

double clebsch::pattern::raising_coeff(int k, int l) const {
    double result = 1.0;

    for (int i = 1; i <= l + 1; ++i) {
        result *= (*this)(i, l + 1) - (*this)(k, l) + k - i;
    }

    for (int i = 1; i <= l - 1; ++i) {
        result *= (*this)(i, l - 1) - (*this)(k, l) + k - i - 1;
    }

    for (int i = 1; i <= l; ++i) {
        if (i == k) continue;
        result /= (*this)(i, l) - (*this)(k, l) + k - i;
        result /= (*this)(i, l) - (*this)(k, l) + k - i  - 1;
    }

    return std::sqrt(-result);
}

// implementation of "decomposition" starts here

clebsch::decomposition::decomposition(const weight &factor1, const weight &factor2) :
        N(factor1.N), factor1(factor1), factor2(factor2) {
    assert(factor1.N == factor2.N);
    std::vector<clebsch::weight> result;
    pattern low(factor1), high(factor1);
    weight trial(factor2);
    int k = 1, l = N;

    do {
        while (k <= N) {
            --l;
            if (k <= l) {
                low(k, l) = std::max(high(k + N - l, N), high(k, l + 1) + trial(l + 1) - trial(l));
                high(k, l) = high(k, l + 1);
                if (k > 1 && high(k, l) > high(k - 1, l - 1)) {
                    high(k, l) = high(k - 1, l - 1);
                }
                if (l > 1 && k == l && high(k, l) > trial(l - 1) - trial(l)) {
                    high(k, l) = trial(l - 1) - trial(l);
                }
                if (low(k, l) > high(k, l)) {
                    break;
                }
                trial(l + 1) += high(k, l + 1) - high(k, l);
            } else {
                trial(l + 1) += high(k, l + 1);
                ++k;
                l = N;
            }
        }

        if (k > N) {
            result.push_back(trial);
            for (int i = 1; i <= N; ++i) {
                result.back()(i) -= result.back()(N);
            }
        } else {
            ++l;
        }

        while (k != 1 || l != N) {
            if (l == N) {
                l = --k - 1;
                trial(l + 1) -= high(k, l + 1);
            } else if (low(k, l) < high(k, l)) {
                --high(k, l);
                ++trial(l + 1);
                break;
            } else {
                trial(l + 1) -= high(k, l + 1) - high(k, l);
            }
            ++l;
        }
    } while (k != 1 || l != N);

    sort(result.begin(), result.end());
    for (std::vector<clebsch::weight>::iterator it = result.begin(); it != result.end(); ++it) {
        if (it != result.begin() && *it == weights.back()) {
            ++multiplicities.back();
        } else {
            weights.push_back(*it);
            multiplicities.push_back(1);
        }
    }
}

int clebsch::decomposition::size() const {
    return weights.size();
}

const clebsch::weight &clebsch::decomposition::operator()(int j) const {
    return weights[j];
}

int clebsch::decomposition::multiplicity(const weight &irrep) const {
    assert(irrep.N == N);
    std::vector<clebsch::weight>::const_iterator it
        = std::lower_bound(weights.begin(), weights.end(), irrep);

    return it != weights.end() && *it == irrep ? multiplicities[it - weights.begin()] : 0;
}

// implementation of "index_adapter" starts here

clebsch::index_adapter::index_adapter(const clebsch::decomposition &decomp) :
        N(decomp.N),
        factor1(decomp.factor1.index()),
        factor2(decomp.factor2.index()) {
    for (int i = 0, s = decomp.size(); i < s; ++i) {
        indices.push_back(decomp(i).index());
        multiplicities.push_back(decomp.multiplicity(decomp(i)));
    }
}

int clebsch::index_adapter::size() const {
    return indices.size();
}

int clebsch::index_adapter::operator()(int j) const {
    return indices[j];
}

int clebsch::index_adapter::multiplicity(int irrep) const {
    std::vector<int>::const_iterator it = std::lower_bound(indices.begin(), indices.end(), irrep);

    return it != indices.end() && *it == irrep ? multiplicities[it - indices.begin()] : 0;
}

// implementation of "clebsch" starts here

void clebsch::coefficients::set(int factor1_state,
                                int factor2_state,
                                int multiplicity_index,
                                int irrep_state,
                                double value) {
    assert(0 <= factor1_state && factor1_state < factor1_dimension);
    assert(0 <= factor2_state && factor2_state < factor2_dimension);
    assert(0 <= multiplicity_index && multiplicity_index < multiplicity);
    assert(0 <= irrep_state && irrep_state < irrep_dimension);

    int coefficient_label[] = { factor1_state,
                                factor2_state,
                                multiplicity_index,
                                irrep_state };
    clzx[std::vector<int>(coefficient_label, coefficient_label
            + sizeof coefficient_label / sizeof coefficient_label[0])] = value;
}

void clebsch::coefficients::highest_weight_normal_form() {
    int hws = irrep_dimension - 1;

    // bring CGCs into reduced row echelon form
    for (int h = 0, i = 0; h < multiplicity - 1 && i < factor1_dimension; ++i) {
        for (int j = 0; h < multiplicity - 1 && j < factor2_dimension; ++j) {
            int k0 = h;

            for (int k = h + 1; k < multiplicity; ++k) {
                if (fabs((*this)(i, j, k, hws)) > fabs((*this)(i, j, k0, hws))) {
                    k0 = k;
                }
            }

            if ((*this)(i, j, k0, hws) < -EPS) {
                for (int i2 = i; i2 < factor1_dimension; ++i2) {
                    for (int j2 = i2 == i ? j : 0; j2 < factor2_dimension; ++j2) {
                        set(i2, j2, k0, hws, -(*this)(i2, j2, k0, hws));
                    }
                }
            } else if ((*this)(i, j, k0, hws) < EPS) {
                continue;
            }

            if (k0 != h) {
                for (int i2 = i; i2 < factor1_dimension; ++i2) {
                    for (int j2 = i2 == i ? j : 0; j2 < factor2_dimension; ++j2) {
                        double x = (*this)(i2, j2, k0, hws);
                        set(i2, j2, k0, hws, (*this)(i2, j2, h, hws));
                        set(i2, j2, h, hws, x);
                    }
                }
            }

            for (int k = h + 1; k < multiplicity; ++k) {
                for (int i2 = i; i2 < factor1_dimension; ++i2) {
                    for (int j2 = i2 == i ? j : 0; j2 < factor2_dimension; ++j2) {
                        set(i2, j2, k, hws, (*this)(i2, j2, k, hws) - (*this)(i2, j2, h, hws)
                                * (*this)(i, j, k, hws) / (*this)(i, j, h, hws));
                    }
                }
            }

            // next 3 lines not strictly necessary, might improve numerical stability
            for (int k = h + 1; k < multiplicity; ++k) {
                set(i, j, k, hws, 0.0);
            }

            ++h;
        }
    }

    // Gram-Schmidt orthonormalization
    for (int h = 0; h < multiplicity; ++h) {
        for (int k = 0; k < h; ++k) {
            double overlap = 0.0;
            for (int i = 0; i < factor1_dimension; ++i) {
                for (int j = 0; j < factor2_dimension; ++j) {
                    overlap += (*this)(i, j, h, hws) * (*this)(i, j, k, hws);
                }
            }

            for (int i = 0; i < factor1_dimension; ++i) {
                for (int j = 0; j < factor2_dimension; ++j) {
                    set(i, j, h, hws, (*this)(i, j, h, hws) - overlap * (*this)(i, j, k, hws));
                }
            }
        }

        double norm = 0.0;
        for (int i = 0; i < factor1_dimension; ++i) {
            for (int j = 0; j < factor2_dimension; ++j) {
                norm += (*this)(i, j, h, hws) * (*this)(i, j, h, hws);
            }
        }
        norm = std::sqrt(norm);

        for (int i = 0; i < factor1_dimension; ++i) {
            for (int j = 0; j < factor2_dimension; ++j) {
                set(i, j, h, hws, (*this)(i, j, h, hws) / norm);
            }
        }
    }
}

void clebsch::coefficients::compute_highest_weight_coeffs() {
    if (multiplicity == 0) {
        return;
    }

    std::vector<std::vector<int> > map_coeff(factor1_dimension,
                                             std::vector<int>(factor2_dimension, -1));
    std::vector<std::vector<int> > map_states(factor1_dimension,
                                              std::vector<int>(factor2_dimension, -1));
    int n_coeff = 0, n_states = 0;
    pattern p(factor1, 0);

    for (int i = 0; i < factor1_dimension; ++i, ++p) {
        weight pw(p.get_weight());
        pattern q(factor2, 0);
        for (int j = 0; j < factor2_dimension; ++j, ++q) {
            if (pw + q.get_weight() == irrep) {
                map_coeff[i][j] = n_coeff++;
            }
        }
    }

    if (n_coeff == 1) {
        for (int i = 0; i < factor1_dimension; ++i) {
            for (int j = 0; j < factor2_dimension; ++j) {
                if (map_coeff[i][j] >= 0) {
                    set(i, j, 0, irrep_dimension - 1, 1.0);
                    return;
                }
            }
        }
    }

    double *hw_system = new double[n_coeff * (factor1_dimension * factor2_dimension)];
    assert(hw_system != NULL);
    memset(hw_system, 0, n_coeff * (factor1_dimension * factor2_dimension) * sizeof (double));

    pattern r(factor1, 0);
    for (int i = 0; i < factor1_dimension; ++i, ++r) {
        pattern q(factor2, 0);

        for (int j = 0; j < factor2_dimension; ++j, ++q) {
            if (map_coeff[i][j] >= 0) {
                for (int l = 1; l <= N - 1; ++l) {
                    for (int k = 1; k <= l; ++k) {
                        if ((k == 1 || r(k, l) + 1 <= r(k - 1, l - 1)) && r(k, l) + 1 <= r(k, l + 1)) {
                            ++r(k, l);
                            int h = r.index();
                            --r(k, l);

                            if (map_states[h][j] < 0) {
                                map_states[h][j] = n_states++;
                            }

                            hw_system[n_coeff * map_states[h][j] + map_coeff[i][j]]
                                += r.raising_coeff(k, l);
                        }

                        if ((k == 1 || q(k, l) + 1 <= q(k - 1, l - 1)) && q(k, l) + 1 <= q(k, l + 1)) {
                            ++q(k, l);
                            int h = q.index();
                            --q(k, l);

                            if (map_states[i][h] < 0) {
                                map_states[i][h] = n_states++;
                            }


                            hw_system[n_coeff * map_states[i][h] + map_coeff[i][j]]
                                += q.raising_coeff(k, l);
                        }
                    }
                }
            }
        }
    }

    int lwork = -1, info;
    double worksize;

    double *singval = new double[std::min(n_coeff, n_states)];
    assert(singval != NULL);
    double *singvec = new double[n_coeff * n_coeff];
    assert(singvec != NULL);

    dgesvd_("A",
            "N",
            &n_coeff,
            &n_states,
            hw_system,
            &n_coeff,
            singval,
            singvec,
            &n_coeff,
            NULL,
            &n_states,
            &worksize,
            &lwork,
            &info);
    assert(info == 0);

    lwork = worksize;
    double *work = new double[lwork];
    assert(work != NULL);

    dgesvd_("A",
            "N",
            &n_coeff,
            &n_states,
            hw_system,
            &n_coeff,
            singval,
            singvec,
            &n_coeff,
            NULL,
            &n_states,
            work,
            &lwork,
            &info);
    assert(info == 0);

    for (int i = 0; i < multiplicity; ++i) {
        for (int j = 0; j < factor1_dimension; ++j) {
            for (int k = 0; k < factor2_dimension; ++k) {
                if (map_coeff[j][k] >= 0) {
                    double x = singvec[n_coeff * (n_coeff - 1 - i) + map_coeff[j][k]];

                    if (fabs(x) > EPS) {
                        set(j, k, i, irrep_dimension - 1, x);
                    }
                }
            }
        }
    }

    // uncomment next line to bring highest-weight coefficients into "normal form"
    // highest_weight_normal_form();

    delete[] work;
    delete[] singvec;
    delete[] singval;
    delete[] hw_system;
}

void clebsch::coefficients::compute_lower_weight_coeffs(int multip_index,
                                                        int state,
                                                        std::vector<char> &done) {
    weight statew(pattern(irrep, state).get_weight());
    pattern p(irrep, 0);
    std::vector<int> map_parent(irrep_dimension, -1),
                     map_multi(irrep_dimension, -1),
                     which_l(irrep_dimension, -1);
    int n_parent = 0, n_multi = 0;

    for (int i = 0; i < irrep_dimension; ++i, ++p) {
        weight v(p.get_weight());

        if (v == statew) {
            map_multi[i] = n_multi++;
        } else for (int l = 1; l < N; ++l) {
            --v(l);
            ++v(l + 1);
            if (v == statew) {
                map_parent[i] = n_parent++;
                which_l[i] = l;
                if (!done[i]) {
                    compute_lower_weight_coeffs(multip_index, i, done);
                }
                break;
            }
            --v(l + 1);
            ++v(l);
        }
    }

    double *irrep_coeffs = new double[n_parent * n_multi];
    assert(irrep_coeffs != NULL);
    memset(irrep_coeffs, 0, n_parent * n_multi * sizeof (double));

    double *prod_coeffs = new double[n_parent * factor1_dimension * factor2_dimension];
    assert(prod_coeffs != NULL);
    memset(prod_coeffs, 0, n_parent * factor1_dimension * factor2_dimension * sizeof (double));

    std::vector<std::vector<int> > map_prodstat(factor1_dimension,
                                                std::vector<int>(factor2_dimension, -1));
    int n_prodstat = 0;

    pattern r(irrep, 0);
    for (int i = 0; i < irrep_dimension; ++i, ++r) {
        if (map_parent[i] >= 0) {
            for (int k = 1, l = which_l[i]; k <= l; ++k) {
                if (r(k, l) > r(k + 1, l + 1) && (k == l || r(k, l) > r(k, l - 1))) {
                    --r(k, l);
                    int h = r.index();
                    ++r(k, l);

                    irrep_coeffs[n_parent * map_multi[h] + map_parent[i]] += r.lowering_coeff(k, l);
                }
            }

            pattern q1(factor1, 0);
            for (int j1 = 0; j1 < factor1_dimension; ++j1, ++q1) {
                pattern q2(factor2, 0);

                for (int j2 = 0; j2 < factor2_dimension; ++j2, ++q2) {
                    if (std::fabs((*this)(j1, j2, multip_index, i)) > EPS) {
                        for (int k = 1, l = which_l[i]; k <= l; ++k) {
                            if (q1(k, l) > q1(k + 1, l + 1) && (k == l || q1(k, l) > q1(k, l - 1))) {
                                --q1(k, l);
                                int h = q1.index();
                                ++q1(k, l);

                                if (map_prodstat[h][j2] < 0) {
                                    map_prodstat[h][j2] = n_prodstat++;
                                }

                                prod_coeffs[n_parent * map_prodstat[h][j2] + map_parent[i]] +=
                                        (*this)(j1, j2, multip_index, i) * q1.lowering_coeff(k, l);
                            }

                            if (q2(k, l) > q2(k + 1, l + 1) && (k == l || q2(k, l) > q2(k, l - 1))) {
                                --q2(k, l);
                                int h = q2.index();
                                ++q2(k, l);

                                if (map_prodstat[j1][h] < 0) {
                                    map_prodstat[j1][h] = n_prodstat++;
                                }

                                prod_coeffs[n_parent * map_prodstat[j1][h] + map_parent[i]] +=
                                        (*this)(j1, j2, multip_index, i) * q2.lowering_coeff(k, l);
                            }
                        }
                    }
                }
            }
        }
    }

    double worksize;
    int lwork = -1, info;

    dgels_("N",
           &n_parent,
           &n_multi,
           &n_prodstat,
           irrep_coeffs,
           &n_parent,
           prod_coeffs,
           &n_parent,
           &worksize,
           &lwork,
           &info);
    assert(info == 0);

    lwork = worksize;
    double *work = new double[lwork];
    assert(work != NULL);

    dgels_("N",
           &n_parent,
           &n_multi,
           &n_prodstat,
           irrep_coeffs,
           &n_parent,
           prod_coeffs,
           &n_parent,
           work,
           &lwork,
           &info);
    assert(info == 0);

    for (int i = 0; i < irrep_dimension; ++i) {
        if (map_multi[i] >= 0) {
            for (int j = 0; j < factor1_dimension; ++j) {
                for (int k = 0; k < factor2_dimension; ++k) {
                    if (map_prodstat[j][k] >= 0) {
                        double x = prod_coeffs[n_parent * map_prodstat[j][k] + map_multi[i]];

                        if (fabs(x) > EPS) {
                            set(j, k, multip_index, i, x);
                        }
                    }
                }
            }

            done[i] = true;
        }
    }

    delete[] work;
    delete[] prod_coeffs;
    delete[] irrep_coeffs;
}

clebsch::coefficients::coefficients(const weight &irrep, const weight &factor1, const weight &factor2) :
        N(irrep.N),
        factor1(factor1),
        factor2(factor2),
        irrep(irrep),
        factor1_dimension(factor1.dimension()),
        factor2_dimension(factor2.dimension()),
        irrep_dimension(irrep.dimension()),
        multiplicity(decomposition(factor1, factor2).multiplicity(irrep)) {
    assert(factor1.N == irrep.N);
    assert(factor2.N == irrep.N);

    compute_highest_weight_coeffs();

    for (int i = 0; i < multiplicity; ++i) {
        std::vector<char> done(irrep_dimension, 0);
        done[irrep_dimension - 1] = true;
        for (int j = irrep_dimension - 1; j >= 0; --j) {
            if (!done[j]) {
                compute_lower_weight_coeffs(i, j, done);
            }
        }
    }
}

double clebsch::coefficients::operator()(int factor1_state,
                                         int factor2_state,
                                         int multiplicity_index,
                                         int irrep_state) const {
    assert(0 <= factor1_state && factor1_state < factor1_dimension);
    assert(0 <= factor2_state && factor2_state < factor2_dimension);
    assert(0 <= multiplicity_index && multiplicity_index < multiplicity);
    assert(0 <= irrep_state && irrep_state < irrep_dimension);

    int coefficient_label[] = { factor1_state,
                                factor2_state,
                                multiplicity_index,
                                irrep_state };
    std::map<std::vector<int>, double>::const_iterator it(
            clzx.find(std::vector<int>(coefficient_label, coefficient_label
                    + sizeof coefficient_label / sizeof coefficient_label[0])));

    return it != clzx.end() ? it->second : 0.0;
}

// sample driver routine

using namespace std;

int main() {
    while (true) {
        int choice, N;

        cout << "What would you like to do?" << endl;
        cout << "1) Translate an i-weight S to its index P(S)" << endl;
        cout << "2) Recover an i-weight S from its index P(S)" << endl;
        cout << "3) Translate a pattern M to its index Q(M)" << endl;
        cout << "4) Recover a pattern M from its index Q(M)" << endl;
        cout << "5) Calculate Clebsch-Gordan coefficients for S x S' -> S''" << endl;
        cout << "6) Calculate all Glebsch-Gordan coefficients for S x S'" << endl;
        cout << "0) Quit" << endl;

        do {
            cin >> choice;
        } while (choice < 0 || choice > 6);

        if (choice == 0) {
            break;
        }

        cout << "N (e.g. 3): ";
        cin >> N;

        switch (choice) {
            case 1: {
                clebsch::weight S(N);
                cout << "Irrep S: ";
                for (int k = 1; k <= N; ++k) {
                    cin >> S(k);
                }
                cout << S.index() << endl;
                break;
            }
            case 2: {
                int P;
                cout << "Index: ";
                cin >> P;
                clebsch::weight S(N, P);
                cout << "I-weight:";
                for (int k = 1; k <= N; ++k) {
                    cout << ' ' << S(k);
                }
                cout << endl;
                break;
            }
            case 3: {
                clebsch::pattern M(N);
                for (int l = N; l >= 1; --l) {
                    cout << "Row l = " << l << ": ";
                    for (int k = 1; k <= l; ++k) {
                        cin >> M(k, l);
                    }
                }
                cout << "Index: " << M.index() + 1 << endl;
                break;
            }
            case 4: {
                clebsch::weight S(N);
                cout << "Irrep S: ";
                for (int i = 1; i <= N; ++i) {
                    cin >> S(i);
                }

                int Q;
                cout << "Index (1..dim(S)): ";
                cin >> Q;
                clebsch::pattern M(S, Q - 1);
                for (int l = N; l >= 1; --l) {
                    for (int k = 1; k <= l; ++k) {
                        cout << M(k, l) << '\t';
                    }
                    cout << endl;
                }
                break;
            }
            case 5: {
                clebsch::weight S(N);
                cout << "Irrep S (e.g.";
                for (int k = N - 1; k >= 0; --k) {
                    cout << ' ' << k;
                }
                cout << "): ";
                for (int k = 1; k <= N; ++k) {
                    cin >> S(k);
                }

                clebsch::weight Sprime(N);
                cout << "Irrep S' (e.g.";
                for (int k = N - 1; k >= 0; --k) {
                    cout << ' ' << k;
                }
                cout << "): ";
                for (int k = 1; k <= N; ++k) {
                    cin >> Sprime(k);
                }

                clebsch::decomposition decomp(S, Sprime);
                cout << "Littlewood-Richardson decomposition S \\otimes S' = \\oplus S'':" << endl;
                cout << "[irrep index] S'' (outer multiplicity) {dimension d_S}" << endl;
                for (int i = 0; i < decomp.size(); ++i) {
                    cout << "[" << decomp(i).index() << "] ";
                    for (int k = 1; k <= N; ++k) {
                        cout << decomp(i)(k) << ' ';
                    }
                    cout << '(' << decomp.multiplicity(decomp(i)) << ") {"
                         << decomp(i).dimension() << "}" << endl;;
                }

                clebsch::weight Sdoubleprime(N);
                for (bool b = true; b; ) {
                    cout << "Irrep S'': ";
                    for (int k = 1; k <= N; ++k) {
                        cin >> Sdoubleprime(k);
                    }
                    for (int i = 0; i < decomp.size(); ++i) {
                        if (decomp(i) == Sdoubleprime) {
                            b = false;
                            break;
                        }
                    }
                    if (b) {
                        cout << "S'' does not occur in the decomposition" << endl;
                    }
                }

                int alpha;
                while (true) {
                    cout << "Outer multiplicity index: ";
                    cin >> alpha;
                    if (1 <= alpha && alpha <= decomp.multiplicity(Sdoubleprime)) {
                        break;
                    }
                    cout << "S'' does not have this multiplicity" << endl;
                }

                string file_name;
                cout << "Enter file name to write to file (leave blank for screen output): ";
                cin.ignore(1234, '\n');
                getline(cin, file_name);

                const clebsch::coefficients C(Sdoubleprime, S, Sprime);
                int dimS = S.dimension(),
                    dimSprime = Sprime.dimension(),
                    dimSdoubleprime = Sdoubleprime.dimension();

                ofstream os(file_name.c_str());
                (file_name.empty() ? cout : os).setf(ios::fixed);
                (file_name.empty() ? cout : os).precision(15);
                (file_name.empty() ? cout : os) << "List of nonzero CGCs for S x S' => S'', alpha" << endl;
                (file_name.empty() ? cout : os) << "Q(M)\tQ(M')\tQ(M'')\tCGC" << endl;
                for (int i = 0; i < dimSdoubleprime; ++i) {
                    for (int j = 0; j < dimS; ++j) {
                        for (int k = 0; k < dimSprime; ++k) {
                            double x = double(C(j, k, alpha - 1, i));

                            if (fabs(x) > clebsch::EPS) {
                                (file_name.empty() ? cout : os) << j + 1 << '\t'
                                    << k + 1 << '\t' << i + 1 << '\t' << x << endl;
                            }
                        }
                    }
                }

                break;
            }
            case 6: {
                clebsch::weight S(N);
                cout << "Irrep S (e.g.";
                for (int k = N - 1; k >= 0; --k) {
                    cout << ' ' << k;
                }
                cout << "): ";
                for (int k = 1; k <= N; ++k) {
                    cin >> S(k);
                }

                clebsch::weight Sprime(N);
                cout << "Irrep S' (e.g.";
                for (int k = N - 1; k >= 0; --k) {
                    cout << ' ' << k;
                }
                cout << "): ";
                for (int k = 1; k <= N; ++k) {
                    cin >> Sprime(k);
                }

                string file_name;
                cout << "Enter file name to write to file (leave blank for screen output): ";
                cin.ignore(1234, '\n');
                getline(cin, file_name);

                ofstream os(file_name.c_str());
                (file_name.empty() ? cout : os).setf(ios::fixed);
                (file_name.empty() ? cout : os).precision(15);

                clebsch::decomposition decomp(S, Sprime);
                (file_name.empty() ? cout : os) <<
                    "Littlewood-Richardson decomposition S \\otimes S' = \\oplus S'':" << endl;
                (file_name.empty() ? cout : os) <<
                    "[irrep index] S'' (outer multiplicity) {dimension d_S}" << endl;
                for (int i = 0; i < decomp.size(); ++i) {
                    (file_name.empty() ? cout : os) << "[" << decomp(i).index() << "] ";
                    for (int k = 1; k <= N; ++k) {
                        (file_name.empty() ? cout : os) << decomp(i)(k) << ' ';
                    }
                    (file_name.empty() ? cout : os) << '(' << decomp.multiplicity(decomp(i)) << ") {"
                         << decomp(i).dimension() << "}" << endl;;
                }

                for (int i = 0; i < decomp.size(); ++i) {
                    const clebsch::coefficients C(decomp(i),S, Sprime);
                    int dimS = S.dimension(),
                        dimSprime = Sprime.dimension(),
                        dimSdoubleprime = decomp(i).dimension();

                    for (int m = 0; m < C.multiplicity; ++m) {
                        (file_name.empty() ? cout : os) << "List of nonzero CGCs for S x S' => S'' = (";
                        for (int j = 1; j <= N; ++j) cout << decomp(i)(j) << (j < N ? ' ' : ')');
                        (file_name.empty() ? cout : os) << ", alpha = " << m + 1 << endl;
                        (file_name.empty() ? cout : os) << "Q(M)\tQ(M')\tQ(M'')\tCGC" << endl;
                        for (int i = 0; i < dimSdoubleprime; ++i) {
                            for (int j = 0; j < dimS; ++j) {
                                for (int k = 0; k < dimSprime; ++k) {
                                    double x = double(C(j, k, m, i));

                                    if (fabs(x) > clebsch::EPS) {
                                        (file_name.empty() ? cout : os) << j  + 1<< '\t'
                                            << k + 1 << '\t' << i + 1 << '\t' << x << endl;
                                    }
                                }
                            }
                        }

                        (file_name.empty() ? cout : os) << endl;
                    }
                }

                break;
            }
        }
    }

    return 0;
}
