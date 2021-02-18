#include "specc.hpp"

#include "dkm/dkm.hpp"

#include <cassert>

auto clusters_sizes(const std::vector<uint32_t>& label)
{
    auto res = std::vector<int>(label.size());
    auto n   = 0;
    for(auto idx : label)
    {
        res[idx]++;
        n = std::max(n, static_cast<int>(idx + 1));
    }
    res.resize(n);
    return std::make_tuple(n, res);
}

auto crossprod(const arma::mat& mat)
{
    arma::mat res(mat.n_cols, mat.n_cols);
    for(size_t i = 0; i < mat.n_cols; i++)
    {
        for(size_t j = 0; j < mat.n_cols; j++)
        {
            arma::vec a = mat.col(i);
            arma::vec b = mat.col(j);
            res(i, j) = arma::sum(a % b);
        }
    }
    return res;
}

// This an adaptation of R specc function.
// Only the relevant part were ported

// FIXME: avoid performing row wise operations
auto specc(const arma::mat& x, int nc)
{
    assert(x.n_rows >= static_cast<size_t>(nc));
    arma::vec dota = arma::sum(x % x, 1) / 2;
    arma::mat dis  = crossprod(x.t());
    arma::vec s    = arma::vec(x.n_rows);

    for(size_t i = 0; i < x.n_rows; i++)
    {
        dis.row(i) = 2 * (-dis.row(i).t() + dota + dota[i]).t();
    }
    // fix numerical prob.
    std::transform(dis.begin(), dis.end(), dis.begin(), [](auto val)
    {
    return std::max(0.0, val);
    });

    for(size_t i = 0; i < x.n_rows; i++)
    {
        arma::vec temp = arma::sort(dis.row(i).t());
        temp.resize(5);
        s[i] = arma::median(arma::sqrt(temp));
    }

    // Compute Affinity Matrix
    arma::mat km = arma::exp((-dis) / (s * s.t()));

    arma::vec d = 1.0 / arma::sqrt(arma::sum(km, 1));
    arma::mat l = km * arma::diagmat(d);
    l.each_col() %= d;
    arma::mat xi;
    arma::vec place_holder;
    arma::eig_sym(place_holder, xi, l);

    xi = xi.cols(xi.n_cols - nc, xi.n_cols - 1);
    arma::vec temp = arma::sqrt(arma::sum(xi % xi, 1));
    arma::mat yi = -xi;
    yi.each_col() /= temp;

    // This is a fork of dkm, which changes may not have been merged to the original.
    // By the time you are reading this I may have added performance improvements
    // https://github.com/Eleobert/dkm
    auto [means, label] = dkm::kmeans_lloyd(dkm::as_matrix(yi.memptr(), yi.n_rows, yi.n_cols), nc);

    auto [n, sizes] = clusters_sizes(label);
    auto centers = arma::mat(n, x.n_cols, arma::fill::zeros);

    for(size_t i = 0; i < label.size(); i++)
    {
        centers.row(label[i]) += x.row(i) / sizes[label[i]];
    }
    return std::make_tupe(label, centers);
}
