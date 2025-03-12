import numpy as np
from pyrepo_mcda.mcda_methods.mcda_method import MCDA_method
from pyrepo_mcda import normalizations as norms
from pyrepo_mcda import distance_metrics as dists
from pyrepo_mcda import correlations as corrs
from pyrepo_mcda.additions import rank_preferences

import pandas as pd
import matplotlib.pyplot as plt


class TOPSIS(MCDA_method):
    def __init__(self, normalization_method = norms.minmax_normalization, distance_metric = dists.euclidean, distance_params = None):
        """
        Create the TOPSIS method object and select normalization method `normalization_method` and
        distance metric `distance metric`.

        Parameters
        -----------
            normalization_method : function
                method for decision matrix normalization chosen from `normalizations`

            distance_metric : functions
                method for calculating the distance between two vectors
        """
        self.normalization_method = normalization_method
        self.distance_metric = distance_metric
        self.distance_params = distance_params


    def __call__(self, matrix, weights, types):
        """
        Score alternatives provided in decision matrix `matrix` with m alternatives in rows and 
        n criteria in columns using criteria `weights` and criteria `types`.

        Parameters
        ----------
            matrix : ndarray
                Decision matrix with m alternatives in rows and n criteria in columns.
            weights: ndarray
                Vector with criteria weights. Sum of weights must be equal to 1.
            types: ndarray
                Vector with criteria types. Profit criteria are represented by 1 and cost by -1.

        Returns
        -------
            ndrarray
                Vector with preference values of each alternative. The best alternative has the highest preference value. 

        Examples
        ---------
        >>> topsis = TOPSIS(normalization_method = minmax_normalization, distance_metric = euclidean)
        >>> pref = topsis(matrix, weights, types)
        >>> rank = rank_preferences(pref, reverse = True)
        """
        TOPSIS._verify_input_data(matrix, weights, types)
        return TOPSIS._topsis(matrix, weights, types, self.normalization_method, self.distance_metric, self.distance_params)


    @staticmethod
    def _topsis(matrix, weights, types, normalization_method, distance_metric, distance_params):
        # Normalize matrix using chosen normalization (for example linear normalization)
        norm_matrix = normalization_method(matrix, types)

        # Multiply all rows of normalized matrix by weights
        weighted_matrix = norm_matrix * weights

        # Calculate vectors of PIS (ideal solution) and NIS (anti-ideal solution)
        pis = np.max(weighted_matrix, axis=0)
        nis = np.min(weighted_matrix, axis=0)

        # Calculate chosen distance of every alternative from PIS and NIS using chosen distance metric `distance_metric` from `distance_metrics`
        print('Name of distance metric is ', distance_metric.__name__)
        if distance_metric.__name__ == 'ROR_distance' or distance_metric.__name__ == 'minkowski':
            Dp = np.array([distance_metric(x, pis, distance_params) for x in weighted_matrix])
            Dm = np.array([distance_metric(x, nis, distance_params) for x in weighted_matrix])
        else:
            Dp = np.array([distance_metric(x, pis) for x in weighted_matrix])
            Dm = np.array([distance_metric(x, nis) for x in weighted_matrix])

        C = Dm / (Dm + Dp)
        return C
    

# Manhattan distance metric
def manhattan(A, B):
    return np.sum(np.abs(A - B))
    

# Chebyshev distance metric
def chebyshev(A, B):
    return max(abs(A - B))

# Manhattan

# dla x czyli A i B z indexem 0 oraz dla y czyli z indexem 1
# for x is A and B with index 0; for y is A and B with index 1
def euclidean(A, B):
    return np.sqrt(np.square(A[0] - B[0]) + np.square(A[1] - B[1]))

# Minkowski
def minkowski(A, B, params):
    p = params[0]
    r = params[1]
    return np.sum((np.abs(A - B)**p))**(1/r)

# Zielniewicz ROR distance metric
# alpha from 0 (Chebyshev) to 1 (Euclidean)
def ROR_distance(A, B, alpha):
    return alpha * np.sqrt(np.sum(np.square(A - B))) + (1 - alpha) * max(abs(A - B))


# plot line sensitivity analysis weights
def plot_sensitivity(vec_ticks, data_sust, metric_name):

    vec = np.arange(0, len(vec_ticks))
    plt.figure(figsize = (12, 4))
    for j in range(data_sust.shape[0]):
        
        plt.plot(vec, data_sust.iloc[j, :], '-o', linewidth = 2)
        ax = plt.gca()
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        plt.annotate(data_sust.index[j], (x_max, data_sust.iloc[j, -1]),
                        #style='italic',
                        horizontalalignment='left', fontsize = 14)

    
    plt.ylabel("Rank", fontsize = 14)
    plt.vlines(x = [4, 9, 14, 19, 24, 29], ymin = y_min, ymax = y_max, colors='k', linestyles='--')
    plt.yticks(ticks = np.arange(1, data_sust.shape[0] + 1, 1), fontsize = 14)
    
    plt.xticks(ticks=vec, labels = vec_ticks, fontsize = 11)


    ax2 = ax.secondary_xaxis('top')
    ax2.tick_params(axis='x', color='k')
    ax2.set_xticks(ticks = [2, 7, 12, 17, 22, 27], labels = [r'$C_1$', r'$C_2$', r'$C_3$', r'$C_4$', r'$C_5$', r'$C_6$'], minor=False, fontsize = 14)
    plt.gca().invert_yaxis()
    plt.title(metric_name + ' distance metric', fontsize = 14)
    plt.grid(linestyle = ':')
    plt.tight_layout()
    plt.savefig('./results/sa_visualization_' + metric_name + '.pdf')
    plt.show()


def main():
    dataset = pd.read_csv('dataset_clouds.csv', index_col = 'Cloud')
    df = dataset.iloc[:len(dataset) - 2, :]
    weights = dataset.iloc[len(dataset) - 2, :].to_numpy()
    types = dataset.iloc[len(dataset) - 1, :].to_numpy()

    matrix = df.to_numpy()

    print(df)
    print(weights)
    print(types)

    index = [r'$A_{' + str(i) + '}$' for i in range(1, 4 + 1)]
    df_pref = pd.DataFrame(index = index)
    df_rank = pd.DataFrame(index = index)

    # manhattan
    print('Manhattan')
    topsis = TOPSIS(distance_metric=dists.manhattan)
    pref_manhattan = topsis(matrix, weights, types)
    rank = rank_preferences(pref_manhattan, reverse = True)
    print(pref_manhattan)
    print(rank)

    # euclidean
    print('Euclidean')
    topsis = TOPSIS(distance_metric=dists.euclidean)
    pref_euclidean = topsis(matrix, weights, types)
    rank = rank_preferences(pref_euclidean, reverse = True)
    print(pref_euclidean)
    print(rank)

    # chebyshev
    print('Chebyshev')
    topsis = TOPSIS(distance_metric=chebyshev)
    pref_chebyshev = topsis(matrix, weights, types)
    rank = rank_preferences(pref_chebyshev, reverse = True)
    print(pref_chebyshev)
    print(rank)


    for p in range(1, 11):
        print(f'Minkowski {str(p)}')
        topsis = TOPSIS(distance_metric=minkowski, distance_params = (p, p))
        pref = topsis(matrix, weights, types)
        rank = rank_preferences(pref, reverse = True)
        print(pref)
        print(rank)
        df_pref[f'p = {str(p)}'] = pref
        df_rank[f'p = {str(p)}'] = rank

    df_pref.to_csv('./results/df_pref.csv')
    df_rank.to_csv('./results/df_rank.csv')

    # ================================================================================
    # numerical experiment
    corr_manhattan = []
    corr_euclidean = []
    corr_chebyshev = []

    p_params = np.arange(1, 41)

    for p in p_params:
        topsis = TOPSIS(distance_metric=minkowski, distance_params = (p, p))
        pref = topsis(matrix, weights, types)

        corr_manhattan.append(corrs.pearson_coeff(pref, pref_manhattan))
        corr_euclidean.append(corrs.pearson_coeff(pref, pref_euclidean))
        corr_chebyshev.append(corrs.pearson_coeff(pref, pref_chebyshev))

    fig, ax = plt.subplots(figsize=(8, 4))
    plt.plot(p_params, corr_manhattan, label = 'Manhattan', linewidth = 3)
    plt.plot(p_params, corr_euclidean, label = 'Euclidean', linewidth = 3)
    plt.plot(p_params, corr_chebyshev, label = 'Chebyshev', linewidth = 3)
    plt.xlabel(r'$p$' + ' parameter of Minkowski metric', fontsize = 12)
    plt.ylabel('Pearson correlation value', fontsize = 12)
    plt.title('Correlation of TOPSIS scores for Minkowski metric and other metrics', y = 1.2, fontsize = 12)
    # plt.legend(bbox_to_anchor=(1.01, 1),
    #                      loc='upper left', borderaxespad=0., title = 'Compared metrics:')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
    ncol=3, mode="expand", borderaxespad=0., edgecolor = 'black', fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)
    plt.grid(True, linestyle = '-.')
    plt.tight_layout()
    plt.savefig('./results/correlations_metrics.pdf')
    plt.show()


    # Analiza wrazliwosci - Sensitivity analysis
    # Manhattan
    sa_results = pd.DataFrame(index = index)
    weights_vector = np.arange(0.05, 1, 0.2)
    print(weights_vector)
    n = 6
    for j in range(n):
        for w in weights_vector:
            w_val = 1 - w
            ww_val = w_val / (n - 1)
            weights = np.ones(n) * ww_val
            weights[j] = w
            print('Weights')
            print(weights)

            topsis = TOPSIS(distance_metric=dists.manhattan)
            pref = topsis(matrix, weights, types)
            rank = rank_preferences(pref, reverse = True)
            sa_results['C' + str(j + 1) + ' ' + str(np.round(w, 2))] = rank

    print(sa_results)
    sa_results.to_csv('./results/sa_results_manhattan.csv')
    str_weights_vector = [str(int(w * 100)) + '%' for w in weights_vector]
    str_weights_vector = str_weights_vector * 6

    plot_sensitivity(str_weights_vector, sa_results, 'Manhattan')

    # Euclidean
    sa_results = pd.DataFrame(index = index)
    weights_vector = np.arange(0.05, 1, 0.2)
    print(weights_vector)
    n = 6
    for j in range(n):
        for w in weights_vector:
            w_val = 1 - w
            ww_val = w_val / (n - 1)
            weights = np.ones(n) * ww_val
            weights[j] = w
            print('Weights')
            print(weights)

            topsis = TOPSIS(distance_metric=dists.euclidean)
            pref = topsis(matrix, weights, types)
            rank = rank_preferences(pref, reverse = True)
            sa_results['C' + str(j + 1) + ' ' + str(np.round(w, 2))] = rank

    print(sa_results)
    sa_results.to_csv('./results/sa_results_euclidean.csv')
    plot_sensitivity(str_weights_vector, sa_results, 'Euclidean')

    # Chebyshev
    sa_results = pd.DataFrame(index = index)
    weights_vector = np.arange(0.05, 1, 0.2)
    print(weights_vector)
    n = 6
    for j in range(n):
        for w in weights_vector:
            w_val = 1 - w
            ww_val = w_val / (n - 1)
            weights = np.ones(n) * ww_val
            weights[j] = w
            print('Weights')
            print(weights)

            topsis = TOPSIS(distance_metric=chebyshev)
            pref = topsis(matrix, weights, types)
            rank = rank_preferences(pref, reverse = True)
            sa_results['C' + str(j + 1) + ' ' + str(np.round(w, 2))] = rank

    print(sa_results)
    sa_results.to_csv('./results/sa_results_chebyshev.csv')
    plot_sensitivity(str_weights_vector, sa_results, 'Chebyshev')


if __name__ == '__main__':
    main()