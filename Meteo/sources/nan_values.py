# pylint: disable=line-too-long

""" functions_nan module """

################################################################
## Copyright(C) 2024, Charles Theetten, <chalimede@proton.me> ##
################################################################

################################################################################

import  numpy                   as np
import  pandas                  as pd

from    scipy                   import spatial
from    sklearn.cluster         import KMeans
from    sklearn.impute          import KNNImputer

################################################################################

def fillna_kmeans(data, n_clusters):
    """ fill nan values with kmeans imputation """
    cat_data    = data.select_dtypes(["object", "datetime"]).reset_index(drop = True)
    num_data    = data.select_dtypes(np.number).reset_index(drop = True)
    clean_data  = num_data.copy().dropna()

    kmeans      = KMeans(n_clusters = n_clusters, max_iter = 1000)
    kmeans.fit(clean_data)

    centroids   = kmeans.cluster_centers_

    for i in range(0, len(num_data)):
        if num_data.loc[i, :].isna().any():
            nan_vector      = num_data.loc[i, :].values                     # récupération du vecteur avec valeurs manquantes
            nan_index       = np.argwhere(np.isnan(nan_vector)).reshape(-1) # récupération index des valeurs manquantes
            clean_vector    = nan_vector[~np.isnan(nan_vector)]             # vecteur delesté de ses valeurs manquantes
            clean_centroids = []                                            # liste des centroids libérés de leurs éléments correspondant aux indices des nan du vecteur initial
            for c in centroids:
                clean_centroids.append(np.delete(c, nan_index))             # suppression des éléments aux indices nan du vecteur initial pour chaque centroid
            tree                = spatial.KDTree(clean_centroids)           # comparaison spatiale des vecteurs
            nearest_centroid    = centroids[tree.query(clean_vector)[1]]    # récupération du centroid le plus proche du vecteur initial
            clean_centroids.clear()
            for j in nan_index:
                nan_vector[j] = nearest_centroid[j]                         # mise à jour des valeurs manquantes dans le vecteur initial
            num_data.loc[i, :] = nan_vector
    return pd.concat([cat_data, num_data], axis = 1)

def fillna_knn(data, neighbors, weights, metric):
    """ fillna_knn nan values with KNN imputer """
    cat_data    = data.select_dtypes(["object", "datetime"]).reset_index(drop = True)
    num_data    = data.select_dtypes(np.number).reset_index(drop = True)
    imputer     = KNNImputer(n_neighbors = neighbors, weights = weights, metric = metric, keep_empty_features = True)
    num_data    = pd.DataFrame(imputer.fit_transform(num_data), columns = num_data.columns)
    return  pd.concat([cat_data, num_data], axis = 1)

def fillna_rain_today(data):
    """ fill nan values on RainToday variable """
    data["RainToday"] = np.where((data["Rainfall"] > 1) & (data["RainToday"].isnull()), 1, data["RainToday"])
    data["RainToday"] = np.where((data["Rainfall"] <= 1) & (data["RainToday"].isnull()), 0, data["RainToday"])
    return data
