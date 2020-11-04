"""Work with representational dissimilarity models."""
import math
import os

import numpy as np
import pandas as pd
from scipy import io as sio
from scipy.spatial import distance as sd


def haversine(u, v):
    """
    Calculate the great circle distance between two points.

    Points are specified in decimal degrees.
    """
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [u[0], u[1], v[0], v[1]])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def get_category_ind(category):
    """Get indices for a category."""
    if category == 'face':
        ind = slice(0, 60)
    elif category == 'scene':
        ind = slice(60, 120)
    elif category == 'both':
        ind = slice(0, 120)
    else:
        raise ValueError(f'Invalid category: {category}')
    return ind


def save_model_rdm(model_dir, model_name, items=None, vectors=None, df=None,
                   distance='correlation'):
    """Save a model representational dissimilarity matrix."""
    if df is not None:
        items = df.stim.to_numpy()
        vectors = df.filter(like='dim').to_numpy()
    else:
        if items is None:
            raise ValueError('Items not specified.')
        elif vectors is None:
            raise ValueError('Vectors not specified.')

    rdm = sd.squareform(sd.pdist(vectors, distance))
    mat = {'rdm': rdm, 'vectors': vectors, 'items': items}
    model_file = os.path.join(model_dir, f'mat_{model_name}.mat')
    sio.savemat(model_file, mat)


def load_model_rdm(model_dir, model_name):
    """Load a model representational dissimilarity matrix."""
    # load the MAT-file
    model_file = os.path.join(model_dir, f'mat_{model_name}.mat')
    mat = sio.loadmat(model_file)
    if 'rdm' not in mat:
        raise ValueError(f'Model RDM file not in standard format: {model_file}')
    rdm = mat['rdm']

    # check that the rdm is sensible
    assert np.allclose(np.diag(rdm), 0)
    assert rdm.shape[0] == rdm.shape[1]

    # get items in list format
    f = mat['items']
    if len(f) == 1:
        items = [i[0] for i in f[0]]
    elif type(f[0]) == np.str_:
        items = [i.strip() for i in f]
    else:
        items = [i[0][0] for i in f]
    assert rdm.shape[0] == len(items)

    return rdm, items


def load_model_set(model_dir, model_names, model_labels=None, item_groups=None):
    """Load a set of models."""
    if model_labels is None:
        model_labels = model_names

    if item_groups is None:
        item_groups = {'face': slice(None, 60), 'scene': slice(60, None)}

    # load all full rdms
    model_rdms = {}
    for model, model_label in zip(model_names, model_labels):
        rdm, items = load_model_rdm(model_dir, model)
        model_rdms[model_label] = rdm

    # split by item group
    model_set = {}
    for group, ind in item_groups.items():
        group_set = {}
        for model, rdm in model_rdms.items():
            group_set[model] = rdm[ind, ind]
        model_set[group] = group_set
    return model_set


def load_face_features(features_csv):
    """Load features from face csv."""
    features = pd.read_csv(features_csv)

    # occupations for each person and a master set of all occupations
    occ_set = set()
    all_occ = []
    for occ in features['occupations']:
        person_occ = [o.strip() for o in occ.split(',')]
        occ_set = occ_set | set(person_occ)
        all_occ.append(person_occ)
    occ_list = list(occ_set)

    # place occupations into a matrix
    occ_mat = np.zeros((len(all_occ), len(occ_list)), dtype=int)
    for i, person_occ in enumerate(all_occ):
        ind = [occ_list.index(o) for o in person_occ]
        occ_mat[i, ind] = 1
    occ = pd.DataFrame(occ_mat, columns=occ_list, index=features['name'])
    return features, occ


def match_rdm(x):
    dsm = x[:, None] != x.T
    return dsm.astype(int)


def load_face_rdms(features_csv, pool_file=None):
    """Load RDMs based on face features."""
    features, occ = load_face_features(features_csv)

    # run optional consistency check
    if pool_file is not None:
        pool = pd.read_csv(pool_file)
        pool_faces = pool.loc[pool['category'] == 'face']
        np.testing.assert_array_equal(features['name'], pool_faces['stim'])

    age = features['age'].to_numpy()
    rdms = {
        'gender': match_rdm(features['gender'].to_numpy()),
        'age': sd.squareform(sd.pdist(age[:, None], 'euclidean')),
        'occupation': sd.squareform(sd.pdist(occ.to_numpy(), 'hamming')),
        'main': match_rdm(features['main'].to_numpy())
    }
    return rdms


def load_scene_rdms(features_csv, pool_file=None):
    """Load RDMs based on scene features."""
    features = pd.read_csv(features_csv)

    # run optional consistency check
    if pool_file is not None:
        pool = pd.read_csv(pool_file)
        pool_scene = pool.loc[pool['category'] == 'scene']
        np.testing.assert_array_equal(features['name'], pool_scene['stim'])

    year = features['year'].to_numpy()[:, None]
    coord = features[['latitude', 'longitude']].to_numpy()
    age = sd.squareform(sd.pdist(year, 'euclidean'))
    age[:30, 30:] = 10000
    age[30:, :30] = 10000

    rdms = {
        'subcategory': match_rdm(features['subcategory'].to_numpy()),
        'type': match_rdm(features['type'].to_numpy()),
        'year': age,
        'geo': sd.squareform(sd.pdist(coord, haversine)),
        'region': match_rdm(features['region'].to_numpy()),
        'state': match_rdm(features['state'].to_numpy()),
    }
    return rdms


def load_feature_rdms(model_dir, category):
    """Load features RDMs for a category."""
    if category == 'face':
        features_file = os.path.join(model_dir, 'face_features.csv')
        rdms = load_face_rdms(features_file)
    elif category == 'scene':
        features_file = os.path.join(model_dir, 'scene_features.csv')
        rdms = load_scene_rdms(features_file)
    else:
        raise ValueError(f'Invalid category: {category}')
    return rdms


def load_category_rdms(model_dir, category, model_names, model_labels=None,
                       subsample=None):
    """Load RDMs for a category."""
    feature_rdms = load_feature_rdms(model_dir, category)
    ind = get_category_ind(category)

    if subsample is not None:
        sub_ind = subsample.query(f'category == "{category}"').index.to_numpy()
        sub_ind -= ind.start
    else:
        sub_ind = None

    # compile list of rdms
    if model_labels is None:
        model_labels = model_names
    model_rdms = {}
    for name, label in zip(model_names, model_labels):
        if name in feature_rdms:
            rdm = feature_rdms[name]
        else:
            rdm_full = load_model_rdm(model_dir, name)[0]
            rdm = rdm_full[ind, ind]
        if sub_ind is not None:
            rdm = rdm[np.ix_(sub_ind, sub_ind)]
        model_rdms[label] = rdm
    return model_rdms
