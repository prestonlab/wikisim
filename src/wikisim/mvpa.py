"""Measures for use with pymvpa2."""

import os
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as sd
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.measures.base import Measure
import mindstorm.subjutil as su
from mindstorm import prsa


def load_prex_beta(sp, beta_suffix, mask, feature_mask=None, verbose=1):
    """Load pre-exposure task beta series as a dataset."""
    # file with beta series data
    beta_dir = os.path.join(sp.study_dir, 'batch', 'glm',
                            'prex' + beta_suffix, 'beta')
    beta_file = su.impath(beta_dir, sp.subject + '_beta')
    if not os.path.exists(beta_file):
        raise IOError(f'Beta series file does not exist: {beta_file}')
    if verbose:
        print(f'Loading beta series data from: {beta_file}')

    # mask image to select voxels to load
    mask_file = sp.image_path('anatomy', 'bbreg', 'data', mask)
    if not os.path.exists(mask_file):
        raise IOError(f'Mask file does not exist: {mask_file}')
    if verbose:
        print(f'Masking with: {mask_file}')

    if feature_mask is not None:
        # load feature mask
        feature_file = sp.image_path('anatomy', 'bbreg', 'data', feature_mask)
        if not os.path.exists(feature_file):
            raise IOError(f'Feature mask does not exist: {feature_file}')
        if verbose:
            print(f'Using features within: {feature_file}')

        # label voxels with included flag
        ds = fmri_dataset(beta_file, mask=mask_file,
                          add_fa={'include': feature_file})
        ds.fa.include = ds.fa.include.astype(bool)
    else:
        # mark all voxels as included
        ds = fmri_dataset(beta_file, mask=mask_file)
        ds.fa['include'] = np.ones(ds.shape[1], dtype=bool)
    return ds


class ItemPartialRSA(Measure):
    """Test for unique contribution of different item models."""
    def _call(self, ds):
        pass

    def __init__(self, model_rdms, n_perm, min_voxels=10):
        Measure.__init__(self)
        self.perm = prsa.init_pRSA(n_perm, model_rdms)
        self.min_voxels = min_voxels
        self.n_model = len(model_rdms)

    def __call__(self, dataset):
        zstat = np.zeros(self.n_model)
        if np.count_nonzero(dataset.fa.include) < 10:
            return (*zstat, 0)
        dataset = dataset[:, dataset.fa.include]

        data_vec = st.rankdata(sd.pdist(dataset, 'correlation'))
        for i in range(self.n_model):
            stat = prsa.perm_partial(data_vec, self.perm['model_mats'][i],
                                     self.perm['model_resid'][i])
            zstat[i] = prsa.perm_z(stat)
        return (*zstat, 1)
