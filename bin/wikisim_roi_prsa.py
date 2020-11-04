#!/usr/bin/env python
#
# Run partial representational similarity analysis on a region.

import os
import numpy as np
import scipy.stats as st
import scipy.spatial.distance as sd
import pandas as pd

import mindstorm.subjutil as su
from mindstorm import prsa

from wikisim import rsa
from wikisim import model


def main(subject, study_dir, roi, res_name, models, category='both',
         suffix='_stim2', n_perm=100000, match_fam=False):

    # load ROI data and volume information
    input_dir = os.path.join(study_dir, 'batch', 'glm', 'prex' + suffix,
                             'roi', roi)
    patterns, vols = rsa.load_roi_pattern(input_dir, subject)

    # get items of interest
    if category == 'face':
        cond = [1, 2]
    elif category == 'scene':
        cond = [3, 4]
    else:
        ValueError(f'Invalid category code: {category}')
    include = np.isin(vols.cond.to_numpy(), cond)

    # subsample to match familiarity between subcategories
    model_dir = os.path.join(study_dir, 'batch', 'models3')
    if match_fam:
        subsample = pd.read_csv(os.path.join(model_dir, 'subsample.csv'),
                                index_col=0)
        vol_stim_id = vols['itemno'].to_numpy() - 1
        subsample_stim_id = subsample.index.to_numpy()
        include &= np.isin(vol_stim_id, subsample_stim_id)
    else:
        subsample = None

    # get models of interest
    model_names = models.split('-')
    model_rdms_dict = model.load_category_rdms(
        model_dir, category, model_names, subsample=subsample
    )
    model_rdms = [model_rdms_dict[name] for name in model_names]

    # initialize the permutation test
    perm = prsa.init_pRSA(n_perm, model_rdms)
    data_vec = st.rankdata(sd.pdist(patterns[include, :], 'correlation'))
    n_model = len(model_rdms)

    # calculate permutation correlations
    zstat = np.zeros(n_model)
    for i in range(n_model):
        stat = prsa.perm_partial(data_vec, perm['model_mats'][i],
                                 perm['model_resid'][i])
        zstat[i] = prsa.perm_z(stat)

    # save results
    df = pd.DataFrame({'zstat': zstat}, index=model_names)
    res_dir = os.path.join(study_dir, 'batch', 'rsa', res_name, 'roi', roi)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    res_file = os.path.join(res_dir, f'zstat_{subject}.csv')
    df.to_csv(res_file)


if __name__ == '__main__':

    parser = su.SubjParser(include_log=False)
    parser.add_argument('roi', help="name of roi")
    parser.add_argument('res_name', help="name of results directory")
    parser.add_argument('models',
                        help="models to include (e.g., hmax-subcat-wiki_w2v)")
    parser.add_argument('--category', '-c', default='both',
                        help="category to include (face,scene,[both])")
    parser.add_argument('--suffix', '-b', default='_stim2',
                        help="suffix for beta images (_stim2)")
    parser.add_argument('--n-perm', '-p', type=int, default=100000,
                        help="number of permutations to run (100000)")
    parser.add_argument('--match-fam', '-f', action='store_true',
                        help="sub-sample to match familiarity")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.roi, args.res_name, args.models,
         category=args.category, suffix=args.suffix, n_perm=args.n_perm,
         match_fam=args.match_fam)
