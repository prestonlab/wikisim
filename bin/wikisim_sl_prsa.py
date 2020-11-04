#!/usr/bin/env python
#
# Run partial representational similarity analysis in a searchlight.

import os

import mindstorm.subjutil as su

from wikisim import task
from wikisim import model


def main(subject, study_dir, mask, feature_mask, models, category, res_name,
         suffix='_stim_fix2', radius=3, n_perm=1000, n_proc=None):
    from mvpa2.mappers.zscore import zscore
    from mvpa2.mappers.fx import mean_group_sample
    from mvpa2.measures.searchlight import sphere_searchlight
    from mvpa2.datasets.mri import map2nifti
    from wikisim import mvpa

    # load subject data
    sp = su.SubjPath(subject, study_dir)
    vols = task.prex_vols(sp.path('behav', 'log'))

    # load fmri data
    ds = mvpa.load_prex_beta(sp, suffix, mask, feature_mask=feature_mask,
                             verbose=1)

    # zscore
    ds.sa['run'] = vols.run.values
    zscore(ds, chunks_attr='run')

    # average over item presentations
    ds.sa['itemno'] = vols.itemno.to_numpy()
    m = mean_group_sample(['itemno'])
    dsm = ds.get_mapped(m)

    # get items of interest
    if category == 'face':
        cond = [1, 2]
    elif category == 'scene':
        cond = [3, 4]
    else:
        ValueError(f'Invalid category code: {category}')
    include = vols.groupby('itemno').first()['cond'].isin(cond)

    # get models of interest
    model_dir = os.path.join(study_dir, 'batch', 'models3')
    model_names = models.split('-')
    model_rdms_dict = model.load_category_rdms(model_dir, category, model_names)
    model_rdms = [model_rdms_dict[name] for name in model_names]

    # set up searchlight
    m = mvpa.ItemPartialRSA(model_rdms, n_perm)
    sl = sphere_searchlight(m, radius=radius, nproc=n_proc)
    sl_map = sl(dsm[include])

    nifti_include = map2nifti(ds, sl_map[-1])
    for i, name in enumerate(model_names):
        # save zstat map
        res_dir = sp.path('rsa', f'{res_name}_{name}')
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        filepath = os.path.join(res_dir, 'zstat.nii.gz')
        nifti = map2nifti(ds, sl_map[i])
        nifti.to_filename(filepath)

        # save mask of included voxels
        include_file = os.path.join(res_dir, 'included.nii.gz')
        nifti_include.to_filename(include_file)


if __name__ == '__main__':

    parser = su.SubjParser(include_log=False)
    parser.add_argument('mask', help="name of mask for searchlight centers")
    parser.add_argument('feature_mask',
                        help="name of mask for included voxels")
    parser.add_argument('models',
                        help="models to include (e.g., hmax-wiki_w2v)")
    parser.add_argument('category', help="category to include (face,scene)")
    parser.add_argument('res_name', help="name of results directory")
    parser.add_argument('--suffix', '-b', default='_stim2',
                        help="suffix for beta images (_stim2)")
    parser.add_argument('--radius', '-r', type=int, default=3,
                        help="searchlight radius")
    parser.add_argument('--n-perm', '-p', type=int, default=10000,
                        help="number of permutations to run (10000)")
    parser.add_argument('--n-proc', '-n', type=int, default=None,
                        help="processes for searchlight")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.mask, args.feature_mask,
         args.models, args.category, args.res_name,
         suffix=args.suffix, radius=args.radius, n_perm=args.n_perm,
         n_proc=args.n_proc)
