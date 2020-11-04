#!/usr/bin/env python
#
# Export pre-exposure task data from an ROI.

import os
import numpy as np

import mindstorm.subjutil as su
from wikisim import task


def main(subject, study_dir, mask, suffix='_stim2'):
    from mvpa2.mappers.zscore import zscore
    from mvpa2.mappers.fx import mean_group_sample
    from wikisim import mvpa

    # load subject data
    sp = su.SubjPath(subject, study_dir)
    vols = task.prex_vols(sp.path('behav', 'log'))

    # load fmri data
    ds = mvpa.load_prex_beta(sp, suffix, mask, verbose=1)

    # zscore
    ds.sa['run'] = vols.run.values
    zscore(ds, chunks_attr='run')

    # average over item presentations
    ds.sa['itemno'] = vols.itemno.to_numpy()
    m = mean_group_sample(['itemno'])
    dsm = ds.get_mapped(m)
    m_vols = vols.groupby('itemno', as_index=False).mean()

    # save data samples and corresponding volume information
    res_dir = os.path.join(sp.study_dir, 'batch', 'glm', 'prex' + suffix,
                           'roi', mask)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    mat_file = os.path.join(res_dir, f'pattern_{subject}.txt')
    tab_file = os.path.join(res_dir, f'pattern_{subject}.csv')
    np.savetxt(mat_file, dsm.samples)
    m_vols.to_csv(tab_file)


if __name__ == '__main__':
    parser = su.SubjParser(include_log=False)
    parser.add_argument('mask', help="name of mask file")
    parser.add_argument('--suffix', '-b', default='_stim2',
                        help="suffix for beta images (_stim2)")
    args = parser.parse_args()

    main(args.subject, args.study_dir, args.mask, suffix=args.suffix)
