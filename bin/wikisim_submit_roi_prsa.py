#!/usr/bin/env python
#
# Submit searchlight partial RSA analyses.

import os
import argparse
from wikisim import rsa


def submit_roi(roi, model_set, categories=None, match_fam=False):

    # settings
    n_perm = 100000

    # expand shortened model codes
    full_name = {'hmx': 'hmax', 'sub': 'subcat', 'w2v': 'wiki_w2v',
                 'use': 'wiki_use1', 'vem': 'vem', 'sem': 'sem', 'geo': 'geo',
                 'occ': 'main', 'typ': 'type', 'yea': 'year',
                 'reg': 'region', 'sta': 'state'}
    models = []
    for name in model_set.split('_'):
        model = full_name[name] if name in full_name else name
        models.append(model)
    model_names = '-'.join(models)

    # relevant categories
    if categories is None:
        if 'geo' in models:
            categories = ['scene']
        else:
            categories = ['face', 'scene']

    # subjects
    subject_list = os.environ['WSUBJIDS']
    subjects = subject_list.split(':')

    for category in categories:
        opt = f'-b _stim2 -c {category} -p {n_perm}'
        res_name = f'prex_prsa_{model_set}_{category}'
        if match_fam:
            opt += ' -f'
            res_name += '_match'
        inputs = f'{roi} {res_name} {model_names}'
        for subject in subjects:
            print(f'wikisim_roi_prsa.py {subject} {inputs} {opt}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Print code to run searchlights.")
    parser.add_argument(
        'model_set',
        help="comma separated list of model sets to use in short name form")
    parser.add_argument('roi_group',
                        help="regions to search within (func,pmat)")
    parser.add_argument('--categories', '-c', help="categories to test")
    parser.add_argument('--match-fam', '-f', action='store_true',
                        help="sub-sample to match familiarity")
    args = parser.parse_args()

    for m in args.model_set.split(','):
        for roi_group in args.roi_group.split(','):
            if roi_group in ['func', 'pmat']:
                rois = rsa.get_rois(roi_group).index.to_list()
            else:
                rois = [roi_group]
            for r in rois:
                submit_roi(r, m, args.categories.split(','), args.match_fam)
