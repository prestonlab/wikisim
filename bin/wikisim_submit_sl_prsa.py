#!/usr/bin/env python
#
# Submit searchlight partial RSA analyses.

import os
import argparse


def main(step, model_set, roi, n_proc=1, category='both'):

    # study variables
    study_dir = os.environ['STUDYDIR']
    subject_list = os.environ['WSUBJIDS']
    subjects = subject_list.split(':')
    template_dir = os.path.join(study_dir, 'gptemplate', 'highres_brain_all')

    # expand shortened model codes
    full_name = {'hmx': 'hmax', 'sub': 'subcat', 'w2v': 'wiki_w2v',
                 'use': 'wiki_use1', 'vem': 'vem', 'sem': 'sem', 'geo': 'geo',
                 'occ': 'main', 'typ': 'type', 'yea': 'year',
                 'reg': 'region', 'sta': 'state'}
    models = [full_name[name] for name in model_set.split('_')]

    # relevant categories
    categories = ['face', 'scene'] if category == 'both' else [category]

    if step == 'sl':
        # searchlight settings
        suffix = '_stim2'
        radius = 3
        n_perm = 10000
        model_names = '-'.join(models)

        # expand roi to a mask, feature mask pair
        masks = {'hip': ('b_hip_fshs_dil3', 'b_hip_fshs')}
        mask, feat = masks[roi]

        flags = f'-r {radius} -b {suffix} -p {n_perm} -n {n_proc}'
        for category in categories:
            for subject in subjects:
                inputs = f'{subject} {mask} {feat} {model_names} {category}'
                res_name = f'prex_prsa_{model_set}_{category}_{roi}'
                print(f'wikisim_sl_prsa.py {inputs} {res_name} {flags}')

    elif step == 'randomise':
        # randomise settings
        template_mask = os.path.join(template_dir,
                                     'gp_template_mni_affine_mask.nii.gz')
        n_zstat_perm = 10000
        interp = 'Linear'

        flags = f'-n {n_zstat_perm} -m {template_mask} -a 2 -i {interp} -o'
        for category in categories:
            for model in models:
                res_name = f'prex_prsa_{model_set}_{category}_{roi}_{model}'
                inputs = f'{flags} rsa/{res_name} {subject_list}'
                print(f'zstat_randomise.sh {inputs}')

    elif step == 'svc':
        # svc settings
        template = os.path.join(template_dir, 'gp_template_mni_affine.nii.gz')
        glm = 'prex_stim2'
        svc_masks = {'hip': 'b_hip_fshs_dil1'}
        svc_mask = svc_masks[roi]

        flags = f'-o -t {template}'
        for category in categories:
            for model in models:
                res_name = f'prex_prsa_{model_set}_{category}_{roi}_{model}'
                inputs = f'{flags} rsa/{res_name} {glm} {svc_mask}'
                print(f'apply_clustsim_svc.sh {inputs}')

    else:
        raise ValueError(f'Invalid step: {step}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Print code to run searchlights.")
    parser.add_argument('step', help="step to run (sl, randomise, svc)")
    parser.add_argument(
        'model_sets',
        help="comma separated list of model sets to use in short name form")
    parser.add_argument('roi', help="region to search within (hip)")
    parser.add_argument('--n-proc', '-n', default=1,
                        help="number of processes for each searchlight")
    parser.add_argument('--category', '-c', default='both',
                        help="categories to include (both, face, scene)")
    args = parser.parse_args()

    model_set_list = args.model_sets.split(',')
    for m_set in model_set_list:
        main(args.step, m_set, args.roi, n_proc=args.n_proc, category=args.category)
