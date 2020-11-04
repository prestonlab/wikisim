#!/usr/bin/env python
#
# Print commands to create a cluster mask.

import os
import argparse


def main(res_name, cluster_ind, mask_name, radius=1.75, intersect='b_hip_fshs'):
    study_dir = os.environ['STUDYDIR']
    svc_dir = os.path.join(study_dir, 'batch', 'rsa', res_name, 'b_hip_fshs_dil1')
    cluster_mask = os.path.join(svc_dir, 'cluster_mask_cope1.nii.gz')
    subjects = os.environ['WSUBJNOS']
    inputs = f'{subjects} {cluster_mask} {cluster_ind} {mask_name}'
    print(f'wikisim_clust_mask.sh -r {radius} -i {intersect} {inputs}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', '-r', type=str, default=1,
                        help='comma-separated list of radius values in voxels')
    args = parser.parse_args()

    res_names = ['prex_prsa_hmx_use_face_hip_wiki_use1',
                 'prex_prsa_hmx_use_scene_hip_wiki_use1']
    mask_names = ['hmx_use_face_hip',
                  'hmx_use_scene_hip']
    cluster_inds = [1, 1]
    radii = args.radius.split(',')
    for res, cluster, mask in zip(res_names, cluster_inds, mask_names):
        for rad in radii:
            rad = int(rad)
            mask_name = f'{mask}_dil{rad}'
            main(res, cluster, mask_name, radius=round(rad * 1.7 + 0.05, 2))
