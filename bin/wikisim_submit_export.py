#!/usr/bin/env python
#
# Submit ROI data export.

import os
import argparse
from wikisim import rsa


def main(roi_group):
    rois = rsa.get_rois(roi_group)
    subjects = os.environ['WSUBJIDS']
    for roi in rois.index.tolist():
        for subject in subjects.split(':'):
            print(f'wikisim_export_prex.py {subject} {roi} -b _stim2')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('roi_group',
                        help="Comma-separated list of ROI groups or ROIs")
    args = parser.parse_args()

    for group in args.roi_group.split(','):
        main(group)
