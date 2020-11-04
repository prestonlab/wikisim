#!/usr/bin/env
#
# Write colormaps for use with fsleyes.

import os
import numpy as np
from matplotlib import cm


def main():
    eyes_dir = os.path.join(os.environ['HOME'], '.fsleyes')
    map_names = ['Reds', 'Blues', 'Greens']
    xs = np.linspace(.1, .5, 1024)
    for map_name in map_names:
        cmap = cm.get_cmap(map_name + '_r', 1024)
        sub_cmap = cmap(xs)[:, :3]
        np.savetxt(os.path.join(eyes_dir, f'{map_name}_r.cmap'), sub_cmap)


if __name__ == '__main__':
    main()
