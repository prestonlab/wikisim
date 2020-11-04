#!/usr/bin/env python
#
# Run Universal Sentence Encoder on Wikipedia text for each item.

import argparse
import numpy as np
import scipy.io as sio
import scipy.spatial.distance as sd
import tensorflow_hub as hub
from wikisim import task


def run_use(pool_file, text_dir, model, model_file):
    items = task.read_items(pool_file)
    text = task.read_text(items, text_dir)
    if model == 'normal':
        hub_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    elif model == 'large':
        hub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    else:
        raise ValueError(f'Unknown model: {model}')
    embed = hub.load(hub_url)
    v_list = []
    for article in text:
        embedding = embed([article])
        vector = embedding.numpy()
        v_list.append(vector)
    vectors = np.vstack(v_list)
    rdm = sd.squareform(sd.pdist(vectors, 'correlation'))
    sio.savemat(model_file, {'vectors': vectors, 'items': items, 'rdm': rdm})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Universal Sentence Encoder.")
    parser.add_argument('pool')
    parser.add_argument('textdir')
    parser.add_argument('model')
    parser.add_argument('outfile')
    args = parser.parse_args()

    run_use(args.pool, args.textdir, args.model, args.outfile)
