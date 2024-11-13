from typing import List
from behaviors import ALL_BEHAVIORS, get_vector_path
from utils.helpers import get_model_path
import argparse
import torch as t
import os


def normalize_vectors(
    model_size: str, is_base: bool, layers: List[int], behaviors: List[str] = ALL_BEHAVIORS
):
    # make normalized_vectors directory
    normalized_vectors_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "normalized_vectors"
    )
    if not os.path.exists(normalized_vectors_dir):
        os.makedirs(normalized_vectors_dir)
    for layer in layers:
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, get_model_path(model_size, is_base=is_base))
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in behaviors:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])


if __name__ == "__main__":
    # Add Argparse for model_size and is_base
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    args = parser.parse_args()

    normalize_vectors(args.model_size, args.use_base_model, args.layers, args.behaviors)

    # normalize_vectors("7b", True, 32)
    # normalize_vectors("7b", False, 32)
    # normalize_vectors("13b", False, 36)
