"""Deterministic nearest-neighbor cache from the EXISTING frozen text vectors."""
import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data/processed/beauty_5core"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_neighbors(embeddings_path, num_items, neighbors=100, chunk_size=256):
    if min(neighbors, chunk_size) < 1:
        raise ValueError("neighbors and chunk_size must be positive.")
    with np.load(embeddings_path, allow_pickle=False) as source:
        vectors = source["embeddings"].copy()
        has_text = source["has_text"].copy()
    if vectors.shape != (num_items+1,384) or has_text.shape != (num_items+1,):
        raise ValueError("Text vectors do not match the shared catalog.")
    if has_text[0] or not np.isfinite(vectors).all():
        raise ValueError("Invalid padding flag or nonfinite text vector.")
    valid = np.flatnonzero(has_text)
    if len(valid) < 2:
        raise ValueError("Semantic sampling requires at least two text-bearing items.")
    norms = np.linalg.norm(vectors[valid], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Expected the existing normalized text vectors.")
    k = min(neighbors, len(valid)-1)
    table = np.zeros((num_items+1,k), dtype=np.int32)
    matrix = torch.from_numpy(vectors)
    ids = torch.from_numpy(valid)
    for offset in range(0, len(valid), chunk_size):
        anchors = ids[offset:offset+chunk_size]
        similarities = matrix[anchors] @ matrix[ids].T
        similarities[anchors[:,None].eq(ids[None,:])] = -torch.inf
        # Candidate IDs are ascending: stable sorting breaks ties by smaller ID.
        order = torch.argsort(similarities,dim=1,descending=True,stable=True)[:,:k]
        table[anchors.numpy()] = ids[order].numpy()
    return table, has_text


def load_or_build(embeddings_path, cache_dir, num_items, neighbors=100):
    source_sha = file_sha256(embeddings_path)
    path = Path(cache_dir)/f"neighbors_{source_sha[:16]}_k{neighbors}.npz"
    start = perf_counter()
    if path.exists():
        with np.load(path,allow_pickle=False) as cache:
            if str(cache["source_sha256"]) != source_sha or int(cache["num_items"]) != num_items:
                raise ValueError("Neighbor cache provenance mismatch.")
            table, has_text = cache["neighbors"].copy(), cache["has_text"].copy()
        if table.shape[0] != num_items+1 or has_text.shape != (num_items+1,):
            raise ValueError("Neighbor cache shape mismatch.")
        if table.min() < 0 or table.max() > num_items:
            raise ValueError("Neighbor cache has invalid item IDs.")
        cached = True
    else:
        table, has_text = build_neighbors(embeddings_path,num_items,neighbors)
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,neighbors=table,has_text=has_text,
                            source_sha256=source_sha,num_items=num_items)
        cached = False
    metadata = {"embeddings_sha256":source_sha,"cache_path":str(path),
                "cache_sha256":file_sha256(path),"requested_neighbors":neighbors,
                "actual_neighbors":table.shape[1],"cache_hit":cached,
                "preprocessing_seconds":perf_counter()-start}
    return table,has_text,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings",type=Path,default=SPLIT_DIR/"item_text_embeddings.npz")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"artifacts/adaptive_sampling")
    parser.add_argument("--num-items",type=int,default=12101)
    parser.add_argument("--neighbors",type=int,default=100)
    args=parser.parse_args()
    torch.set_num_threads(4)
    import json
    _,_,metadata=load_or_build(args.embeddings,args.cache_dir,args.num_items,args.neighbors)
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
