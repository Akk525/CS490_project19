from typing import Dict, List
from collections import defaultdict
import random


def build_splits_by_source(
    examples: List[Dict],
    train_frac: float = 0.7,
    dev_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    random.seed(seed)

    # ---- Group by source_item_id ----
    groups = defaultdict(list)
    for ex in examples:
        groups[ex["source_item_id"]].append(ex["example_id"])

    group_ids = list(groups.keys())
    random.shuffle(group_ids)

    # ---- Split at group level ----
    n = len(group_ids)
    train_end = int(n * train_frac)
    dev_end = train_end + int(n * dev_frac)

    train_groups = group_ids[:train_end]
    dev_groups = group_ids[train_end:dev_end]
    test_groups = group_ids[dev_end:]

    def collect(group_list):
        out = []
        for gid in group_list:
            out.extend(groups[gid])
        return out

    train_ids = collect(train_groups)
    dev_ids = collect(dev_groups)
    test_ids = collect(test_groups)

    return {
        "train": train_ids,
        "dev": dev_ids,
        "test": test_ids,
        "retrieval_library": train_ids,  # keep this
    }

def check_leakage(examples, splits):
    id_to_source = {ex["example_id"]: ex["source_item_id"] for ex in examples}

    split_sources = {}
    for split, ids in splits.items():
        split_sources[split] = set(id_to_source[i] for i in ids)

    print("Train ∩ Dev:", len(split_sources["train"] & split_sources["dev"]))
    print("Train ∩ Test:", len(split_sources["train"] & split_sources["test"]))
    print("Dev ∩ Test:", len(split_sources["dev"] & split_sources["test"]))