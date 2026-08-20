"""Official τ²-bench retail split, smoke-sliced for GEPA episodes.

Full split in sierra-research/tau2-bench is 74 train / 40 test.
Each rollout is a multi-turn user-simulator episode, so the GEPA fixture
keeps the first 20 train ids and first 16 test ids.
"""

from __future__ import annotations

OFFICIAL_TRAIN_IDS = [
    "0", "1", "2", "3", "4", "6", "7", "8", "10", "11",
    "13", "14", "15", "16", "19", "20", "21", "22", "23", "24",
    "25", "28", "29", "30", "31", "34", "35", "37", "41", "43",
    "44", "46", "47", "48", "50", "52", "54", "57", "58", "59",
    "63", "66", "67", "69", "72", "73", "75", "76", "78", "80",
    "81", "82", "83", "84", "85", "87", "88", "89", "91", "92",
    "93", "95", "96", "98", "99", "103", "104", "105", "106", "107",
    "109", "110", "112", "113",
]
OFFICIAL_TEST_IDS = [
    "5", "9", "12", "17", "18", "26", "27", "32", "33", "36",
    "38", "39", "40", "42", "45", "49", "51", "53", "55", "56",
    "60", "61", "62", "64", "65", "68", "70", "71", "74", "77",
    "79", "86", "90", "94", "97", "100", "101", "102", "108", "111",
]

TRAIN_IDS = OFFICIAL_TRAIN_IDS[:20]
HELDOUT_IDS = OFFICIAL_TEST_IDS[:16]


def rows_for(split: str) -> list[dict[str, str | int]]:
    ids = TRAIN_IDS if split == "train" else HELDOUT_IDS
    prefix = "train" if split == "train" else "heldout"
    return [
        {"task_id": f"{prefix}:{task_id}", "example_id": f"{prefix}:{task_id}", "split": split, "seed": int(task_id)}
        for task_id in ids
    ]


def tau2_task_id(task_id: str, split: str, seed: int) -> str:
    _, _, rest = str(task_id).partition(":")
    if rest:
        return rest
    ids = TRAIN_IDS if split == "train" else HELDOUT_IDS
    return ids[seed % len(ids)]
