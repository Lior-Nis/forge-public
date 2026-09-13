"""
Build the Salomon(her_sid) -> ours(our_sid) daily-living mapping deterministically.

Salomon shared only gait-filtered predictions (frames where walking OR FOG-label==1)
under her own re-hashed session IDs — zero overlap with our IDs, no subject/day/time
metadata. The only deterministic link is the ABSOLUTE kept-frame index set, which
matches our (Activity==1 | FOG-label) set exactly.

Coverage ceiling is 2126 (what she shared); 1144 fully-sedentary recordings were
filtered out of her export and are unrecoverable by any method.

Two-stage match:
  1. unique index-set hash      -> map directly                         (1848)
  2. collision (>1 our recording with the same kept-index set):
       - if all candidates are byte-identical in (FOG, Activity), the
         assignment is irrelevant to any label-based metric -> recover   (+47)
       - otherwise candidates genuinely differ and her file carries no
         non-circular signal to choose -> leave UNMAPPED (using her own
         scores to pick would bias the comparison) .                    (231 dropped)

Writes research/paper_final/data/kaggle_pred_vectors/_id_map.json
(the previous map is backed up to _id_map.prev.json).
"""
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "eval"))
from _private_inputs import private_path  # author-only inputs; see that module

RAW = os.path.join(os.environ.get("FORGE_DATASETS_ROOT", os.path.expanduser("~/Datasets")), "fogathome_dailyliving/preprocesseddata")
KD = private_path("kaggle_pred_vectors")
MAP_CACHE = KD / "_id_map.json"


def fp(idx):
    return hashlib.md5(np.asarray(sorted(idx), dtype=np.int32).tobytes()).hexdigest()


def kaggle_frames(tag="3rd"):
    d = pd.read_csv(KD / f"pred_{tag}.csv", usecols=["Id"])
    d["sid"] = d.Id.str.rsplit("_", n=1).str[0]
    d["fr"] = d.Id.str.rsplit("_", n=1).str[1].astype(int)
    return {sid: g.fr.values for sid, g in d.groupby("sid")}


def main():
    # ── our recordings: kept-index-set hash -> [sids]; plus (fog,act) content hash ──
    groups, content = {}, {}
    files = [f for f in os.listdir(RAW) if f.endswith(".parquet")]
    for i, fn in enumerate(files):
        sid = fn[:-8]
        d = pd.read_parquet(os.path.join(RAW, fn),
                            columns=["StartHesitation", "Turn", "Walking", "Activity"])
        fog = ((d.StartHesitation.values + d.Turn.values + d.Walking.values) > 0).astype(np.int8)
        act = d.Activity.values.astype(np.int16)
        oi = np.where((act == 1) | (fog == 1))[0]
        if not len(oi):
            continue
        groups.setdefault(fp(oi), []).append(sid)
        content[sid] = hashlib.md5(fog.tobytes() + act.tobytes()).hexdigest()
        if i % 800 == 0:
            print(f"  ..{i}/{len(files)}", flush=True)

    kf = kaggle_frames("3rd")

    # group her_sids by the index-set hash they match
    her_by_hash = {}
    for her, frames in kf.items():
        her_by_hash.setdefault(fp(frames), []).append(her)

    mapping = {}
    n_unique = n_recovered = n_ambiguous = 0
    for h, hers in her_by_hash.items():
        cand = groups.get(h)
        if not cand:
            continue  # not in our cohort (shouldn't happen)
        if len(cand) == 1:
            for her in hers:
                mapping[her] = cand[0]
            n_unique += len(hers)
        elif len({content[s] for s in cand}) == 1:
            # identical (fog,activity) across candidates -> assignment irrelevant.
            # bijective sorted pairing; reuse a candidate only if she shared more
            # duplicates than we hold (still label-identical, so metric-safe).
            cs = sorted(cand)
            for j, her in enumerate(sorted(hers)):
                mapping[her] = cs[min(j, len(cs) - 1)]
            n_recovered += len(hers)
        else:
            n_ambiguous += len(hers)

    if MAP_CACHE.exists():
        shutil.copy(MAP_CACHE, KD / "_id_map.prev.json")
    json.dump(mapping, open(MAP_CACHE, "w"))

    print(f"\nher sessions shared        : {len(kf)}")
    print(f"  unique index-set match   : {n_unique}")
    print(f"  recovered (identical dup): {n_recovered}")
    print(f"  ambiguous (dropped)      : {n_ambiguous}")
    print(f"mapped total -> {len(mapping)}  (was 1848)")
    print(f"wrote {MAP_CACHE}  (backup _id_map.prev.json)")


if __name__ == "__main__":
    main()
