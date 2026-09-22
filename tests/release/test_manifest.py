from pathlib import Path

import pytest

from scripts.release.manifest import RELEASE_SET, build_manifest

REPO = Path(__file__).resolve().parents[2]

def test_release_set_has_exactly_36_artifacts():
    n = len(RELEASE_SET["encoders"]) + len(RELEASE_SET["classification"])
    assert n == 36, f"expected 36 release artifacts, got {n}"

def test_release_set_is_three_encoders_and_33_classification():
    assert len(RELEASE_SET["encoders"]) == 3
    assert len(RELEASE_SET["classification"]) == 33
    phases = {c["phase"] for c in RELEASE_SET["classification"]}
    assert phases == {"probe", "finetune", "supervised"}
    contexts = {c["context"] for c in RELEASE_SET["classification"]}
    assert contexts == {"lc", "mc", "sc"}
    folds = {c["fold"] for c in RELEASE_SET["classification"]}
    assert folds == {0, 1, 2}

def test_manifest_marks_mc_probe_as_headline():
    m = build_manifest()
    headline = [c for c in m["classification"] if c.get("headline")]
    assert all(c["context"] == "mc" and c["phase"] == "probe" for c in headline)
    assert len(headline) == 9

def test_local_release_paths_are_a_supported_download_set():
    m = build_manifest()
    all_paths = {
        entry["local"]
        for entry in [*m["encoders"].values(), *m["classification"]]
    }
    headline_paths = {
        m["encoders"]["mc"]["local"],
        *(entry["local"] for entry in m["classification"] if entry.get("headline")),
    }
    present = {path for path in all_paths if (REPO / path).is_file()}

    if not present:
        pytest.skip("released checkpoints not downloaded (bare clone / CI); run the onboard-repo steps to fetch them")
    assert present in (headline_paths, all_paths), (
        "local checkpoints are neither the documented 10-file reproduction subset "
        f"nor the complete release: {sorted(present)}"
    )


def test_public_name_is_the_hf_path_stem():
    """`name` is the published identity and is baked into forge_meta inside each
    safetensors file, so it must never drift from the file it names. Deriving it in
    manifest.py makes this structural; this pins it."""
    m = build_manifest()
    for c in m["classification"]:
        assert c["name"] == Path(c["hf_path"]).stem, (
            f"{c['name']!r} != stem of {c['hf_path']!r}")


def test_public_names_carry_no_internal_cohort_tag():
    """The released set is one cohort, so the historical `all128` tag belongs to
    local training dirs only — never to a published name or hf_path."""
    m = build_manifest()
    for c in m["classification"]:
        assert "all128" not in c["name"], f"internal tag leaked into name: {c['name']}"
        assert "all128" not in c["hf_path"], f"internal tag leaked into hf_path: {c['hf_path']}"
    assert "all128" not in m["meta"]["headline_model"]


def test_manifest_reproduction_commands_use_the_public_model_names():
    """`results[].script` must be runnable as-written against the current CLI."""
    m = build_manifest()
    for r in m.get("results", []):
        script = r.get("script")
        if script:
            assert "all128" not in script, f"stale model name in {r['id']}: {script}"


def _reproduce_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_rp", REPO / "reproduce.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_result_declares_a_verification_status():
    """A reported number must say whether this repo can recompute it. Silence is
    what let 4-of-13 print as unqualified success."""
    m = build_manifest()
    for r in m["results"]:
        assert r.get("verification") in {"one_click", "scripted", "recorded"}, \
            f"{r['id']}: missing/invalid verification status"


def test_recorded_results_explain_why_and_scripted_ones_give_a_command():
    m = build_manifest()
    for r in m["results"]:
        if r["verification"] == "recorded":
            assert r.get("verification_note"), f"{r['id']}: recorded but no reason given"
            assert not r.get("script"), f"{r['id']}: recorded but claims a script"
        if r["verification"] == "scripted":
            assert r.get("script"), f"{r['id']}: scripted but no command"


def test_one_click_results_match_what_reproduce_actually_checks():
    """The manifest's `one_click` set and reproduce.py's RESULT_CHECKS are two
    halves of one contract; drift between them re-opens the silent-pass gap."""
    m = build_manifest()
    declared = {r["id"] for r in m["results"] if r["verification"] == "one_click"}
    implemented = set(_reproduce_module().RESULT_CHECKS)
    assert declared == implemented, (
        f"declared-but-unchecked: {sorted(declared - implemented)}; "
        f"checked-but-undeclared: {sorted(implemented - declared)}")


def test_results_cover_only_cohorts_the_pipeline_knows_or_are_marked_recorded():
    """A reproducible result must name a cohort the released data actually has."""
    m = build_manifest()
    known = set(m["datasets"])
    for r in m["results"]:
        if r["verification"] in {"one_click", "scripted"}:
            assert r["cohort"] in known, (
                f"{r['id']}: cohort {r['cohort']!r} is not in the datasets block, "
                "so it cannot be reproduced -- mark it `recorded`")
