from pathlib import Path
import pytest
from scripts.release.manifest import build_manifest, validate_manifest, RELEASE_SET

REPO = Path(__file__).resolve().parents[2]

def test_release_set_has_exactly_30_artifacts():
    n = len(RELEASE_SET["encoders"]) + len(RELEASE_SET["classification"])
    assert n == 30, f"expected 30 release artifacts, got {n}"

def test_release_set_is_three_encoders_and_27_classification():
    assert len(RELEASE_SET["encoders"]) == 3
    assert len(RELEASE_SET["classification"]) == 27
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
    assert len(headline) == 3  # 3 folds

def test_every_local_path_exists_on_disk():
    m = build_manifest()
    missing = validate_manifest(m, REPO)
    total = len(RELEASE_SET["encoders"]) + len(RELEASE_SET["classification"])
    # Post-onboard release-integrity check. On a bare clone (e.g. CI) the released
    # checkpoints have not been downloaded — skip. Only fail on a PARTIAL/corrupt set
    # (some present, some missing), which is a real integrity problem.
    if len(missing) == total:
        pytest.skip("released checkpoints not downloaded (bare clone / CI); run the onboard-repo steps to fetch them")
    assert missing == [], f"missing checkpoints: {missing}"


def test_public_name_is_the_hf_path_stem():
    """`name` is the published identity and is baked into forge_meta inside each
    .ckpt, so it must never drift from the file it names. Deriving it in
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


def test_manifest_matches_the_eval_scripts_weight_maps():
    """The manifest and eval_comprehensive.py describe one release set. Drift
    between them is how a published file stops matching the model it names."""
    import scripts.eval.eval_comprehensive as ec

    m = build_manifest()
    for c in m["classification"]:
        key = (c["context"], c["phase"])
        assert ec.CKPT[key].format(fold=c["fold"]) == c["local"], c["name"]
        assert ec.EXPERIMENT[key] == c["experiment"], c["name"]
        assert ec.SPLITS[(c["context"], c["fold"])] == c["splits"], c["name"]


def test_every_referenced_config_exists():
    """`experiment` and `splits` are what rebuild a released model, so a typo in
    either makes the weights unloadable."""
    m = build_manifest()
    for entry in list(m["classification"]) + list(m["encoders"].values()):
        assert (REPO / "configs" / "experiment" / f"{entry['experiment']}.yaml").is_file(), \
            f"{entry['name']}: missing experiment config {entry['experiment']}"
        if entry.get("splits"):
            assert (REPO / "configs" / "data" / "splits" / f"{entry['splits']}.yaml").is_file(), \
                f"{entry['name']}: missing splits config {entry['splits']}"


def test_release_is_weights_only():
    """No published path may be a pickled Lightning checkpoint."""
    m = build_manifest()
    for entry in list(m["classification"]) + list(m["encoders"].values()):
        assert entry["hf_path"].endswith(".safetensors"), entry["name"]
        assert not entry["local"].endswith(".ckpt"), entry["name"]


def test_every_result_names_a_documented_cohort_and_model_set():
    """A number must say which cohort and which model set it belongs to. The
    released detector and the controlled comparison report different values on
    the same cohort, so an unlabelled number is unreadable."""
    m = build_manifest()
    for r in m["results"]:
        assert r["cohort"] in m["cohorts"], f"{r['id']}: cohort not in cohorts block"
        assert r["model"] in m["model_sets"], f"{r['id']}: model set not documented"


def test_every_result_cites_where_the_manuscript_reports_it():
    m = build_manifest()
    for r in m["results"]:
        assert r.get("table"), f"{r['id']}: no manuscript table cited"


def test_controlled_comparison_carries_both_arms_and_the_contrast():
    """Table 2 is a contrast, not a single value; dropping an arm would turn the
    headline effect into an unattributed number."""
    m = build_manifest()
    matched = [r for r in m["results"] if r["model"] == "mc_matched_arms"]
    assert len(matched) == 6  # 2 cohorts x {AUROC, AP, ICC(%TF)}
    for r in matched:
        arms = r["arms"]
        assert set(arms) == {"self_supervised", "supervised_from_scratch"}
        assert r["value"] == arms["self_supervised"]
        assert r["difference"] == pytest.approx(
            arms["self_supervised"] - arms["supervised_from_scratch"], abs=1e-4)
        assert len(r["difference_ci"]) == 2 and r["p_value"] <= 0.05


def test_released_and_matched_numbers_are_not_confusable():
    """Same cohort, same metric, different model sets: both must be present and
    distinctly labelled, so no reader takes one for a restatement of the other."""
    m = build_manifest()
    by_id = {r["id"]: r for r in m["results"]}
    released, matched = by_id["fogathome_auroc"], by_id["matched_fogathome_auroc"]
    assert released["cohort"] == matched["cohort"] == "fogathome"
    assert released["model"] != matched["model"]
    assert released["value"] != matched["value"]
    assert released["table"] == "Table 3" and matched["table"] == "Table 2"


def test_results_cover_only_cohorts_the_pipeline_knows_or_are_marked_recorded():
    """A reproducible result must name a cohort the released data actually has."""
    m = build_manifest()
    known = set(m["datasets"])
    for r in m["results"]:
        if r["verification"] in {"one_click", "scripted"}:
            assert r["cohort"] in known, (
                f"{r['id']}: cohort {r['cohort']!r} is not in the datasets block, "
                "so it cannot be reproduced -- mark it `recorded`")
