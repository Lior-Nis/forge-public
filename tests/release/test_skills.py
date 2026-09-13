import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKILLS = ["reproduce-evaluations", "onboard-repo"]

def _read(name):
    return (REPO / ".claude/skills" / name / "SKILL.md").read_text()

def test_skill_files_have_frontmatter():
    for s in SKILLS:
        txt = _read(s)
        assert txt.startswith("---"), f"{s} missing frontmatter"
        assert "name:" in txt and "description:" in txt

def test_skills_reference_the_manifest():
    for s in SKILLS:
        assert "release/manifest.yaml" in _read(s)

def test_referenced_scripts_exist():
    # every scripts/... path mentioned in a skill must exist
    for s in SKILLS:
        for m in re.findall(r"scripts/[A-Za-z0-9_./-]+\.(?:py|sh)", _read(s)):
            assert (REPO / m).exists(), f"{s} references missing {m}"
