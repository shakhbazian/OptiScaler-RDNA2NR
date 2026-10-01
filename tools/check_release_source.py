"""Check the source checkout before publishing; no model or GPU is required."""

from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"!?\[[^\]\n]*\]\(([^\s)]+)\)")


def heading_ids(path):
    # These public guides use ordinary headings. Track duplicate slugs as GitHub does.
    counts = {}
    result = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", path.read_text(encoding="utf-8-sig"), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        number = counts.get(slug, 0)
        counts[slug] = number + 1
        result.add(f"{slug}-{number}" if number else slug)
    return result


def main():
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8").split("\0")
    errors = []
    forbidden_names = {"nvngx_dlssnr.dll", "amdhip64_6.dll", "amdhip64_7.dll", "amd_comgr_2.dll", "amd_comgr_3.dll"}
    for name in filter(None, tracked):
        path = Path(name)
        if path.suffix.lower() == ".nrwgt" or path.name.lower() in forbidden_names:
            errors.append(f"Private model or driver asset is tracked: {name}")
        if path.parts[0].lower() in {"build", "release", "local_assets"}:
            errors.append(f"Local artifact directory is tracked: {name}")

    guides = [ROOT / "README.md", ROOT / "INSTALL-DLSSNR.md", ROOT / "Changelog.md"]
    guides += sorted((ROOT / "docs").glob("*.md"))
    guides += sorted((ROOT / "docs/release-notes").glob("*.md"))
    guides += [ROOT / "docs/upstream/README.md"]
    links = 0
    for guide in guides:
        if not guide.is_file():
            errors.append(f"Public guide is missing: {guide.relative_to(ROOT)}")
            continue
        for reference in LINK.findall(guide.read_text(encoding="utf-8-sig")):
            if urlsplit(reference).scheme:
                continue
            filename, _, anchor = reference.partition("#")
            target = (guide.parent / unquote(filename)).resolve() if filename else guide
            links += 1
            if not target.is_relative_to(ROOT):
                errors.append(f"Link escapes checkout: {guide.relative_to(ROOT)} -> {reference}")
            elif not target.exists():
                errors.append(f"Missing link target: {guide.relative_to(ROOT)} -> {reference}")
            elif anchor and target.is_file() and target.suffix == ".md" and unquote(anchor) not in heading_ids(target):
                errors.append(f"Missing heading: {guide.relative_to(ROOT)} -> {reference}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(guides)} public guides, {links} local links; private model/driver assets excluded.")


if __name__ == "__main__":
    main()
