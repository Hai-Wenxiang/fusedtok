#!/usr/bin/env python3
"""Repository-wide text hygiene gate.

The PyPI 0.1.1/0.1.2 pages shipped mojibake twice (UTF-8 bytes re-read
through a legacy code page while editing). This script exists so that can
never happen again: CI runs it on every push and fails the build on

1. files that are not strict UTF-8 (decoded with errors="strict"),
2. a UTF-8 BOM (0xEF 0xBB 0xBF) in any tracked text file,
3. the U+FFFD replacement character (evidence of a lossy round trip),
4. double-encoded UTF-8 - the classic mojibake shape where every original
   UTF-8 byte was re-encoded as a Latin-1/CP1252 character, producing
   runs of U+00C0..U+00FF followed by U+0080..U+00BF (e.g. the Chinese
   word for "middle" appearing as six Latin characters starting with
   an a-grave).

Scanned set: every git-tracked file whose extension looks like text
(binaries such as .png are skipped). Exit code 0 = clean, 1 = violations
printed with file/line context.
"""

import re
import subprocess
import sys

TEXT_EXTENSIONS = {
    ".py", ".pyi", ".md", ".txt", ".rst", ".yml", ".yaml", ".toml",
    ".cfg", ".ini", ".json", ".sh", ".bash", ".cu", ".cuh", ".cpp",
    ".cc", ".cxx", ".h", ".hpp", ".hh", ".cmake", ".in", ".ninja",
    ".gitignore", ".gitattributes", ".bzl",
}
TEXT_BASENAMES = {
    "CMakeLists.txt", "LICENSE", "NOTICES.md", "CHANGELOG.md",
    "CODE_OF_CONDUCT.md", "SECURITY.md", "CONTRIBUTING.md", "README.md",
    "README_zh.md", ".gitignore", ".gitattributes",
}
# One lead byte in the Latin-1 supplement followed by one (or more) bytes in
# the continuation range: exactly what UTF-8-as-CP1252 double encoding does.
MOJIBAKE_RUN = re.compile(r"[\u00c0-\u00ff][\u0080-\u00bf]{1,}")
MAX_EXAMPLES = 5


def tracked_files():
    """All non-deleted paths known to git (empty tree safe)."""
    out = subprocess.run(
        ["git", "ls-files", "-z"], capture_output=True, check=True)
    return [p for p in out.stdout.decode("utf-8", "replace").split("\0") if p]


def is_text(path):
    stem = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if stem in TEXT_BASENAMES:
        return True
    dot = stem.rfind(".")
    return dot >= 0 and stem[dot:].lower() in TEXT_EXTENSIONS


def check_file(path):
    """Return a list of human-readable violations for one file."""
    problems = []
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        problems.append("UTF-8 BOM present (strip it)")
        raw = raw[3:]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        problems.append(f"not valid UTF-8: {exc}")
        return problems
    for lineno, line in enumerate(text.splitlines(), 1):
        if "\ufffd" in line:
            problems.append(f"line {lineno}: U+FFFD replacement character")
        hit = MOJIBAKE_RUN.search(line)
        if hit:
            problems.append(
                f"line {lineno}: double-encoded UTF-8 (mojibake) shape "
                f"{hit.group(0)[:16]!r}")
    return problems


def main():
    # Windows consoles default to a legacy code page (GBK and friends);
    # violating text may not survive printing there. Force UTF-8 with
    # replacement so the report itself can never crash the gate.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = 0
    for path in tracked_files():
        if not is_text(path):
            continue
        try:
            problems = check_file(path)
        except OSError as exc:
            problems = [f"unreadable: {exc}"]
        if problems:
            failures += 1
            print(f"FAIL {path}")
            for problem in problems[:MAX_EXAMPLES]:
                print(f"     {problem}")
            extra = len(problems) - MAX_EXAMPLES
            if extra > 0:
                print(f"     ... and {extra} more")
    if failures:
        print(f"\n{failures} file(s) failed the text hygiene gate")
        return 1
    print("text hygiene gate: all tracked text files are clean UTF-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
