"""Markdown hygiene gate: catches README benchmark-table regressions
before they ship.

The 1.8.1 release shipped two table defects this gate exists to catch:
broken bold markers ('****' from a regex that kept the old wrappers)
and one GPU's numbers written into the other GPU's table. Both are
mechanical, so the check is mechanical:

1. No '****' anywhere; no table cell with an odd number of '**'.
2. Every benchmark data row (fused cell = "N µs") must carry a
   (fused, torch) pair that exists in the JSON of the table's OWN GPU
   (3060 above the Blackwell heading, 5060 Ti below) - numbers copied
   from the wrong GPU cannot match. Values compare as their formatted
   strings (the tables render f"{v:.0f} µs"), so the x.5 rounding
   boundary is handled exactly. Torch range cells ("a-b µs") are
   checked as ranges.
3. The row's speedup token must equal that JSON row's rounded speedup
   (bold or plain - no bare '**' fragments).

Exits non-zero, printing every offending row.
"""
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
READMES = ['README.md', 'README_zh.md']
BM = ROOT / 'docs' / 'benchmarks'

errors = []
checked = 0
checked_per_readme = {}

j3060 = json.loads((BM / 'benchmark_rtx3060.json').read_text(encoding='utf-8'))['results']
j5060 = json.loads((BM / 'benchmark_rtx5060ti.json').read_text(encoding='utf-8'))['results']
by_gpu = {'3060': j3060, '5060': j5060}


def fmt_us(v):
    return f"{v:.0f} µs"


def parse_pair(cell):
    """'123 µs ...' -> 123.0 | '~12-34 µs ...' -> (12.0, 34.0) | else None.

    Prefix match: benchmark torch cells carry annotations after the
    number ('7913 µs (SDPA)'), ranges carry both endpoints."""
    m = re.match(r'~?([\d.]+) µs', cell)
    if m:
        return float(m.group(1))
    m = re.match(r'~?([\d.]+)-([\d.]+) µs', cell)
    if m:
        return (float(m.group(1)), float(m.group(2)))
    return None


for path in READMES:
    p = ROOT / path
    if not p.exists():
        continue
    text = p.read_text(encoding='utf-8')
    if '****' in text:
        for i, ln in enumerate(text.split('\n'), 1):
            if '****' in ln:
                errors.append(f'{path}:{i}: broken bold "****": {ln[:110]}')
    for i, ln in enumerate(text.split('\n'), 1):
        if ln.startswith('|') and ln.count('**') % 2 != 0:
            errors.append(f'{path}:{i}: unbalanced ** in table row: {ln[:110]}')
    cut = text.find('**RTX 5060 Ti')
    if cut < 0:
        cut = len(text)
    for span, gpu in (((0, cut), '3060'), ((cut, len(text)), '5060')):
        for ln in text[span[0]:span[1]].split('\n'):
            if not ln.startswith('|'):
                continue
            cells = ln.split('|')
            if len(cells) < 6:
                continue
            fused_cell = cells[3].strip()
            fused = parse_pair(fused_cell)
            label = cells[1].strip()[:44]
            torch_cell = cells[4].strip()
            torch = parse_pair(torch_cell)
            if fused is None and torch is None:
                continue          # header / non-data row
            # asymmetric parse = a broken cell, never a silent skip
            # (a table-generator format change must FAIL, not pass
            # with a shrunken scan)
            if fused is None or torch is None:
                errors.append(
                    f'{path}: "{label}" data row has exactly one '
                    f'parseable µs cell (fused={fused_cell[:24]!r}, '
                    f'torch={torch_cell[:24]!r})')
                continue
            checked += 1
            checked_per_readme[path] = checked_per_readme.get(path, 0) + 1

            def row_ok(j):
                # a leading '~' marks an approximate/aggregated cell
                stripped = fused_cell.lstrip('~')
                if isinstance(fused, tuple):
                    lo, hi = fused
                    if not (lo - 0.51 <= j['fusedtok_us'] <= hi + 0.51):
                        return False
                elif not stripped.startswith(fmt_us(j['fusedtok_us'])):
                    return False
                if isinstance(torch, tuple):
                    lo, hi = torch
                    return lo - 0.51 <= j['torch_us'] <= hi + 0.51
                return torch_cell.lstrip('~').startswith(fmt_us(j['torch_us']))

            cands = [j for j in by_gpu[gpu] if row_ok(j)]
            if not cands:
                errors.append(
                    f'{path}: "{label}" [{cells[2].strip()}] carries '
                    f'({fused_cell} / {torch_cell}) which matches no '
                    f'{gpu} JSON row (wrong GPU or stale number?)')
                continue
            sp_cell = cells[5].strip()
            if isinstance(fused, tuple):
                # aggregate range row: the speedup is a "~N.Nx" label,
                # no single JSON speedup to compare against
                if not re.match(r'^~\d\.\dx(\*\*)?$', sp_cell.replace('**', '')) \
                        and not re.match(r'^~\d\.\dx$', sp_cell):
                    errors.append(
                        f'{path}: "{label}" range-row speedup cell looks '
                        f'wrong: {sp_cell[:40]}')
            else:
                if re.match(r'^~\d\.\dx$', sp_cell):
                    pass             # approximate aggregate label (~1.0x)
                else:
                    want = f"{cands[0]['speedup']:.2f}x"
                    if not (sp_cell.startswith(f'**{want}**')
                            or sp_cell.startswith(f'{want}')):
                        errors.append(
                            f'{path}: "{label}" speedup cell '
                            f'"{sp_cell[:40]}" does not carry the JSON '
                            f'speedup {want}')
            if sp_cell.count('**') not in (0, 2):
                errors.append(
                    f'{path}: "{label}" speedup cell has broken bold: '
                    f'{sp_cell[:50]}')

print(f'markdown hygiene: {checked} benchmark rows scanned over {len(READMES)} files')
# coverage floor: the shipped tables hold ~60 data rows per README; a
# scan that suddenly sees far fewer means the row format drifted and
# the value checks silently stopped running
for path, n in checked_per_readme.items():
    if n < 55:
        errors.append(
            f'{path}: only {n} benchmark rows parsed (expected >= 55) '
            f'- table format drift?')
if errors:
    print(f'MARKDOWN HYGIENE FAIL ({len(errors)}):')
    for e in errors[:30]:
        print(' ', e)
    sys.exit(1)
print('markdown hygiene: CLEAN')
