"""Parse taxonomy labels and build the spectrum manifest."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True)
class LabelRecord:
    spectrum_path: str
    asteroid_id: str
    source: str
    class_raw: str
    class_bd: str
    dynamical_group: str
    has_spectrum: bool


def asteroid_key_from_path(path: Path) -> str:
    """Map spectrum filename to asteroid identifier (e.g. a000004 -> 4)."""
    stem = path.stem.split(".")[0]
    base = stem
    if base.startswith("a"):
        base = base[1:]
    if base.startswith("u"):
        base = base[1:]
    if base.isdigit():
        return base.lstrip("0") or "0"
    return base


def normalize_asteroid_id(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return raw.lstrip("0") or "0"
    return raw


def apply_class_alias(class_raw: str, aliases: Dict[str, str], bd_classes: Set[str]) -> Tuple[str, bool]:
    """Return (class_bd, used_alias)."""
    token = class_raw.strip()
    if token in aliases:
        return aliases[token], True
    if token in bd_classes:
        return token, False
    # Try longest-prefix match for compound tokens already in BD list
    for bd in sorted(bd_classes, key=len, reverse=True):
        if token == bd:
            return bd, False
    raise ValueError(f"Unmapped class token: {token!r}")


def parse_demeotax(path: Path) -> Dict[str, Tuple[str, str]]:
    """
    Parse demeotax.tab -> asteroid_id -> (class_raw, flag).
    """
    id_to_class: Dict[str, Tuple[str, str]] = {}
    for line in path.read_text().splitlines():
        line = line.rstrip()
        if not line.strip():
            continue
        m = DATE_RE.search(line)
        if not m:
            continue
        pre = line[: m.start()].strip()
        post = line[m.end() :].strip()
        flag = post.split()[0] if post.split() else ""
        tokens = pre.split()
        if not tokens:
            continue
        ast_id = normalize_asteroid_id(tokens[0])
        cls = None
        for tok in reversed(tokens):
            if tok in ("-",):
                continue
            if re.fullmatch(r"\d{4}", tok):
                continue
            if tok.isdigit() and len(tok) >= 4 and tok != ast_id:
                continue
            cls = tok
            break
        if cls is None:
            continue
        id_to_class[ast_id] = (cls, flag)
    return id_to_class


def parse_binzel_classes(path: Path) -> Dict[str, Tuple[str, List[str]]]:
    """
    Parse Binzel_classes.txt -> asteroid_id -> (primary_class, all_classes).
    """
    out: Dict[str, Tuple[str, List[str]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[0][0].isdigit():
            continue
        ast_id = normalize_asteroid_id(parts[0])
        raw = parts[1]
        classes = [c.strip() for c in raw.split(",") if c.strip()]
        if not classes:
            continue
        out[ast_id] = (classes[0], classes)
    return out


def load_mars_crosser_ids(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    ids: Set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            ids.add(normalize_asteroid_id(line))
    return ids


def infer_dynamical_group(
    source: str,
    asteroid_id: str,
    demeo_ids: Set[str],
    mars_crosser_ids: Set[str],
) -> str:
    if source == "demeo":
        return "main_belt"
    if asteroid_id in mars_crosser_ids:
        return "mars_crosser"
    # Binzel-only objects (not in DeMeo taxonomy table) treated as Mars-crosser
    # candidates for Run 2 test set (paper ~202 Mars-crossers).
    if asteroid_id not in demeo_ids:
        return "mars_crosser"
    return "nea"


def build_manifest(
    data_root: Path,
    demeo_gp_dir: Path,
    binzel_gp_dir: Path,
    demeotax_path: Path,
    binzel_classes_path: Path,
    aliases: Dict[str, str],
    bd_classes: List[str],
    excluded_classes: Iterable[str],
    mars_crossers_path: Optional[Path] = None,
    quality_filter: Optional[Callable[[Path], bool]] = None,
) -> Tuple[List[LabelRecord], Dict[str, int], List[str]]:
    """
    Build manifest rows and return (records, stats, alias_log_lines).

    If ``quality_filter`` is provided it is called with each absolute spectrum
    path and the row is dropped when the callable returns False.
    """
    bd_set = set(bd_classes)
    excluded = {c.upper() for c in excluded_classes}
    demeo_tax = parse_demeotax(demeotax_path)
    binzel_tax = parse_binzel_classes(binzel_classes_path)
    demeo_ids = set(demeo_tax.keys())
    mars_ids = load_mars_crosser_ids(mars_crossers_path) if mars_crossers_path else set()

    alias_log: List[str] = []
    records: List[LabelRecord] = []
    unmapped: List[str] = []
    skipped_no_label: List[str] = []
    dropped_low_quality: List[str] = []

    def add_row(path: Path, source: str, class_raw: str, ast_id: str) -> None:
        nonlocal records, alias_log, unmapped
        classes_multi = [c.strip() for c in class_raw.split(",") if c.strip()]
        primary = classes_multi[0] if classes_multi else class_raw
        if primary.upper() in excluded:
            return
        try:
            class_bd, used_alias = apply_class_alias(primary, aliases, bd_set)
        except ValueError:
            unmapped.append(f"{ast_id}:{primary}")
            return
        if used_alias:
            alias_log.append(f"{ast_id}: {primary} -> {class_bd}")
        if quality_filter is not None and not quality_filter(path):
            dropped_low_quality.append(f"{source}:{ast_id}:{path.name}")
            return
        dyn = infer_dynamical_group(source, ast_id, demeo_ids, mars_ids)
        records.append(
            LabelRecord(
                spectrum_path=str(path.relative_to(data_root)),
                asteroid_id=ast_id,
                source=source,
                class_raw=primary,
                class_bd=class_bd,
                dynamical_group=dyn,
                has_spectrum=path.is_file(),
            )
        )

    for gp_dir, source in (
        (demeo_gp_dir, "demeo"),
        (binzel_gp_dir, "binzel"),
    ):
        if not gp_dir.is_dir():
            continue
        for spec_path in sorted(gp_dir.glob("a*.txt")):
            ast_id = asteroid_key_from_path(spec_path)
            class_raw = None
            if ast_id in demeo_tax:
                class_raw, _flag = demeo_tax[ast_id]
            elif ast_id in binzel_tax:
                class_raw, _all_cls = binzel_tax[ast_id]
            else:
                skipped_no_label.append(f"{source}:{ast_id}")
                continue
            add_row(spec_path, source, class_raw, ast_id)

    if unmapped:
        raise RuntimeError(
            "Unmapped class tokens:\n  " + "\n  ".join(unmapped[:30])
            + (f"\n  ... and {len(unmapped) - 30} more" if len(unmapped) > 30 else "")
        )

    stats = {
        "total_rows": len(records),
        "demeo_rows": sum(1 for r in records if r.source == "demeo"),
        "binzel_rows": sum(1 for r in records if r.source == "binzel"),
        "unique_asteroids": len({r.asteroid_id for r in records}),
        "alias_applications": len(alias_log),
        "multi_class_binzel": sum(
            1
            for _id, (_p, all_c) in binzel_tax.items()
            if len(all_c) > 1
        ),
        "skipped_no_label": len(skipped_no_label),
        "dropped_low_quality": len(dropped_low_quality),
    }
    if dropped_low_quality:
        stats["dropped_low_quality_examples"] = dropped_low_quality[:10]
    return records, stats, alias_log


def write_manifest_csv(path: Path, records: List[LabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "spectrum_path",
                "asteroid_id",
                "source",
                "class_raw",
                "class_bd",
                "dynamical_group",
                "has_spectrum",
            ]
        )
        for r in records:
            w.writerow(
                [
                    r.spectrum_path,
                    r.asteroid_id,
                    r.source,
                    r.class_raw,
                    r.class_bd,
                    r.dynamical_group,
                    r.has_spectrum,
                ]
            )


def read_manifest_csv(path: Path) -> List[LabelRecord]:
    records: List[LabelRecord] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                LabelRecord(
                    spectrum_path=row["spectrum_path"],
                    asteroid_id=row["asteroid_id"],
                    source=row["source"],
                    class_raw=row["class_raw"],
                    class_bd=row["class_bd"],
                    dynamical_group=row["dynamical_group"],
                    has_spectrum=row["has_spectrum"].lower() in ("true", "1", "yes"),
                )
            )
    return records


def class_to_index_from_manifest(records: List[LabelRecord], bd_classes: List[str]) -> Dict[str, int]:
    present = {r.class_bd for r in records}
    ordered = [c for c in bd_classes if c in present]
    extra = sorted(present - set(ordered))
    ordered.extend(extra)
    return {c: i for i, c in enumerate(ordered)}


def coarse_class_for(fine_class: str, coarse_groups: Dict[str, List[str]]) -> str:
    """Return the coarse-group key that contains ``fine_class``.

    Falls back to the first-letter convention if the fine class is not
    explicitly listed under any group (so the function is total).
    """
    for coarse, children in coarse_groups.items():
        if fine_class in children:
            return coarse
    return fine_class[:1].upper() if fine_class else ""


def coarse_class_to_index(
    fine_class_to_index: Dict[str, int],
    coarse_groups: Dict[str, List[str]],
) -> Tuple[Dict[str, int], Dict[int, int], Dict[str, List[int]]]:
    """Build (coarse_to_index, fine_to_coarse_index, coarse_to_fine_indices).

    Coarse keys are sorted alphabetically for stable ordering.  Only coarse
    groups that contain at least one fine class present in
    ``fine_class_to_index`` are kept.
    """
    coarse_present: List[str] = []
    coarse_to_children: Dict[str, List[str]] = {}
    for coarse, children in coarse_groups.items():
        present = [c for c in children if c in fine_class_to_index]
        if not present:
            continue
        coarse_present.append(coarse)
        coarse_to_children[coarse] = present
    coarse_present.sort()

    coarse_to_index = {c: i for i, c in enumerate(coarse_present)}
    fine_to_coarse: Dict[int, int] = {}
    coarse_to_fine_indices: Dict[str, List[int]] = {c: [] for c in coarse_present}
    for fine, idx in fine_class_to_index.items():
        coarse = coarse_class_for(fine, coarse_groups)
        if coarse not in coarse_to_index:
            coarse_to_index[coarse] = len(coarse_to_index)
            coarse_to_fine_indices.setdefault(coarse, [])
        fine_to_coarse[idx] = coarse_to_index[coarse]
        coarse_to_fine_indices[coarse].append(idx)
    return coarse_to_index, fine_to_coarse, coarse_to_fine_indices
