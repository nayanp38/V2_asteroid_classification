import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.labels import (
    asteroid_key_from_path,
    normalize_asteroid_id,
    normalize_marsset_taxon,
    parse_demeotax,
    parse_marsset_classes,
    apply_class_alias,
)


def test_asteroid_key_from_path():
    assert asteroid_key_from_path(Path("a000004.sp02.txt")) == "4"
    assert asteroid_key_from_path(Path("au2000PG3.sp01.txt")) == "2000PG3"


def test_normalize_id():
    assert normalize_asteroid_id("0004") == "4"


def test_parse_demeotax_count():
    path = ROOT / "data" / "demeotax.tab"
    tax = parse_demeotax(path)
    assert len(tax) == 371
    assert tax["4"][0] == "V"


def test_parse_marsset_classes_count():
    path = ROOT / "data" / "Marsset2022_classes.txt"
    tax = parse_marsset_classes(path)
    assert len(tax) == 491
    assert tax["a000433.sp223.txt"] == "S"
    assert tax["a002059.sp257.txt"] == "Sq;Q"


def test_normalize_marsset_taxon():
    assert normalize_marsset_taxon("S_comp") == "S_comp"
    assert normalize_marsset_taxon("S;Sr") == "S"
    assert normalize_marsset_taxon("Sq;Q") == "Sq"
    assert normalize_marsset_taxon("Xk:") == "Xk"
    assert normalize_marsset_taxon("CX:") == "CX"
    assert normalize_marsset_taxon("::") == ""


def test_class_aliases():
    bd = {"S", "Sq", "V", "Sv"}
    aliases = {"Sw": "S", "Vw": "V", "Svw": "Sv"}
    assert apply_class_alias("Sw", aliases, bd) == ("S", True)
    assert apply_class_alias("Sq", aliases, bd) == ("Sq", False)
    bd2 = bd | {"C"}
    aliases2 = {**aliases, "S_comp": "S", "CX": "C"}
    assert apply_class_alias("S_comp", aliases2, bd2) == ("S", True)
    assert apply_class_alias("CX", aliases2, bd2) == ("C", True)
