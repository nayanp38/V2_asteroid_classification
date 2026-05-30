import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.labels import (
    asteroid_key_from_path,
    normalize_asteroid_id,
    parse_demeotax,
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


def test_class_aliases():
    bd = {"S", "Sq", "V", "Sv"}
    aliases = {"Sw": "S", "Vw": "V", "Svw": "Sv"}
    assert apply_class_alias("Sw", aliases, bd) == ("S", True)
    assert apply_class_alias("Sq", aliases, bd) == ("Sq", False)
