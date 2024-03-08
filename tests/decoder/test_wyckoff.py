from xtal2txt.core import TextRep
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_p1.cif"))
Sr = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif"))
In = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "InCuS2_p1.cif"))

N2_str = "N4\n198\nN 4a\n"

Sr_str = "Sr1 Ti1 O3\n221\nSr 1a\nTi 1b\nO 3c\n"

In_str = "In4 Cu4 S8\n122\nIn 4b\nCu 4a\nS 8d\n"


def test_get_wyckoff_positions() -> None:
    expected = "Sr 1a\nTi 1b\nO 3c\n"
    assert Sr.get_wyckoff_positions() == expected


def test_get_wycryst() -> None:
    expected = "Sr1 Ti1 O3\n221\nSr 1a\nTi 1b\nO 3c\n"
    assert Sr.get_wycryst() == expected


def test_wyckoff_matcher() -> None:
    
    assert N2.wyckoff_matcher(N2_str, lattice_params=True) == True
    assert Sr.wyckoff_matcher(Sr_str, lattice_params=True) == True
    assert In.wyckoff_matcher(In_str, lattice_params=True) == True
