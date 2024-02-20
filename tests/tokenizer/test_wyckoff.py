from xtal2txt.core import TextRep
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2.cif"))
srtio3_p1 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif"))
incus2_p1 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "InCuS2.cif"))


def test_get_wyckoff_positions() -> None:
    expected = "Sr 1a\nTi 1b\nO 3c\n"
    assert srtio3_p1.get_wyckoff_positions() == expected


def test_get_wycryst() -> None:
    expected = "Sr1 Ti1 O3\n221\nSr 1a\nTi 1b\nO 3c\n"
    assert srtio3_p1.get_wycryst() == expected


def test_wyckoff_matcher() -> None:
    results = []
    for i in range(10):
        if N2.wyckoff_matcher(lattice_params=True):
            results.append(True)
            break
    
    for i in range(10):
        if srtio3_p1.wyckoff_matcher(lattice_params=True):
            results.append(True)
            break
    
    for i in range(10):
        if incus2_p1.wyckoff_matcher(lattice_params=True):
            results.append(True)
            break
    
    assert [True, True, True] == results
