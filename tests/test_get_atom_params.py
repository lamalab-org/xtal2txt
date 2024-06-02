from xtal2txt.core import TextRep
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

srtio3_p1 = TextRep.from_input(os.path.join(THIS_DIR, "data", "SrTiO3_p1.cif"))
tiCrSe_p1 = TextRep.from_input(os.path.join(THIS_DIR, "data", "TlCr5Se8_p1.cif"))


def test_get_atom_params() -> None:
    expected_wo_params = "Sr Ti O O O"
    expected_w_params = "Sr Ti O O O 3.9 3.9 3.9 90 90 90"
    expected_tiCrSe_p1 = "Tl Tl Cr Cr Cr Cr Cr Cr Cr Cr Cr Cr Se Se Se Se Se Se Se Se Se Se Se Se Se Se Se Se"
    assert srtio3_p1.get_atom_sequences_plusplus() == expected_wo_params
    assert tiCrSe_p1.get_atom_sequences_plusplus() == expected_tiCrSe_p1
    assert (
        srtio3_p1.get_atom_sequences_plusplus(lattice_params=True) == expected_w_params
    )
