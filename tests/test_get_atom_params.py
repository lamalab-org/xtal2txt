from xtal2txt.core import TextRep
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

srtio3_p1 = TextRep.from_input(os.path.join(THIS_DIR, "data", "SrTiO3_p1.cif"))


def test_get_atom_params() -> None: 
    expected_wo_params = "Sr Ti O O O"
    expected_w_params = "Sr Ti O O O 90 90 90 3.91 3.91 3.91"
    assert srtio3_p1.get_atom_params() == expected_wo_params
    assert srtio3_p1.get_atom_params(lattice_params=True) == expected_w_params