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
    expected = "Sr 1 a\nTi 1 b\nO 3 c\n"
    assert Sr.get_wyckoff_positions() == expected


def test_get_wycryst() -> None:
    expected = "Sr1 Ti1 O3\n221\nSr 1 a\nTi 1 b\nO 3 c\n"
    assert Sr.get_wycryst() == expected


# @pytest.mark.parametrize("text_rep_str, pmg_structure", [(N2_str, N2.structure), (Sr_str, Sr.structure), (In_str, In.structure)])
# def test_wyckoff_matcher(text_rep_str: str, pmg_structure) -> None:
#     matcher = MatchRep(text_rep_str, pmg_structure)
#     assert matcher.wyckoff_matcher() == True
