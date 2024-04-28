import pytest
from xtal2txt.tokenizer import CifTokenizer
from xtal2txt.core import TextRep
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

structures = {
    "N2": TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_p1.cif")),
    "srtio3_p1": TextRep.from_input(
        os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif")
    ),
    "srtio3_symmetrized": TextRep.from_input(
        os.path.join(THIS_DIR, "..", "data", "SrTiO3_symmetrized.cif")
    ),
}


@pytest.fixture
def tokenizer(scope="module"):
    return CifTokenizer()


def test_encode_decode(tokenizer):
    for name, struct in structures.items():
        input_string = struct.get_cif_string()
        token_ids = tokenizer.encode(input_string)
        decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)
        assert input_string == decoded_tokens
