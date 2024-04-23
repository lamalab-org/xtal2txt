import pytest
from xtal2txt.tokenizer import RobocrysTokenizer


@pytest.fixture
def tokenizer(scope="module"):
    return RobocrysTokenizer()


def test_tokenize(tokenizer):
    input_string = "(Zn)2SnSe2 is Indium-derived structured and crystallizes in the monoclinic Cm space group. The structure is zero-dimensional and consists of two stannic selenide molecules and two zinc molecules."
    tokens = tokenizer.tokenize(input_string)
    assert isinstance(tokens, list)


def test_convert_tokens_to_string(tokenizer):
    input_string = "(Zn)2SnSe2 is Indium-derived structured and crystallizes in the monoclinic Cm space group. The structure is zero-dimensional and consists of two stannic selenide molecules and two zinc molecules."
    tokens = [
        "(",
        "Zn",
        ")",
        "2",
        "SnSe2",
        "is",
        "Indium",
        "-",
        "derived",
        "structured",
        "and",
        "crystallizes",
        "in",
        "the",
        "monoclinic",
        "Cm",
        "space",
        "group",
        ".",
        "The",
        "structure",
        "is",
        "zero",
        "-",
        "dimensional",
        "and",
        "consists",
        "of",
        "two",
        "stan",
        "nic",
        "selenide",
        "molecules",
        "and",
        "two",
        "zinc",
        "molecules",
        ".",
    ]
    result = tokenizer.tokenize(input_string)
    assert result == tokens
