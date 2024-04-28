import pytest
from xtal2txt.tokenizer import CompositionTokenizer


@pytest.fixture
def tokenizer(scope="module"):
    return CompositionTokenizer()


def test_tokenize(tokenizer):
    input_string = "SrTiO3"
    tokens = tokenizer.tokenize(input_string)
    assert isinstance(tokens, list)


def test_convert_tokens_to_string(tokenizer):
    tokens = ["Sr", "Ti", "O", "3"]
    result = tokenizer.convert_tokens_to_string(tokens)
    assert result == "SrTiO3"


def test_convert_token_to_id(tokenizer):
    token_id = tokenizer._convert_token_to_id("F")
    assert isinstance(token_id, int)


def test_convert_id_to_token(tokenizer):
    token = tokenizer._convert_id_to_token(0)
    assert isinstance(token, str)


def test_encode_decode(tokenizer):
    input_string = "SrTiO3"
    token_ids = tokenizer.encode(input_string)
    decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert input_string == decoded_tokens
    input_string_2 = "Cr4P16Pb4"
    token_ids = tokenizer.encode(input_string_2)
    decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)


@pytest.mark.parametrize(
    "input_string,expected",
    [
        ("Ba2ClSr", ["[CLS]", "Ba", "2", "Cl", "Sr", "[SEP]"]),
        ("BrMn2V", ["[CLS]", "Br", "Mn", "2", "V", "[SEP]"]),
        ("La2Ta4", ["[CLS]", "La", "2", "Ta", "4", "[SEP]"]),
        ("Cr4P16Pb4", ["[CLS]", "Cr", "4", "P", "1", "6", "Pb", "4", "[SEP]"]),
    ],
)
def test_tokenizer(tokenizer, input_string, expected):
    tokens = tokenizer.tokenize(input_string)
    assert tokens == expected
