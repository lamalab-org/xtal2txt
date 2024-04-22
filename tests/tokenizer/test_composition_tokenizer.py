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
    decoded_tokens = tokenizer.decode(token_ids)
    assert input_string == decoded_tokens
    input_string_2 = "Cr4P16Pb4"
    token_ids = tokenizer.encode(input_string_2)
    decoded_tokens = tokenizer.decode(token_ids)


@pytest.mark.parametrize(
    "input_string,expected",
    [
        ("Ba2ClSr", ["Ba", "2", "Cl", "Sr"]),
        ("BrMn2V", ["Br", "Mn", "2", "V"]),
        ("La2Ta4", ["La", "2", "Ta", "4"]),
        ("Cr4P16Pb4", ["Cr", "4", "P", "1", "6", "Pb", "4"]),
    ],
)
def test_tokenizer(tokenizer, input_string, expected):
    tokens = tokenizer.tokenize(input_string)
    assert tokens == expected
