import pytest
from xtal2txt.tokenizer import SliceTokenizer


@pytest.fixture
def tokenizer(scope="module"):
    return SliceTokenizer()


def test_tokenize(tokenizer):
    input_string = "Ga Ga Ga Ga P P P P 0 3 - - o 0 2 - o - 0 1 o - - 0 7 o o o 0 6 o o o 0 5 o o o 1 2 - + o 1 3 - o + 2 3 o - + 4 5 o o o 4 6 o o o 4 7 o o o 5 7 o o o 5 6 o o o 6 7 o o o"
    tokens = tokenizer.tokenize(input_string)
    assert isinstance(tokens, list)


def test_convert_tokens_to_string(tokenizer):
    tokens = [
        "Ga",
        "Ga",
        "P",
        "P",
        "0",
        "3",
        "- - o",
        "0",
        "2",
        "- o -",
        "0",
        "1",
        "o - -",
    ]
    result = tokenizer.convert_tokens_to_string(tokens)
    assert result == "Ga Ga P P 0 3 - - o 0 2 - o - 0 1 o - -"


def test_convert_token_to_id(tokenizer):
    token_id = tokenizer._convert_token_to_id("F")
    assert isinstance(token_id, int)


def test_convert_id_to_token(tokenizer):
    token = tokenizer._convert_id_to_token(0)
    assert isinstance(token, str)


def test_encode_decode(tokenizer):
    input_string = "Ga Ga Ga Ga P P P P 0 3 - - o 0 2 - o - 0 1 o - - 0 7 o o o 0 6 o o o 0 5 o o o 1 2 - + o 1 3 - o + 2 3 o - + 4 5 o o o 4 6 o o o 4 7 o o o 5 7 o o o 5 6 o o o 6 7 o o o"
    token_ids = tokenizer.encode(input_string)
    decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert input_string == decoded_tokens


@pytest.mark.parametrize(
    "input_string,expected",
    [
        (
            "Se Se Mo 0 2 o o + 0 2 + o o 0 2 o + o 1 2 o o + 1 2 o + o 1 2 + o o",
            [
                "[CLS]",
                "Se",
                "Se",
                "Mo",
                "0",
                "2",
                "o o +",
                "0",
                "2",
                "+ o o",
                "0",
                "2",
                "o + o",
                "1",
                "2",
                "o o +",
                "1",
                "2",
                "o + o",
                "1",
                "2",
                "+ o o",
                "[SEP]",
            ],
        ),
        ("H H O", ["[CLS]", "H", "H", "O", "[SEP]"]),
        ("Sc Sc 0 1 - - - ", ["[CLS]", "Sc", "Sc", "0", "1", "- - -", "[SEP]"]),
        (
            "Cu Cu Cu Cu 0 3 - - o 0 2 - o - 0 1 o - - 1 2 - + o 1 3 - o + 2 3 o - + ",
            [
                "[CLS]",
                "Cu",
                "Cu",
                "Cu",
                "Cu",
                "0",
                "3",
                "- - o",
                "0",
                "2",
                "- o -",
                "0",
                "1",
                "o - -",
                "1",
                "2",
                "- + o",
                "1",
                "3",
                "- o +",
                "2",
                "3",
                "o - +",
                "[SEP]",
            ],
        ),
    ],
)
def test_tokenizer(tokenizer, input_string, expected):
    tokens = tokenizer.tokenize(input_string)
    assert tokens == expected
