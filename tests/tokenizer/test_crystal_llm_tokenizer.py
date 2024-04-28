import pytest
from xtal2txt.tokenizer import CrysllmTokenizer


@pytest.fixture
def tokenizer(scope="module"):
    return CrysllmTokenizer()


def test_tokenize(tokenizer):
    input_string = "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    tokens = tokenizer.tokenize(input_string)
    assert isinstance(tokens, list)


def test_convert_tokens_to_string(tokenizer):
    tokens = [
        "5",
        ".",
        "6",
        " ",
        "5",
        ".",
        "6",
        " ",
        "5",
        ".",
        "6",
        "\n",
        "9",
        "0",
        " ",
        "9",
        "0",
        " ",
        "9",
        "0",
        "\n",
        "N",
        "0",
        "+",
        "\n",
        "0",
        ".",
        "4",
        "8",
        " ",
        "0",
        ".",
        "9",
        "8",
        " ",
        "0",
        ".",
        "5",
        "2",
        "\n",
        "N",
        "0",
        "+",
        "\n",
        "0",
        ".",
        "9",
        "8",
        " ",
        "0",
        ".",
        "5",
        "2",
        " ",
        "0",
        ".",
        "4",
        "8",
        "\n",
        "N",
        "0",
        "+",
        "\n",
        "0",
        ".",
        "0",
        "2",
        " ",
        "0",
        ".",
        "0",
        "2",
        " ",
        "0",
        ".",
        "0",
        "2",
        "\n",
        "N",
        "0",
        "+",
        "\n",
        "0",
        ".",
        "5",
        "2",
        " ",
        "0",
        ".",
        "4",
        "8",
        " ",
        "0",
        ".",
        "9",
        "8",
    ]
    result = tokenizer.convert_tokens_to_string(tokens)
    assert (
        result
        == "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    )


def test_convert_token_to_id(tokenizer):
    token_id = tokenizer._convert_token_to_id("F")
    assert isinstance(token_id, int)


def test_convert_id_to_token(tokenizer):
    token = tokenizer._convert_id_to_token(0)
    assert isinstance(token, str)


def test_encode_decode(tokenizer):
    input_string = "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    token_ids = tokenizer.encode(input_string)
    decoded_tokens = tokenizer.decode(token_ids, skip_special_tokens=True)
    assert input_string == decoded_tokens
