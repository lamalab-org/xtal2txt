import json
import os
import re

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from xtal2txt.analysis import (
    ANALYSIS_MASK_TOKENS,
    CIF_ANALYSIS_DICT,
    COMPOSITION_ANALYSIS_DICT,
    CRYSTAL_LLM_ANALYSIS_DICT,
    SLICE_ANALYSIS_DICT,
)

from typing import List


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

SLICE_VOCAB = os.path.join(THIS_DIR, "vocabs", "slice_vocab.txt")
SLICE_RT_VOCAB = os.path.join(THIS_DIR, "vocabs", "slice_vocab_rt.txt")

COMPOSITION_VOCAB = os.path.join(THIS_DIR, "vocabs", "composition_vocab.txt")
COMPOSITION_RT_VOCAB = os.path.join(THIS_DIR, "vocabs", "composition_vocab_rt.txt")

CIF_VOCAB = os.path.join(THIS_DIR, "vocabs", "cif_vocab.json")
CIF_RT_VOCAB = os.path.join(THIS_DIR, "vocabs", "cif_vocab_rt.json")

CRYSTAL_LLM_VOCAB = os.path.join(THIS_DIR, "vocabs", "crystal_llm_vocab.json")
CRYSTAL_LLM_RT_VOCAB = os.path.join(THIS_DIR, "vocabs", "crystal_llm_vocab_rt.json")

SMILES_VOCAB = os.path.join(THIS_DIR, "vocabs", "smiles_vocab.json")
SMILES_RT_VOCAB = os.path.join(THIS_DIR, "vocabs", "smiles_vocab_rt.json")

ROBOCRYS_VOCAB = os.path.join(THIS_DIR, "vocabs", "robocrys_vocab.json")


class NumTokenizer:
    """Tokenize numbers as implemented in Regression Transformer.
    https://www.nature.com/articles/s42256-023-00639-z
    https://github.com/IBM/regression-transformer/tree/main"""

    def __init__(self) -> None:
        """Tokenizer for numbers."""
        self.regex = re.compile(r"(\+|-)?(\d+)(\.)?(\d+)?\s*")

    def num_matcher(self, text: str) -> str:
        """Extract numbers from a sentence and replace them with tokens."""
        # pattern = re.findall(r'(\d+\.\d+|\d+)', text)  # This regex captures both whole numbers and decimal numbers

        pattern = (
            r"\d+(?:\.\d+)?"  # Match any number, whether it is part of a string or not
        )
        matches = list(re.finditer(pattern, text))
        for match in reversed(
            matches
        ):  # since we are replacing substring with a bigger subtring the string we are working on
            start, end = match.start(), match.end()
            tokens = self.tokenize(match.group())
            replacement = "".join(tokens)
            text = text[:start] + replacement + text[end:]
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenization of numbers as in RT.
         '0.9' -> '_0_0_', '_._', '_9_-1_'

        Args:
            text: number as string to be tokenized.

        Returns:
            extracted tokens.
        """
        tokens = []
        matched = self.regex.match(text)
        if matched:
            sign, units, dot, decimals = matched.groups()
            tokens = []
            if sign:
                tokens += [f"_{sign}_"]
            tokens += [
                f"_{number}_{position}_" for position, number in enumerate(units[::-1])
            ][::-1]
            if dot:
                tokens += [f"_{dot}_"]
            if decimals:
                tokens += [
                    f"_{number}_-{position}_"
                    for position, number in enumerate(decimals, 1)
                ]
        return tokens

    @staticmethod
    def convert_tokens_to_float(tokens: List[str]) -> float:
        """Converts tokens representing a float value into a float.
        NOTE: Expects that non-floating tokens are strippped off

        Args:
            tokens: List of tokens, each representing a float.
                E.g.: ['_0_0_', '_._', '_9_-1_', '_3_-2_', '_1_-3_']

        Returns:
            float: Float representation for the list of tokens.
        """
        try:
            float_string = "".join([token.split("_")[1] for token in tokens])
            float_value = float(float_string)
        except ValueError:
            float_value = -1
        return float_value

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Converts tokens to string.

        Args:
            tokens: List of tokens.

        Returns:
            str: String representation of the tokens.
        """
        return "".join([token.split("_")[1] for token in tokens])


class Xtal2txtTokenizer(PreTrainedTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=None,
        special_tokens={
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
        },
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        super(Xtal2txtTokenizer, self).__init__(
            model_max_length=model_max_length, **kwargs
        )
        self.truncation = False
        self.padding = False
        self.padding_length = padding_length

        self.special_num_tokens = special_num_token
        self.vocab = self.load_vocab(vocab_file)
        self.vocab_file = vocab_file

        # Initialize special tokens
        self.special_tokens = special_tokens if special_tokens is not None else {}
        self.add_special_tokens(self.special_tokens)

    def load_vocab(self, vocab_file):
        _, file_extension = os.path.splitext(vocab_file)
        if file_extension == ".txt":
            with open(vocab_file, "r", encoding="utf-8") as file:
                vocab = file.read().splitlines()
            return {token: idx for idx, token in enumerate(vocab)}
        elif file_extension == ".json":
            with open(vocab_file, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def get_vocab(self):
        return self.vocab

    def get_special_num_tokens(self, text):
        num_tokenizer = NumTokenizer()
        return num_tokenizer.num_matcher(text)

    def tokenize(self, text):
        if self.special_num_tokens:
            text = self.get_special_num_tokens(text)

        tokens = list(self.vocab.keys())
        string_tokens = [token for token in tokens if isinstance(token, str)]
        string_tokens.sort(key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in string_tokens]
        pattern_str = "|".join(escaped_tokens)
        pattern = re.compile(pattern_str)
        matches = pattern.findall(text)

        # Add [CLS] and [SEP] tokens if present in the vocabulary
        if self.cls_token is not None:
            matches = [self.cls_token] + matches

        if self.truncation and len(matches) > self.model_max_length:
            matches = matches[
                : self.model_max_length - 1
            ]  # -1 since we add sep token later

        if self.sep_token is not None:
            matches += [self.sep_token]

        if self.padding and len(matches) < self.padding_length:
            matches += [self.pad_token] * (self.padding_length - len(matches))

        return matches

    def convert_tokens_to_string(self, tokens):
        """Converts tokens to string."""
        if self.special_num_tokens:
            return "".join(
                [
                    token
                    if not (token.startswith("_") and token.endswith("_"))
                    else token.split("_")[1]
                    for token in tokens
                ]
            )
        return "".join(tokens)

    def _add_tokens(self, new_tokens, **kwargs):
        for token in new_tokens:
            if token not in self.added_tokens_encoder:
                self.vocab[token] = len(self.vocab)

    def _convert_token_to_id(self, token):
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        return list(self.vocab.keys())[index]

    def enable_truncation(self, max_length):
        self.model_max_length = max_length
        self.truncation = True

    def disable_truncation(self):
        self.truncation = False

    def enable_padding(self, length=None):
        self.padding = True
        self.padding_length = length

    def disable_padding(self):
        self.padding = False

    def add_special_tokens(self, special_tokens):
        for token, value in special_tokens.items():
            if value not in self.vocab:
                setattr(self, token, value)
                self.vocab[value] = len(self.vocab)

        # Ensure [CLS] and [SEP] tokens are added
        cls_token = special_tokens.get("cls_token", None)
        sep_token = special_tokens.get("sep_token", None)
        if cls_token is not None and cls_token not in self.vocab:
            setattr(self, "cls_token", cls_token)
            self.vocab[cls_token] = len(self.vocab)
        if sep_token is not None and sep_token not in self.vocab:
            setattr(self, "sep_token", sep_token)
            self.vocab[sep_token] = len(self.vocab)
        self.save_vocabulary(os.path.dirname(self.vocab_file))

    def token_analysis(self, tokens):
        """This method should be implemented by the Downstream tokenizers."""
        raise NotImplementedError

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocabulary, ensures vocabularies are not overwritten. Filename follow the convention {index}-{filename_prefix}.json. Index keeps track of the latest vocabulary saved."""
        index = 0
        if os.path.isdir(save_directory):
            vocab_files = list(
                filter(lambda x: x.endswith(".json"), os.listdir(save_directory))
            )
            for vocab_file in vocab_files:
                try:
                    index = max(index, int(vocab_file.split("-")[0]))
                except ValueError:
                    pass  # Ignore files that do not start with an integer

        vocab_file = os.path.join(
            save_directory,
            f"{index + 1}-{filename_prefix}.json"
            if filename_prefix
            else f"{index + 1}.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                vocab_files = list(
                    filter(
                        lambda x: x.endswith(".json"),
                        os.listdir(pretrained_model_name_or_path),
                    )
                )
                vocab_files.sort(key=lambda x: int(x.split("-")[0]))
                vocab_file = os.path.join(
                    pretrained_model_name_or_path, vocab_files[-1]
                )

        if vocab_file is None:
            raise ValueError("You should specify a path to a vocab file")

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        tokenizer = cls(vocab_file, *inputs, **kwargs)
        tokenizer.vocab = vocab

        return tokenizer


class SliceTokenizer(Xtal2txtTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=None,
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        if special_num_token:
            vocab_file = SLICE_RT_VOCAB if vocab_file is None else vocab_file
        else:
            vocab_file = SLICE_VOCAB if vocab_file is None else vocab_file
        super(SliceTokenizer, self).__init__(
            special_num_token=special_num_token,
            vocab_file=vocab_file,
            model_max_length=model_max_length,
            padding_length=padding_length,
            **kwargs,
        )

    def convert_tokens_to_string(self, tokens):
        """Converts tokens to string."""
        if self.special_num_tokens:
            return " ".join(
                [
                    token
                    if not (token.startswith("_") and token.endswith("_"))
                    else token.split("_")[1]
                    for token in tokens
                ]
            )
        return " ".join(tokens).rstrip()

    def token_analysis(self, list_of_tokens):
        """Takes tokens after tokenize and returns a list with replacing the tokens with their MASK token. The
        token type is determined from the dict declared globally, and the token is replaced with the corresponding MASK token."""
        analysis_masks = ANALYSIS_MASK_TOKENS
        token_type = SLICE_ANALYSIS_DICT
        return [
            analysis_masks[next((k for k, v in token_type.items() if token in v), None)]
            for token in list_of_tokens
        ]


class CompositionTokenizer(Xtal2txtTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=None,
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        if special_num_token:
            vocab_file = COMPOSITION_RT_VOCAB if vocab_file is None else vocab_file
        else:
            vocab_file = COMPOSITION_VOCAB if vocab_file is None else vocab_file
        super(CompositionTokenizer, self).__init__(
            special_num_token=special_num_token,
            vocab_file=vocab_file,
            model_max_length=model_max_length,
            padding_length=padding_length,
            **kwargs,
        )

    def token_analysis(self, list_of_tokens):
        """Takes tokens after tokenize and returns a list with replacing the tokens with their MASK token. The
        token type is determined from the dict declared globally, and the token is replaced with the corresponding MASK token."""
        analysis_masks = ANALYSIS_MASK_TOKENS
        token_type = COMPOSITION_ANALYSIS_DICT
        return [
            analysis_masks[next((k for k, v in token_type.items() if token in v), None)]
            for token in list_of_tokens
        ]


class CifTokenizer(Xtal2txtTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=None,
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        if special_num_token:
            vocab_file = CIF_RT_VOCAB
        else:
            vocab_file = CIF_VOCAB
        super(CifTokenizer, self).__init__(
            special_num_token=special_num_token,
            vocab_file=vocab_file,
            model_max_length=model_max_length,
            padding_length=padding_length,
            **kwargs,
        )

    def token_analysis(self, list_of_tokens):
        """Takes tokens after tokenize and returns a list with replacing the tokens with their MASK token. The
        token type is determined from the dict declared globally, and the token is replaced with the corresponding MASK token."""
        analysis_masks = ANALYSIS_MASK_TOKENS
        token_type = CIF_ANALYSIS_DICT
        return [
            analysis_masks[next((k for k, v in token_type.items() if token in v), None)]
            for token in list_of_tokens
        ]


class CrysllmTokenizer(Xtal2txtTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=CRYSTAL_LLM_VOCAB,
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        if special_num_token:
            vocab_file = CRYSTAL_LLM_RT_VOCAB
        else:
            vocab_file = CRYSTAL_LLM_VOCAB
        super(CrysllmTokenizer, self).__init__(
            special_num_token=special_num_token,
            vocab_file=vocab_file,
            model_max_length=model_max_length,
            padding_length=padding_length,
            **kwargs,
        )

    def token_analysis(self, list_of_tokens):
        """Takes tokens after tokenize and returns a list with replacing the tokens with their MASK token. The
        token type is determined from the dict declared globally, and the token is replaced with the corresponding MASK token."""
        analysis_masks = ANALYSIS_MASK_TOKENS
        token_type = CRYSTAL_LLM_ANALYSIS_DICT
        return [
            analysis_masks[next((k for k, v in token_type.items() if token in v), None)]
            for token in list_of_tokens
        ]


class SmilesTokenizer(Xtal2txtTokenizer):
    def __init__(
        self,
        special_num_token: bool = False,
        vocab_file=CRYSTAL_LLM_VOCAB,
        model_max_length=None,
        padding_length=None,
        **kwargs,
    ):
        if special_num_token:
            vocab_file = SMILES_RT_VOCAB
        else:
            vocab_file = SMILES_VOCAB
        super(SmilesTokenizer, self).__init__(
            special_num_token=special_num_token,
            vocab_file=vocab_file,
            model_max_length=model_max_length,
            padding_length=padding_length,
            **kwargs,
        )


class RobocrysTokenizer:
    """Tokenizer for Robocrystallographer. Would be BPE tokenizer.
    trained on the Robocrystallographer dataset.
    TODO: Implement this tokenizer.
    """

    def __init__(self, vocab_file=ROBOCRYS_VOCAB, **kwargs):
        tokenizer = Tokenizer.from_file(vocab_file)
        wrapped_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        self._tokenizer = wrapped_tokenizer

    def tokenize(self, text):
        return self._tokenizer.tokenize(text)

    def encode(self, text):
        return self._tokenizer.encode(text)

    def decode(self, token_ids, skip_special_tokens=True):
        # Check if token_ids is a string and convert it to a list of integers
        if isinstance(token_ids, str):
            token_ids = [int(token_ids)]
        return self._tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )
