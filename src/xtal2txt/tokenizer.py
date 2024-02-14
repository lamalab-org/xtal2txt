import os
import re
import json
from pathlib import Path

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from tokenizers import Tokenizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SLICE_VOCAB = os.path.join(THIS_DIR, "vocabs", "slice_vocab.txt")
COMPOSITION_VOCAB = os.path.join(THIS_DIR, "vocabs", "composition_vocab.txt")
CIF_VOCAB = os.path.join(THIS_DIR, "vocabs", "cif_vocab.json")
CRYSTAL_LLM_VOCAB = os.path.join(THIS_DIR, "vocabs", "crystal_llm_vocab.json")
ROBOCRYS_VOCAB = os.path.join(THIS_DIR, "vocabs", "robocrys_vocab.json")


class Xtal2txtTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab_file, model_max_length=None, padding_length=None, **kwargs):
        super(Xtal2txtTokenizer, self).__init__(model_max_length=model_max_length, **kwargs)
        
        self.vocab = self.load_vocab(vocab_file)
        self.vocab_file = vocab_file
        self.truncation = False
        self.padding = False
        self.padding_length = padding_length

    def load_vocab(self, vocab_file):
        _, file_extension = os.path.splitext(vocab_file)
        if file_extension == '.txt':
            with open(vocab_file, 'r', encoding='utf-8') as file:
                vocab = file.read().splitlines()
            return {token: idx for idx, token in enumerate(vocab)}
        elif file_extension == '.json':
            with open(vocab_file, 'r', encoding='utf-8') as file:
                return json.load(file)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
    def get_vocab(self):
        return self.vocab

    def tokenize(self, text):
        tokens = list(self.vocab.keys())
        string_tokens = [token for token in tokens if isinstance(token, str)]
        string_tokens.sort(key=len, reverse=True)
        escaped_tokens = [re.escape(token) for token in string_tokens]
        pattern_str = '|'.join(escaped_tokens)
        pattern = re.compile(pattern_str)
        matches = pattern.findall(text)

        if self.truncation and len(matches) > self.model_max_length:
            matches = matches[:self.model_max_length]

        if self.padding and len(matches) < self.padding_length:
            matches += [self.pad_token] * (self.padding_length - len(matches))

        return matches

    def convert_tokens_to_string(self, tokens):
        return ' '.join(tokens)

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
        self.save_vocabulary(os.path.dirname(self.vocab_file))

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """Save the vocabulary, ensures vocabularies are not overwritten. Filename follow the convention {index}-{filename_prefix}.json. Index keeps track of the latest vocabulary saved."""
        index = 0
        if os.path.isdir(save_directory):
            vocab_files = list(filter(lambda x: x.endswith(".json"), os.listdir(save_directory)))
            for vocab_file in vocab_files:
                try:
                    index = max(index, int(vocab_file.split('-')[0]))
                except ValueError:
                    pass  # Ignore files that do not start with an integer

        vocab_file = os.path.join(save_directory, f"{index + 1}-{filename_prefix}.json" if filename_prefix else f"{index + 1}.json")

        with open(vocab_file, 'w', encoding='utf-8') as f:
            json.dump(self.vocab, f, ensure_ascii=False)

        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if pretrained_model_name_or_path is not None:
            if os.path.isdir(pretrained_model_name_or_path):
                vocab_files = list(filter(lambda x: x.endswith(".json"), os.listdir(pretrained_model_name_or_path)))
                vocab_files.sort(key=lambda x: int(x.split('-')[0]))
                vocab_file = os.path.join(pretrained_model_name_or_path, vocab_files[-1])

        if vocab_file is None:
            raise ValueError("You should specify a path to a vocab file")

        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        tokenizer = cls(vocab_file, *inputs, **kwargs)
        tokenizer.vocab = vocab

        return tokenizer



class SliceTokenizer(Xtal2txtTokenizer):
    def __init__(self, vocab_file=SLICE_VOCAB, model_max_length=None, padding_length=None, **kwargs):
        super(SliceTokenizer, self).__init__(vocab_file, model_max_length=model_max_length, padding_length=padding_length, **kwargs)



class CompositionTokenizer(Xtal2txtTokenizer):
    def __init__(self, vocab_file=COMPOSITION_VOCAB, model_max_length=None, padding_length=None, **kwargs):
        super(CompositionTokenizer, self).__init__(vocab_file, model_max_length=model_max_length, padding_length=padding_length, **kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)
    

class CifTokenizer(Xtal2txtTokenizer):
    def __init__(self, vocab_file=CIF_VOCAB, model_max_length=None, padding_length=None, **kwargs):
        super(CifTokenizer, self).__init__(vocab_file, model_max_length=model_max_length, padding_length=padding_length, **kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)

class CrysllmTokenizer(Xtal2txtTokenizer):
    def __init__(self, vocab_file=CRYSTAL_LLM_VOCAB, model_max_length=None, padding_length=None, **kwargs):
        super(CrysllmTokenizer, self).__init__(vocab_file, model_max_length=model_max_length, padding_length=padding_length, **kwargs)

    def convert_tokens_to_string(self, tokens):
        return ''.join(tokens)


class RobocrysTokenizer():
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
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    

