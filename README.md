
<p align="center">
  <img src="https://github.com/lamalab-org/xtal2txt/raw/main/docs/static/xtal2txt-logo.png" height="150">
</p>


<h1 align="center">
  xtal2txt
</h1>

<p align="center">
    <a href="https://github.com/lamalab-org/xtal2txt/actions/workflows/tests.yml">
        <img alt="Tests" src="https://github.com/lamalab-org/xtal2txt/workflows/Tests/badge.svg" />
    </a>
    <a href="https://pypi.org/project/xtal2txt">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/xtal2txt" />
    </a>
    <a href="https://pypi.org/project/xtal2txt">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/xtal2txt" />
    </a>
    <a href="https://github.com/lamalab-org/xtal2txt/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/xtal2txt" />
    </a>
</p>

Package to define, convert, encode and decode crystal structures into text representations

## 💪 Getting Started

## 🚀 Installation

<!-- Uncomment this section after your first ``tox -e finish``
The most recent release can be installed from
[PyPI](https://pypi.org/project/xtal2txt/) with:

```shell
$ pip install xtal2txt
```
-->

The most recent code and data can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/lamalab-org/xtal2txt.git
```


## Text Representation with xtal2txt

The `TextRep` class in `xtal2txt.core`
facilitates the transformation of crystal structures into different text
representations. Below is an example of its usage:

``` python
from xtal2txt.core import TextRep
from pymatgen.core import Structure


# Load structure from a CIF file
from_file = "InCuS2_p1.cif"
structure = Structure.from_file(from_file, "cif")

# Initialize TextRep Class
text_rep = TextRep.from_input(structure)

requested_reps = [
        "cif_p1",
        "slices",
        "atom_sequences",
        "atom_sequences_plusplus",
        "crystal_text_llm",
        "zmatrix"
]

# Get the requested text representations
requested_text_reps = text_rep.get_requested_text_reps(requested_reps)
```


## Using xtal2txt Tokenizers

By default, the tokenizer is initialized with `\[CLS\]` and `\[SEP\]`
tokens. For an example, see the `SliceTokenizer` usage: 

``` python
from xtal2txt.tokenizer import SliceTokenizer

tokenizer = SliceTokenizer(
                model_max_length=512, 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
print(tokenizer.cls_token) # returns [CLS]
```

You can access the `\[CLS\]` token using the [cls_token]{.title-ref}
attribute of the tokenizer. During decoding, you can utilize the
[skip_special_tokens]{.title-ref} parameter to skip these special
tokens.

Decoding with skipping special tokens:

``` python
tokenizer.decode(token_ids, skip_special_tokens=True)
```


## Initializing tokenizers with custom special tokens

In scenarios where the `\[CLS\]` token is not required, you can initialize
the tokenizer with an empty special_tokens dictionary.

Initialization without `\[CLS\]` and `\[SEP\]` tokens:

``` python
tokenizer = SliceTokenizer(
                model_max_length=512, 
                special_tokens={}, 
                truncation=True,
                padding="max_length", 
                max_length=512
            )
```

All `Xtal2txtTokenizer` instances inherit from
[PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and accept arguments compatible with the Hugging Face tokenizer.

## Tokenizers with special number tokenization

The `special_num_token` argument (by default `False`) can be
set to true to tokenize numbers in a special way as designed and
implemented by
[RegressionTransformer](https://www.nature.com/articles/s42256-023-00639-z).

``` python
tokenizer = SliceTokenizer(
                special_num_token=True,
                model_max_length=512, 
                special_tokens={}, 
                truncation=True,
                padding="max_length", 
                max_length=512
            )
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.md](https://github.com/lamalab-org/xtal2txt/blob/master/.github/CONTRIBUTING.md) for more information on getting involved.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the MIT License.
See the [Notice](NOTICE.txt) for imported LGPL code.

### 💰 Funding

This project has been supported by the [Carl Zeiss Foundation](https://www.carl-zeiss-stiftung.de/en/) as well as Intel and Merck.