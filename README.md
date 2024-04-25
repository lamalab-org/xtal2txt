<!--
<p align="center">
  <img src="https://github.com/lamalab-org/xtal2txt/raw/main/docs/source/logo.png" height="150">
</p>
-->

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
    <a href='https://xtal2txt.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/xtal2txt/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href="https://codecov.io/gh/lamalab-org/xtal2txt/branch/main">
        <img src="https://codecov.io/gh/lamalab-org/xtal2txt/branch/main/graph/badge.svg" alt="Codecov status" />
    </a>  
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-snekpack-blue" /> 
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
    <a href="https://github.com/lamalab-org/xtal2txt/blob/main/.github/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg" alt="Contributor Covenant"/>
    </a>
</p>

Package to define, convert , encode and decode crystal structures into text representrations

## 💪 Getting Started

> TODO show in a very small amount of space the **MOST** useful thing your package can do.
> Make it as short as possible! You have an entire set of docs for later.


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

## Using Xtal2Txt Tokenizers

By default the, the tokenizer is initialized with \[CLS\] and \[SEP\]
tokens. For example see the SliceTokenizer usage.

Initialization with \[CLS\] and \[SEP\] Tokens:

``` python
from xtal2txt.tokenizer import SliceTokenizer

tokenizer = SliceTokenizer(
                model_max_length=512, 
                truncation=True, 
                padding="max_length", 
                max_length=512
            )
print(tokenizer.cls_token) #[CLS]
```

You can access the \[CLS\] token using the [cls_token]{.title-ref}
attribute of the tokenizer. During decoding, you can utilize the
[skip_special_tokens]{.title-ref} parameter to skip these special
tokens.

Decoding with Skipping Special Tokens:

``` python
tokenizer.decode(token_ids, skip_special_tokens=True)
```


## Text Representation with Xtal2Txt

The `TextRep` class in `xtal2txt.core`
facilitates the transformation of crystal structures into different text
representations. Below is an example of its usage:

``` python
from xtal2txt.core import TextRep
from pymatgen.core import Structure


# Load structure from a CIF file
from_file = "/home/so87pot/n0w0f/xtal2txt/tests/data/InCuS2_p1.cif"
structure = Structure.from_file(from_file, "cif")

Initialize TextRep Class
text_rep = TextRep.from_input(structure)

requested_reps = [
        "cif_p1",
        "slice",
        "atoms",
        "atoms_params",
        "crystal_llm_rep",
        "zmatrix"
    ]

# Get the requested text representations
requested_text_reps = text_rep.get_requested_text_reps(requested_reps)

See more details in docs

```


## Initializing tokenizers with custom special tokens

In scenarios where the \[CLS\] token is not required, you can initialize
the tokenizer with an empty special_tokens dictionary.

Initialization without \[CLS\] and \[SEP\] Tokens:

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
[PreTrainedTokenizer](https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer) and accept arguments compatible with the Hugging Face
tokenizer.

## Tokenizers with Special Number Tokenization

The `special_num_token` argument (by default False) can be
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

<!--
### 📖 Citation

Citation goes here!
-->

<!--
### 🎁 Support

This project has been supported by the following organizations (in alphabetical order):

- [Harvard Program in Therapeutic Science - Laboratory of Systems Pharmacology](https://hits.harvard.edu/the-program/laboratory-of-systems-pharmacology/)

-->

<!--
### 💰 Funding

This project has been supported by the following grants:

| Funding Body                                             | Program                                                                                                                       | Grant           |
|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|-----------------|
| DARPA                                                    | [Automating Scientific Knowledge Extraction (ASKE)](https://www.darpa.mil/program/automating-scientific-knowledge-extraction) | HR00111990009   |
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.

## 🛠️ For Developers

<details>
  <summary>See developer instructions</summary>

The final section of the README is for if you want to get involved by making a code contribution.

### Development Installation

To install in development mode, use the following:

```bash
$ git clone git+https://github.com/lamalab-org/xtal2txt.git
$ cd xtal2txt
$ pip install -e .
```

### 🥼 Testing

After cloning the repository and installing `tox` with `pip install tox`, the unit tests in the `tests/` folder can be
run reproducibly with:

```shell
$ tox
```

Additionally, these tests are automatically re-run with each commit in a [GitHub Action](https://github.com/lamalab-org/xtal2txt/actions?query=workflow%3ATests).

### 📖 Building the Documentation

The documentation can be built locally using the following:

```shell
$ git clone git+https://github.com/lamalab-org/xtal2txt.git
$ cd xtal2txt
$ tox -e docs
$ open docs/build/html/index.html
``` 

The documentation automatically installs the package as well as the `docs`
extra specified in the [`setup.cfg`](setup.cfg). `sphinx` plugins
like `texext` can be added there. Additionally, they need to be added to the
`extensions` list in [`docs/source/conf.py`](docs/source/conf.py).

### 📦 Making a Release

After installing the package in development mode and installing
`tox` with `pip install tox`, the commands for making a new release are contained within the `finish` environment
in `tox.ini`. Run the following from the shell:

```shell
$ tox -e finish
```

This script does the following:

1. Uses [Bump2Version](https://github.com/c4urself/bump2version) to switch the version number in the `setup.cfg`,
   `src/xtal2txt/version.py`, and [`docs/source/conf.py`](docs/source/conf.py) to not have the `-dev` suffix
2. Packages the code in both a tar archive and a wheel using [`build`](https://github.com/pypa/build)
3. Uploads to PyPI using [`twine`](https://github.com/pypa/twine). Be sure to have a `.pypirc` file configured to avoid the need for manual input at this
   step
4. Push to GitHub. You'll need to make a release going with the commit where the version was bumped.
5. Bump the version to the next patch. If you made big changes and want to bump the version by minor, you can
   use `tox -e bumpversion -- minor` after.
</details>
