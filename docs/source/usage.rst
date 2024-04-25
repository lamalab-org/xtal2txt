Usage
=====

Using Xtal2Txt Tokenizers 
-------------------------
By default the, the tokenizer is initialized with [CLS] and [SEP] tokens. 
For example see the SliceTokenizer usage. 

Initialization with [CLS] and [SEP] Tokens:

.. code-block:: python


    from xtal2txt.tokenizer import SliceTokenizer

    tokenizer = SliceTokenizer(
                    model_max_length=512, 
                    truncation=True, 
                    padding="max_length", 
                    max_length=512
                )
    print(tokenizer.cls_token) #[CLS]
    
You can access the [CLS] token using the `cls_token` attribute of the tokenizer. During decoding, you can utilize the `skip_special_tokens` parameter to skip these special tokens. 

Decoding with Skipping Special Tokens:

.. code-block:: python

    tokenizer.decode(token_ids, skip_special_tokens=True)
    
 

Initializing tokenizers with custom special tokens
--------------------------------------------------

In scenarios where the [CLS] token is not required, you can initialize the tokenizer with an empty special_tokens dictionary.

Initialization without [CLS] and [SEP] Tokens:

.. code-block:: python

    tokenizer = SliceTokenizer(
                    model_max_length=512, 
                    special_tokens={}, 
                    truncation=True,
                    padding="max_length", 
                    max_length=512
                )


All `Xtal2txtTokenizer` instances inherit from :ref:`PreTrainedTokenizer <regression_transformer>` and accept arguments compatible with the Hugging Face tokenizer.


Tokenizers with Special Number Tokenization
--------------------------------------------

The `special_num_token` argument (by default False) can be set to true to tokenize numbers in a special way as designed and implemented by :ref:`RegressionTransformer <regression_transformer>`.

.. code-block:: python

    tokenizer = SliceTokenizer(
    				special_num_token=True,
                    model_max_length=512, 
                    special_tokens={}, 
                    truncation=True,
                    padding="max_length", 
                    max_length=512
                )
                

.. _pretrained_tokenizer:

PreTrainedTokenizer: https://huggingface.co/docs/transformers/v4.40.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer

.. _regression_transformer:

RegressionTransformer: https://www.nature.com/articles/s42256-023-00639-z

.. automodule:: xtal2txt.api
    :members:
