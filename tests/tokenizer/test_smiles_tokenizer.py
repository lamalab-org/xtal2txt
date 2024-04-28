import pytest
from xtal2txt.tokenizer import SmilesTokenizer
import os


@pytest.fixture
def smiles_rt_tokenizer(scope="module"):
    return SmilesTokenizer(
        special_num_token=True, model_max_length=512, truncation=False, padding=False
    )



def test_smiles_rt_tokens(smiles_rt_tokenizer) -> None:
    excepted_output = ['[CLS]', 'I', '-', '_4_1_', '_2_0_', 'd', '\n', 'S', '_2_0_', '-', ' ', '(', '_8_0_', 'd', ')', ' ', '[', 'Cu', ']', 'S', '(', '[', 'In', ']', ')', '(', '[', 'In', ']', ')', '[', 'Cu', ']', '\n', 'Cu', '+', ' ', '(', '_4_0_', 'a', ')',  ' ','[', 'S', ']', '[', 'Cu', ']', '(', '[', 'S', ']', ')', '(', '[', 'S', ']', ')', '[', 'S', ']', '\n', 'In', '_3_0_', '+', ' ','(', '_4_0_', 'b', ')', ' ','[', 'S', ']', '[', 'In', ']', '(', '[', 'S', ']', ')', '[', 'S', ']', '.', '[', 'S', ']', '[SEP]']
    input_string = "I-42d\nS2- (8d) [Cu]S([In])([In])[Cu]\nCu+ (4a) [S][Cu]([S])([S])[S]\nIn3+ (4b) [S][In]([S])[S].[S]"
    tokens = smiles_rt_tokenizer.tokenize(input_string)
    assert tokens == excepted_output
