import os
import pytest
import difflib

from xtal2txt.core import TextRep
from xtal2txt.tokenizer import (
    CifTokenizer,
    CrysllmTokenizer,
    SliceTokenizer,
    CompositionTokenizer,
)


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


structures = {
    "N2": TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_p1.cif")),
    "srtio3_p1": TextRep.from_input(
        os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif")
    ),
    "srtio3_symmetrized": TextRep.from_input(
        os.path.join(THIS_DIR, "..", "data", "SrTiO3_symmetrized.cif")
    ),
}


@pytest.fixture
def cif_rt_tokenizer(scope="module"):
    return CifTokenizer(
        special_num_token=True, model_max_length=512, truncation=False, padding=False
    )


@pytest.fixture
def crystal_llm_rt_tokenizer(scope="module"):
    return CrysllmTokenizer(
        special_num_token=True, model_max_length=512, truncation=False, padding=False
    )


@pytest.fixture
def slice_rt_tokenizer(scope="module"):
    return SliceTokenizer(
        special_num_token=True, model_max_length=512, truncation=False, padding=False
    )


@pytest.fixture
def composition_rt_tokenizer(scope="module"):
    return CompositionTokenizer(
        special_num_token=True, model_max_length=512, truncation=False, padding=False
    )


def print_diff(input_string, decoded_tokens):
    diff = difflib.ndiff(input_string.splitlines(1), decoded_tokens.splitlines(1))
    print("\n".join(diff))


def test_encode_decode(
    cif_rt_tokenizer,
    crystal_llm_rt_tokenizer,
    slice_rt_tokenizer,
    composition_rt_tokenizer,
):
    for name, struct in structures.items():
        input_string = struct.get_composition()
        token_ids = composition_rt_tokenizer.encode(input_string)
        decoded_tokens = composition_rt_tokenizer.decode(
            token_ids, skip_special_tokens=True
        )
        assert input_string == decoded_tokens

        input_string = struct.get_cif_string(format="p1", decimal_places=2)
        token_ids = cif_rt_tokenizer.encode(input_string)
        decoded_tokens = cif_rt_tokenizer.decode(token_ids, skip_special_tokens=True)
        assert input_string == decoded_tokens

        input_string = struct.get_crystal_text_llm()
        token_ids = crystal_llm_rt_tokenizer.encode(input_string)
        decoded_tokens = crystal_llm_rt_tokenizer.decode(
            token_ids, skip_special_tokens=True
        )
        assert input_string == decoded_tokens

        input_string = struct.get_slices()
        token_ids = slice_rt_tokenizer.encode(input_string)
        decoded_tokens = slice_rt_tokenizer.decode(token_ids, skip_special_tokens=True)
        try:
            assert input_string.strip() == decoded_tokens
        except AssertionError:
            print_diff(input_string, decoded_tokens)
            raise


def test_composition_rt_tokens(composition_rt_tokenizer) -> None:
    excepted_output = ["[CLS]", "Se", "_2_0_", "Se", "_3_0_", "[SEP]"]
    input_string = "Se2Se3"
    tokens = composition_rt_tokenizer.tokenize(input_string)
    assert tokens == excepted_output


def test_cif_rt_tokenize(cif_rt_tokenizer):
    input_string = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"
    tokens = cif_rt_tokenizer.tokenize(input_string)
    assert isinstance(tokens, list)


def test_cif_rt_tokens(cif_rt_tokenizer) -> None:
    excepted_output = [
        "[CLS]",
        "data_",
        "N",
        "_2_0_",
        "\n",
        "_symmetry_space_group_name_H-M",
        "   ",
        "'",
        "P",
        " ",
        "_1_0_",
        "'",
        "\n",
        "_cell_length_a",
        "   ",
        "_5_0_",
        "_._",
        "_6_-1_",
        "_0_-2_",
        "_5_-3_",
        "\n",
        "_cell_length_b",
        "   ",
        "_5_0_",
        "_._",
        "_6_-1_",
        "_0_-2_",
        "_5_-3_",
        "\n",
        "_cell_length_c",
        "   ",
        "_5_0_",
        "_._",
        "_6_-1_",
        "_0_-2_",
        "_5_-3_",
        "\n",
        "_cell_angle_alpha",
        "   ",
        "_9_1_",
        "_0_0_",
        "_._",
        "_0_-1_",
        "\n",
        "_cell_angle_beta",
        "   ",
        "_9_1_",
        "_0_0_",
        "_._",
        "_0_-1_",
        "\n",
        "_cell_angle_gamma",
        "   ",
        "_9_1_",
        "_0_0_",
        "_._",
        "_0_-1_",
        "\n",
        "_symmetry_Int_Tables_number",
        "   ",
        "_1_0_",
        "\n",
        "_chemical_formula_structural",
        "   ",
        "N",
        "_2_0_",
        "\n",
        "_chemical_formula_sum",
        "   ",
        "N",
        "_4_0_",
        "\n",
        "_cell_volume",
        "   ",
        "_1_2_",
        "_7_1_",
        "_6_0_",
        "_._",
        "_1_-1_",
        "_2_-2_",
        "_5_-3_",
        "\n",
        "_cell_formula_units_Z",
        "   ",
        "_2_0_",
        "\n",
        "loop_",
        "\n",
        " ",
        "_symmetry_equiv_pos_site_id",
        "\n",
        " ",
        "_symmetry_equiv_pos_as_xyz",
        "\n",
        "  ",
        "_1_0_",
        "  ",
        "'x, y, z'",
        "\n",
        "loop_",
        "\n",
        " ",
        "_atom_type_symbol",
        "\n",
        " ",
        "_atom_type_oxidation_number",
        "\n",
        "  ",
        "N",
        "_0_0_",
        "+",
        "  ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "\n",
        "loop_",
        "\n",
        " ",
        "_atom_site_type_symbol",
        "\n",
        " ",
        "_atom_site_label",
        "\n",
        " ",
        "_atom_site_symmetry_multiplicity",
        "\n",
        " ",
        "_atom_site_fract_x",
        "\n",
        " ",
        "_atom_site_fract_y",
        "\n",
        " ",
        "_atom_site_fract_z",
        "\n",
        " ",
        "_atom_site_occupancy",
        "\n",
        "  ",
        "N",
        "_0_0_",
        "+",
        "  ",
        "N",
        "_0_0_",
        "  ",
        "_1_0_",
        "  ",
        "_0_0_",
        "_._",
        "_4_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_9_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_1_0_",
        "_._",
        "_0_-1_",
        "\n",
        "  ",
        "N",
        "_0_0_",
        "+",
        "  ",
        "N",
        "_1_0_",
        "  ",
        "_1_0_",
        "  ",
        "_0_0_",
        "_._",
        "_9_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_4_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_1_0_",
        "_._",
        "_0_-1_",
        "\n",
        "  ",
        "N",
        "_0_0_",
        "+",
        "  ",
        "N",
        "_2_0_",
        "  ",
        "_1_0_",
        "  ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_1_0_",
        "_._",
        "_0_-1_",
        "\n",
        "  ",
        "N",
        "_0_0_",
        "+",
        "  ",
        "N",
        "_3_0_",
        "  ",
        "_1_0_",
        "  ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_2_-2_",
        "_3_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_4_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_0_0_",
        "_._",
        "_9_-1_",
        "_7_-2_",
        "_7_-3_",
        "  ",
        "_1_0_",
        "_._",
        "_0_-1_",
        "\n",
        "[SEP]",
    ]
    input_string = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"
    tokens = cif_rt_tokenizer.tokenize(input_string)
    assert tokens == excepted_output


def test_crystal_llm_tokens(crystal_llm_rt_tokenizer) -> None:
    excepted_output = [
        "[CLS]",
        "_3_0_",
        "_._",
        "_9_-1_",
        " ",
        "_3_0_",
        "_._",
        "_9_-1_",
        " ",
        "_3_0_",
        "_._",
        "_9_-1_",
        "\n",
        "_9_1_",
        "_0_0_",
        " ",
        "_9_1_",
        "_0_0_",
        " ",
        "_9_1_",
        "_0_0_",
        "\n",
        "Sr",
        "_2_0_",
        "+",
        "\n",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        "\n",
        "Ti",
        "_4_0_",
        "+",
        "\n",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        "\n",
        "O",
        "_2_0_",
        "-",
        "\n",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        "\n",
        "O",
        "_2_0_",
        "-",
        "\n",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        "\n",
        "O",
        "_2_0_",
        "-",
        "\n",
        "_0_0_",
        "_._",
        "_0_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        " ",
        "_0_0_",
        "_._",
        "_5_-1_",
        "_0_-2_",
        "[SEP]",
    ]
    input_string = "3.9 3.9 3.9\n90 90 90\nSr2+\n0.00 0.00 0.00\nTi4+\n0.50 0.50 0.50\nO2-\n0.50 0.00 0.50\nO2-\n0.50 0.50 0.00\nO2-\n0.00 0.50 0.50"
    tokens = crystal_llm_rt_tokenizer.tokenize(input_string)
    assert tokens == excepted_output
