import pytest
from xtal2txt.core import TextRep
from pymatgen.core.structure import Structure as pyStructure
from xtal2txt.decoder import DecodeTextRep, MatchRep
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_symmetrized.cif"))
Sr = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "SrTiO3_symmetrized.cif"))
In = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "InCuS2_symmetrized.cif"))
Tl = TextRep.from_input(
    os.path.join(THIS_DIR, "..", "data", "TlCr5Se8_symmetrized.cif")
)

N2_str = "data_N2\n_symmetry_space_group_name_H-M   P2_13\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   198\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id \n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  '-x+1/2, -y, z+1/2'\n  3  'x+1/2, -y+1/2, -z'\n  4  '-x, y+1/2, -z+1/2'\n  5  'z, x, y'\n  6  'z+1/2, -x+1/2, -y'\n  7  '-z, x+1/2, -y+1/2'\n  8  '-z+1/2, -x, y+1/2'\n  9  'y, z, x'\n  10  '-y, z+1/2, -x+1/2'\n  11  '-y+1/2, -z, x+1/2'\n  12  'y+1/2, -z+1/2, -x'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  4  0.023  0.023  0.023  1.0\n"

Sr_str = "data_SrTiO3\n_symmetry_space_group_name_H-M   Pm-3m\n_cell_length_a   3.913\n_cell_length_b   3.913\n_cell_length_c   3.913\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   221\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   59.9\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  '-x, -y, -z'\n  3  '-y, x, z'\n  4  'y, -x, -z'\n  5  '-x, -y, z'\n  6  'x, y, -z'\n  7  'y, -x, z'\n  8  '-y, x, -z'\n  9  'x, -y, -z'\n  10  '-x, y, z'\n  11  '-y, -x, -z'\n  12  'y, x, z'\n  13  '-x, y, -z'\n  14  'x, -y, z'\n  15  'y, x, -z'\n  16  '-y, -x, z'\n  17  'z, x, y'\n  18  '-z, -x, -y'\n  19  'z, -y, x'\n  20  '-z, y, -x'\n  21  'z, -x, -y'\n  22  '-z, x, y'\n  23  'z, y, -x'\n  24  '-z, -y, x'\n  25  '-z, x, -y'\n  26  'z, -x, y'\n  27  '-z, -y, -x'\n  28  'z, y, x'\n  29  '-z, -x, y'\n  30  'z, x, -y'\n  31  '-z, y, x'\n  32  'z, -y, -x'\n  33  'y, z, x'\n  34  '-y, -z, -x'\n  35  'x, z, -y'\n  36  '-x, -z, y'\n  37  '-y, z, -x'\n  38  'y, -z, x'\n  39  '-x, z, y'\n  40  'x, -z, -y'\n  41  '-y, -z, x'\n  42  'y, z, -x'\n  43  '-x, -z, -y'\n  44  'x, z, y'\n  45  'y, -z, -x'\n  46  '-y, z, x'\n  47  'x, -z, y'\n  48  '-x, z, -y'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  Sr2+  2.0\n  Ti4+  4.0\n  O2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr2+  Sr0  1  0.0  0.0  0.0  1.0\n  Ti4+  Ti1  1  0.5  0.5  0.5  1.0\n  O2-  O2  3  0.0  0.5  0.5  1.0\n"

In_str = "data_InCuS2\n_symmetry_space_group_name_H-M   I-42d\n_cell_length_a   5.52\n_cell_length_b   5.52\n_cell_length_c   11.126\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   122     \n_chemical_formula_structural   InCuS2 \n_chemical_formula_sum   'In4 Cu4 S8'  \n_cell_volume   339.069\n_cell_formula_units_Z   4\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  'y, -x, -z'\n  3  '-x, -y, z'\n  4  '-y, x, -z'\n  5  'x, -y+1/2, -z+1/4'\n  6  'y, x+1/2, z+1/4'\n  7  '-x, y+1/2, -z+1/4'\n  8  '-y, -x+1/2, z+1/4'\n  9  'x+1/2, y+1/2, z+1/2'\n  10  'y+1/2, -x+1/2, -z+1/2'\n  11  '-x+1/2, -y+1/2, z+1/2'\n  12  '-y+1/2, x+1/2, -z+1/2'\n  13  'x+1/2, -y, -z+3/4'\n  14  'y+1/2, x, z+3/4'\n  15  '-x+1/2, y, -z+3/4'\n  16  '-y+1/2, -x, z+3/4'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  In3+  3.0\n  Cu+  1.0\n  S2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  In3+  In0  4  0.0  0.0  0.5  1.0\n  Cu+  Cu1  4  0.0  0.0  0.0  1.0\n  S2-  S2  8  0.221  0.25  0.125  1.0\n"

Tl_str = "data_TlCr5Se8\n_symmetry_space_group_name_H-M   C2/m\n_cell_length_a   18.931\n_cell_length_b   3.669\n_cell_length_c   9.064\n_cell_angle_alpha   90.0\n_cell_angle_beta   105.05\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   12\n_chemical_formula_structural   TlCr5Se8\n_chemical_formula_sum   'Tl2 Cr10 Se16'\n_cell_volume   607.987\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  '-x, -y, -z'\n  3  '-x, y, -z'\n  4  'x, -y, z'\n  5  'x+1/2, y+1/2, z'\n  6  '-x+1/2, -y+1/2, -z'\n  7  '-x+1/2, y+1/2, -z'\n  8  'x+1/2, -y+1/2, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  Tl+  1.0\n  Cr3+  3.0\n  Se2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Tl+  Tl0  2  0.0  0.0  0.5  1.0\n  Cr3+  Cr1  4  0.158  0.0  0.975  1.0\n  Cr3+  Cr2  4  0.204  0.0  0.334  1.0\n  Cr3+  Cr3  2  0.0  0.5  0.0  1.0\n  Se2-  Se4  4  0.076  0.0  0.153  1.0\n  Se2-  Se5  4  0.084  0.5  0.822  1.0\n  Se2-  Se6  4  0.17  0.5  0.49  1.0\n  Se2-  Se7  4  0.238  0.5  0.156  1.0\n"


@pytest.mark.parametrize("text_rep_str", [N2_str, Sr_str, In_str, Tl_str])
def test_cif_string_decoder_sym(text_rep_str: str) -> None:
    decoder = DecodeTextRep(text_rep_str)
    assert type(decoder.cif_string_decoder_sym(decoder.text)) == pyStructure


@pytest.mark.parametrize(
    "text_rep_str, pmg_structure",
    [
        (N2_str, N2.structure),
        (Sr_str, Sr.structure),
        (In_str, In.structure),
        (Tl_str, Tl.structure),
    ],
)
def test_cif_string_matcher_sym(text_rep_str: str, pmg_structure) -> None:
    matcher = MatchRep(text_rep_str, pmg_structure)
    assert matcher.cif_string_matcher_sym() == True


# def test_cif_string_matcher_p1() -> None:

#     assert N2.cif_string_matcher_sym(N2_str) == True
#     assert Sr.cif_string_matcher_sym(Sr_str) == True
#     assert In.cif_string_matcher_sym(In_str) == True
#     assert Tl.cif_string_matcher_sym(Tl_str) == True
