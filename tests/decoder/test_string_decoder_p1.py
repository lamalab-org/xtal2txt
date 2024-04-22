import pytest
from xtal2txt.core import TextRep
from pymatgen.core.structure import Structure as pyStructure
from xtal2txt.decoder import DecodeTextRep, MatchRep
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_p1.cif"))
Sr = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif"))
In = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "InCuS2_p1.cif"))
Tl = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "TlCr5Se8_p1.cif"))

N2_str = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"

Sr_str = "data_SrTiO3\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   3.913\n_cell_length_b   3.913\n_cell_length_c   3.913\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   SrTiO3\n_chemical_formula_sum   'Sr1 Ti1 O3'\n_cell_volume   59.9\n_cell_formula_units_Z   1\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  Sr2+  2.0\n  Ti4+  4.0\n  O2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Sr2+  Sr0  1  0.0  0.0  0.0  1.0\n  Ti4+  Ti1  1  0.5  0.5  0.5  1.0\n  O2-  O2  1  0.5  0.0  0.5  1.0\n  O2-  O3  1  0.5  0.5  0.0  1.0\n  O2-  O4  1  0.0  0.5  0.5  1.0\n"

In_str = "data_InCuS2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.52\n_cell_length_b   5.52\n_cell_length_c   11.126\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   InCuS2\n_chemical_formula_sum   'In4 Cu4 S8'\n_cell_volume   339.069\n_cell_formula_units_Z   4\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  In3+  3.0\n  Cu+  1.0\n  S2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  In3+  In0  1  0.5  0.5  0.0  1.0\n  In3+  In1  1  0.0  0.5  0.75  1.0\n  In3+  In2  1  0.0  0.0  0.5  1.0\n  In3+  In3  1  0.5  0.0  0.25  1.0\n  Cu+  Cu4  1  0.5  0.0  0.75  1.0\n  Cu+  Cu5  1  0.0  0.0  0.0  1.0\n  Cu+  Cu6  1  0.0  0.5  0.25  1.0\n  Cu+  Cu7  1  0.5  0.5  0.5  1.0\n  S2-  S8  1  0.279  0.25  0.625  1.0\n  S2-  S9  1  0.75  0.221  0.875  1.0\n  S2-  S10  1  0.721  0.75  0.625  1.0\n  S2-  S11  1  0.25  0.779  0.875  1.0\n  S2-  S12  1  0.779  0.75  0.125  1.0\n  S2-  S13  1  0.25  0.721  0.375  1.0\n  S2-  S14  1  0.221  0.25  0.125  1.0\n  S2-  S15  1  0.75  0.279  0.375  1.0\n"

Tl_str = "data_TlCr5Se8\n_symmetry_space_group_name_H-M   'P 1' \n_cell_length_a   18.931\n_cell_length_b   3.669\n_cell_length_c   9.064\n_cell_angle_alpha   90.0\n_cell_angle_beta   105.05\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   TlCr5Se8\n_chemical_formula_sum   'Tl2 Cr10 Se16'\n_cell_volume   607.987\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  Tl+  1.0\n  Cr3+  3.0\n  Se2-  -2.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  Tl+  Tl0  1  0.5  0.5  0.5  1.0\n  Tl+  Tl1  1  0.0  0.0  0.5  1.0\n  Cr3+  Cr2  1  0.0  0.5  0.0  1.0\n  Cr3+  Cr3  1  0.342  0.5  0.025  1.0\n  Cr3+  Cr4  1  0.296  0.5  0.666  1.0\n  Cr3+  Cr5  1  0.204  0.0  0.334  1.0\n  Cr3+  Cr6  1  0.158  0.0  0.975  1.0\n  Cr3+  Cr7  1  0.5  0.0  0.0  1.0\n  Cr3+  Cr8  1  0.842  0.0  0.025  1.0\n  Cr3+  Cr9  1  0.796  0.0  0.666  1.0\n  Cr3+  Cr10  1  0.704  0.5  0.334  1.0\n  Cr3+  Cr11  1  0.658  0.5  0.975  1.0\n  Se2-  Se12  1  0.076  0.0  0.153  1.0\n  Se2-  Se13  1  0.17  0.5  0.49  1.0\n  Se2-  Se14  1  0.33  0.0  0.51  1.0\n  Se2-  Se15  1  0.084  0.5  0.822  1.0\n  Se2-  Se16  1  0.416  0.0  0.178  1.0\n  Se2-  Se17  1  0.262  0.0  0.844  1.0\n  Se2-  Se18  1  0.238  0.5  0.156  1.0\n  Se2-  Se19  1  0.424  0.5  0.847  1.0\n  Se2-  Se20  1  0.576  0.5  0.153  1.0\n  Se2-  Se21  1  0.67  0.0  0.49  1.0\n  Se2-  Se22  1  0.83  0.5  0.51  1.0\n  Se2-  Se23  1  0.584  0.0  0.822  1.0\n  Se2-  Se24  1  0.916  0.5  0.178  1.0\n  Se2-  Se25  1  0.762  0.5  0.844  1.0\n  Se2-  Se26  1  0.738  0.0  0.156  1.0\n  Se2-  Se27  1  0.924  0.0  0.847  1.0\n"


@pytest.mark.parametrize("text_rep_str", [N2_str, Sr_str, In_str, Tl_str])
def test_get_crystal_llm_rep_struc(text_rep_str: str) -> None:
    decoder = DecodeTextRep(text_rep_str)
    assert type(decoder.cif_string_decoder_p1(decoder.text)) == pyStructure


@pytest.mark.parametrize(
    "text_rep_str, pmg_structure",
    [
        (N2_str, N2.structure),
        (Sr_str, Sr.structure),
        (In_str, In.structure),
        (Tl_str, Tl.structure),
    ],
)
def test_cif_string_matcher_p1(text_rep_str: str, pmg_structure) -> None:
    matcher = MatchRep(text_rep_str, pmg_structure)
    assert matcher.cif_string_matcher_p1() == True
