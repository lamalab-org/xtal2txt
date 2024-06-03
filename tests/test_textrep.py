from xtal2txt.core import TextRep
import os
import pytest
from pymatgen.core import Structure

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "data", "N2_p1.cif"))
srtio3_p1 = TextRep.from_input(os.path.join(THIS_DIR, "data", "SrTiO3_p1.cif"))
srtio3_symmetrized = TextRep.from_input(
    os.path.join(THIS_DIR, "data", "SrTiO3_symmetrized.cif")
)

CIF_PATH = os.path.join(THIS_DIR, "data", "SrTiO3_p1.cif")
CIF_STRING = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"


@pytest.fixture
def text_rep_instance():
    return TextRep


def test_from_input_with_structure_instance(text_rep_instance):
    structure = Structure.from_file(CIF_PATH)
    text_rep = text_rep_instance.from_input(structure)
    assert isinstance(text_rep, TextRep)


def test_from_input_with_cif_file_path(text_rep_instance):
    text_rep = text_rep_instance.from_input(CIF_PATH)
    assert isinstance(text_rep, TextRep)


def test_from_input_with_cif_string(text_rep_instance):
    text_rep = text_rep_instance.from_input(CIF_STRING)
    assert isinstance(text_rep, TextRep)


def test_get_cif_string_p1() -> None:
    expected_output = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"
    assert N2.get_cif_string(format="p1", decimal_places=3) == expected_output


def test_get_cif_string_sym(format="symmetrized", decimal_places=3) -> None:
    expected_output = "data_N2\n_symmetry_space_group_name_H-M   P2_13\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   198\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\n  2  '-x+1/2, -y, z+1/2'\n  3  'x+1/2, -y+1/2, -z'\n  4  '-x, y+1/2, -z+1/2'\n  5  'z, x, y'\n  6  'z+1/2, -x+1/2, -y'\n  7  '-z, x+1/2, -y+1/2'\n  8  '-z+1/2, -x, y+1/2'\n  9  'y, z, x'\n  10  '-y, z+1/2, -x+1/2'\n  11  '-y+1/2, -z, x+1/2'\n  12  'y+1/2, -z+1/2, -x'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  4  0.023  0.023  0.023  1.0\n"
    assert N2.get_cif_string(format="symmetrized", decimal_places=3) == expected_output


def test_get_lattice_parameters() -> None:
    expected_output = ["5.605", "5.605", "5.605", "90.0", "90.0", "90.0"]
    assert N2.get_lattice_parameters() == expected_output


def test_get_slices() -> None:
    expected_output = "N N N N 0 1 - o o 0 1 - + o 0 1 o o o 0 1 o + o 0 2 o + o 0 2 o + + 0 2 + + o 0 2 + + + 0 3 o o - 0 3 o o o 0 3 o + - 0 3 o + o 1 3 o o - 1 3 o o o 1 3 + o - 1 3 + o o 1 2 + o o 1 2 + o + 1 2 + + o 1 2 + + + 2 3 - - - 2 3 - o - 2 3 o - - 2 3 o o - "
    assert N2.get_slices() == expected_output


def test_get_crystal_llm_rep() -> None:
    expected_output = "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    assert N2.get_crystal_text_llm() == expected_output


def test_robocrys_for_cif_format() -> None:
    assert srtio3_p1.get_robocrys_rep() == srtio3_symmetrized.get_robocrys_rep()


# TODO: check N2 robocrys represetation
# def test_get_robocrys_rep() -> None:
#     excepted_output = "N2 is Indium-like structured and crystallizes in the cubic P2_13 space group. The structure is zero-dimensional and consists of four ammonia atoms. N(1) is bonded in a 1-coordinate geometry to  atoms."
#     assert N2.get_robocrys_rep() == excepted_output


def test_get_robocrys_rep() -> None:
    excepted_output = "SrTiO3 is (Cubic) Perovskite structured and crystallizes in the cubic Pm-3m space group. Sr(1)2+ is bonded to twelve equivalent O(1)2- atoms to form SrO12 cuboctahedra that share corners with twelve equivalent Sr(1)O12 cuboctahedra, faces with six equivalent Sr(1)O12 cuboctahedra, and faces with eight equivalent Ti(1)O6 octahedra. All Sr(1)-O(1) bond lengths are 2.77 Å. Ti(1)4+ is bonded to six equivalent O(1)2- atoms to form TiO6 octahedra that share corners with six equivalent Ti(1)O6 octahedra and faces with eight equivalent Sr(1)O12 cuboctahedra. The corner-sharing octahedra are not tilted. All Ti(1)-O(1) bond lengths are 1.96 Å. O(1)2- is bonded in a distorted linear geometry to four equivalent Sr(1)2+ and two equivalent Ti(1)4+ atoms."
    assert srtio3_p1.get_robocrys_rep() == excepted_output


def test_get_local_env_rep() -> None:
    expected_output = """Pm-3m
Sr2+ (1a) [O][Sr][O].[O].[O].[O].[O].[O].[O].[O].[O].[O].[O]
Ti4+ (1b) [O][Ti]([O])([O])([O])([O])[O]
O2- (3c) [Ti]O[Ti]"""
    assert srtio3_p1.get_local_env_rep() == expected_output
