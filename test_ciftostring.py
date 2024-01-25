import pytest
from ciftostring import CifToString


N2 = CifToString.from_file("N2.cif")


def test_get_cif_string() -> None:
    output = "data_N2\n_symmetry_space_group_name_H-M   'P 1'\n_cell_length_a   5.605\n_cell_length_b   5.605\n_cell_length_c   5.605\n_cell_angle_alpha   90.0\n_cell_angle_beta   90.0\n_cell_angle_gamma   90.0\n_symmetry_Int_Tables_number   1\n_chemical_formula_structural   N2\n_chemical_formula_sum   N4\n_cell_volume   176.125\n_cell_formula_units_Z   2\nloop_\n _symmetry_equiv_pos_site_id\n _symmetry_equiv_pos_as_xyz\n  1  'x, y, z'\nloop_\n _atom_type_symbol\n _atom_type_oxidation_number\n  N0+  0.0\nloop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_symmetry_multiplicity\n _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n _atom_site_occupancy\n  N0+  N0  1  0.477  0.977  0.523  1.0\n  N0+  N1  1  0.977  0.523  0.477  1.0\n  N0+  N2  1  0.023  0.023  0.023  1.0\n  N0+  N3  1  0.523  0.477  0.977  1.0\n"
    assert N2.get_cif_string() == output


def test_get_parameters() -> None:
    output = ['5.605', '5.605', '5.605', '90.0', '90.0', '90.0']
    assert N2.get_parameters() == output


def test_get_cartesian() -> None:
    output = ('5.605', '5.605', '5.605', '90.0', '90.0', '90.0', 'N', '2.673', '5.475', '2.933', 'N', '5.475', '2.933', '2.673', 'N', '0.13', '0.13', '0.13', 'N', '2.933', '2.673', '5.475')
    assert N2.get_cartesian() == output


def test_get_fractional() -> None:
    output = ('5.605', '5.605', '5.605', '90.0', '90.0', '90.0', 'N', '0.477', '0.977', '0.523', 'N', '0.977', '0.523', '0.477', 'N', '0.023', '0.023', '0.023', 'N', '0.523', '0.477', '0.977')
    assert N2.get_fractional() == output
