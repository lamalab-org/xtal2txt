
ANALYSIS_MASK_TOKENS = {
    "atoms"         :  "[ATOMS]",
    "directions"    :  "[DIR]",
    "numbers"       :  "[NUMS]",
    "bonds"         :  "[BONDS]",
    "miscellaneous" :  "[MISC]",
    "identifier"    :  "[ID]",
    "symmetry"      :  "[SYM]",
    "lattice"       :  "[LATTICE]",
    "composition"   :  "[COMP]",
    None            :  "[NONE]"
}

ATOM_LIST_ = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", 
        "S", "Cl", "K", "Ar", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Ni", "Co", "Cu", "Zn", 
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", 
        "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", 
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", 
        "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", 
        "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", 
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]

NUMS_ = [
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "+", "-"
    ]

SLICE_ANALYSIS_DICT = {
    "atoms": ATOM_LIST_,
    "directions": [
        "o o o", "o o +", "o o -", "o + o", "o + +", "o + -", "o - o", "o - +", "o - -",
        "+ o o", "+ o +", "+ o -", "+ + o", "+ + +", "+ + -", "+ - o", "+ - +", "+ - -", 
        "- o o", "- o +", "- o -", "- + o", "- + +", "- + -", "- - o", "- - +", "- - -"
    ],
    "numbers": NUMS_
}


CRYSTAL_LLM_ANALYSIS_DICT = {
    "atoms": ATOM_LIST_,
    "numbers": NUMS_,
    "miscellaneous": ["\n", " "]
}

CIF_ANALYSIS_DICT = {
    "atoms": ATOM_LIST_,
    "numbers": NUMS_,
    "lattice": ["_cell_length_a", "_cell_length_b", "_cell_length_c",
                "_cell_angle_alpha", "_cell_angle_beta", "_cell_angle_gamma" ],
    "identifier": ["loop_"],
    "composition": [
                    "_chemical_formula_structural",
                    "_chemical_formula_sum"],
    "symmetry": ["_symmetry_space_group_name_H-M", "_symmetry_Int_Tables_number", 
                "_symmetry_equiv_pos_site_id","_symmetry_equiv_pos_as_xyz",],
    "miscellaneous": ["_atom_site_symmetry_multiplicity" ,
                    "_atom_type_symbol",
                    "_atom_type_oxidation_number",
                    "_atom_site_type_symbol",
                    "_atom_site_label",
                    "_atom_site_symmetry_multiplicity",
                    "_atom_site_fract_x",
                    "_atom_site_fract_y",
                    "_atom_site_fract_z",
                    "_atom_site_occupancy",
                    "-", "/", "'", "\"", ",", "'x, y, z'", "x", "y", "z", "-x", "-y", "-z", "  ", "   ", "\n", "_geom_bond_atom_site_label_1", "_geom_bond_atom_site_label_2", "_geom_bond_distance", "_ccdc_geom_bond_type", "_", "a", "n", "c", "b", "m", "d", "R", "A", "(", ")", "[", "]", "*"
                    ],}




