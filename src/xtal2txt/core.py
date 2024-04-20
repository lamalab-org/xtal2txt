import random
import re
from collections import Counter
from pathlib import Path
from typing import List, Union, Tuple

import numpy as np
from invcryrep.invcryrep import InvCryRep
from pymatgen.core import Structure
from pymatgen.core.structure import Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from robocrys import StructureCondenser, StructureDescriber

from xtal2txt.transforms import TransformationCallback


class TextRep:
    """
    Generate text representations of crystal structure for Language modelling.

    Attributes:
        structure : pymatgen structure

    Methods:
        from_input : a classmethod
        get_cif_string(n=3)
        get_parameters(n=3)
        get_coords(name, n=3)
        get_cartesian(n=3)
        get_fractional(n=3)
    """

    backend = InvCryRep()
    condenser = StructureCondenser()
    describer = StructureDescriber()

    def __init__(
        self,
        structure: Structure,
        transformations: List[Tuple[str, dict]] = None,
    ) -> None:
        self.structure = structure
        self.transformations = transformations or []
        self.apply_transformations()

    @classmethod
    def from_input(
        cls,
        input_data: Union[str, Path, Structure],
        transformations: List[Tuple[str, dict]] = None,
    ) -> "TextRep":
        """
        Instantiate the TextRep class object with the pymatgen structure from a cif file, a cif string, or a pymatgen Structure object.

        Parameters:
            input_data : cif file of a crystal structure, a cif string, or a pymatgen Structure object.

        Returns:
            TextRep
        """
        if isinstance(input_data, Structure):
            structure = input_data

        elif isinstance(input_data, (str, Path)):
            try:
                if Path(input_data).is_file():
                    structure = Structure.from_file(str(input_data))
                else:
                    raise ValueError
            except (OSError, ValueError):
                structure = Structure.from_str(str(input_data), "cif")

        else:
            structure = Structure.from_str(str(input_data), "cif")

        return cls(structure, transformations)
    
    def apply_transformations(self) -> None:
        """
        Apply transformations to the structure.
        """
        for transformation, params in self.transformations:
            transform_func = getattr(TransformationCallback, transformation)
            self.structure = transform_func(self.structure, **params)

    @staticmethod
    def _safe_call(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            return None

    @staticmethod
    def round_numbers_in_string(original_string: str, decimal_places: int) -> str:
        """
        Rounds float numbers in the given string to the specified number of decimal places using regex.

        Parameters:
            original_string : str, the input string containing float numbers.
            decimal_places : int, the number of decimal places to round float numbers.

        Returns:
            A new string with rounded float numbers.
        """
        pattern = r"\b\d+\.\d+\b"
        matches = re.findall(pattern, original_string)
        rounded_numbers = [round(float(match), decimal_places) for match in matches]
        new_string = re.sub(
            pattern, lambda x: str(rounded_numbers.pop(0)), original_string
        )
        return new_string

    def get_cif_string(
        self, format: str = "symmetrized", decimal_places: int = 3
    ) -> str:
        """
        Generate CIF as string in multi-line format.

        All float numbers can be rounded to the specified number (decimal_places).
        Currently supports two formats. Symmetrized (cif with symmetry operations and the least symmetric basis) ...
        and P1 (conventional unit cell , with all the atoms listed and only identity as symmetry operation).
        TODO: cif format with bonding blocks


        Parameters:
            format : str, optional, to specify the format of the cif string. Defaults to "symmetrized".
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A multi-line string representation of CIF.
        """

        if format == "symmetrized":
            symmetry_analyzer = SpacegroupAnalyzer(self.structure)
            symmetrized_structure = symmetry_analyzer.get_symmetrized_structure()
            cif_string = str(
                CifWriter(
                    symmetrized_structure,
                    symprec=0.1,
                    significant_figures=decimal_places,
                ).ciffile
            )
            cif = "\n".join(cif_string.split("\n")[1:])
            return self.round_numbers_in_string(cif, decimal_places)

        elif format == "p1":
            cif_string = "\n".join(self.structure.to(fmt="cif").split("\n")[1:])
            return self.round_numbers_in_string(cif_string, decimal_places)

    def get_lattice_parameters(self, decimal_places: int = 3) -> List[str]:
        """
        Return lattice parameters of unit cells in a crystal lattice:
        the lengths of the cell edges (a, b, and c) in angstrom and the angles between them (alpha, beta, and gamma) in degrees.

        All float numbers can be rounded to a specific number (decimal_places).

        Parameters:
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A list of strings of the mentioned parameters.
        """
        return [
            str(round(i, decimal_places)) for i in self.structure.lattice.parameters
        ]

    def get_coords(self, name: str = "cartesian", decimal_places: int = 3) -> List[str]:
        """
        Return list of atoms in unit cell for with their positions in Cartesian or fractional coordinates as per choice.

        Parameters:
            name : str
                Specifies the name of the coordinate system to extract the positions of the particles. default is "cartesian".
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A list of atoms with their positions inside the unit cell.
        """
        elements = []
        for site in self.structure.sites:
            elements.append(str(site.specie))
            coord = [
                str(x)
                for x in (
                    site.coords.round(decimal_places)
                    if name == "cartesian"
                    else site.frac_coords.round(decimal_places)
                )
            ]
            elements.extend(coord)
        return elements

    def get_slice(self, primitive: bool = True) -> str:
        """Returns SLICE representation of the crystal structure.
        https://www.nature.com/articles/s41467-023-42870-7

        Parameters:
            primitive : bool, optional, to specify if the primitive structure is required. Defaults to True.

        Returns:
            str: The calculated slice.
        """

        if primitive:
            primitive_structure = (
                self.structure.get_primitive_structure()
            )  # convert to primitive structure
            return self.backend.structure2SLICES(primitive_structure)
        return self.backend.structure2SLICES(self.structure)

    def get_composition(self, format="hill") -> str:
        """Return composition in hill format.

        Args:
            format (str): format in which the composition is required.

        Returns:
            str: The composition in hill format.
        """
        if format == "hill":
            composition_string = self.structure.composition.hill_formula
            composition = composition_string.replace(" ", "")
        return composition

    def get_crystal_llm_rep(
        self,
        permute_atoms: bool = False,
        translate_atoms: bool = False,
    ) -> str:
        """
        Code adopted from https://github.com/facebookresearch/crystal-llm/blob/main/llama_finetune.py
        https://openreview.net/pdf?id=0r5DE2ZSwJ
        TODO: kwargs and customizable parameters
        TODO: Rounding parameters user defined
        TODO: fractional/ caartesian optional
        TODO: Translation of the structure optional
        Returns the representation as per the above citation,  lattice length( the lengths with one decimal place), angles (as integers), atoms and their coordinates with line breaks...
        Fractional coordinates are always represented with two digits
        3D coordinates are combined with spaces and all other crystal components are combined with newlines
        in the line after that, then the element following with its Cartesian and fractional...
        coordinates as floats in a separate line.

        """

        lengths = self.structure.lattice.parameters[:3]
        angles = self.structure.lattice.parameters[3:]
        atom_ids = self.structure.species
        frac_coords = self.structure.frac_coords

        if permute_atoms:
            atom_coord_pairs = list(zip(atom_ids, frac_coords))
            random.shuffle(atom_coord_pairs)
            atom_ids, frac_coords = zip(*atom_coord_pairs)

        crystal_str = (
            " ".join(["{0:.1f}".format(x) for x in lengths])
            + "\n"
            + " ".join([str(int(x)) for x in angles])
            + "\n"
            + "\n".join(
                [
                    str(t) + "\n" + " ".join(["{0:.2f}".format(x) for x in c])
                    for t, c in zip(atom_ids, frac_coords)
                ]
            )
        )

        return crystal_str

    def get_robocrys_rep(self):
        """
        https://github.com/hackingmaterials/robocrystallographer/tree/main
        TODO: pinned  matminer to 0.9.1.dev14 (check if can be relaxed ?)
        TODO: check any post processing for better tokenization (rounding, replacing unicodes etc..)

        """

        condensed_structure = self.condenser.condense_structure(self.structure)
        return self.describer.describe(condensed_structure)

    def get_wyckoff_positions(self):
        """
        Getting wyckoff positions of the elements in the unit cell as the combination of...
        number and letter.

        Returns:
            ouput: str
                A multi-line string that contain elements of the unit cell along with their...
                wyckoff position in each line.
                Hint: At the end of the string, there is an additional newline character.
        """

        spacegroup_analyzer = SpacegroupAnalyzer(self.structure)
        wyckoff_sites = spacegroup_analyzer.get_symmetry_dataset()
        element_symbols = [site.specie.element.symbol for site in self.structure.sites]

        data = []

        for i in range(len(wyckoff_sites["wyckoffs"])):
            sub_data = (
                element_symbols[i],
                wyckoff_sites["wyckoffs"][i],
                wyckoff_sites["equivalent_atoms"][i],
            )
            data.append(sub_data)

        a = dict(Counter(data))

        output = ""
        for i, j in a.items():
            output += str(i[0]) + " " + str(j) + " " + str(i[1]) + "\n"

        return output

    def get_wycryst(self):
        """
        Obtaining the wyckoff representation for crystal structures that include:
            chemcial formula
            space group number
            elements of the unit cell with their wyckoff positions.

        Returns:
            output: str
                A multi-line string, which contain mentioned properties of the crystal...
                structure in each separate line.
        """
        output = ""
        chemical_formula = self.structure.composition.formula
        output += chemical_formula
        output += "\n" + str(self.structure.get_space_group_info()[1])
        output += "\n" + self.get_wyckoff_positions()

        return output

    

    def get_atoms_params_rep(
        self, lattice_params: bool = False, decimal_places: int = 1
    ) -> str:
        """
        Generating a string with the elements of composition inside the crystal lattice with the option to
        get the lattice parameters as angles (int) and lengths (float) in a string with a space
        between them

        Params:
            lattice_params: boolean, optional
                To specify whether use lattice parameters in generating crystal structure.
                Defaults to False
            decimal_places : int, optional,
                to specify the rounding digit for float numbers.
                Defaults to 2

        Returns:
            output: str
                An oneline string.
        """

        try:
            output = [site.specie.element.symbol for site in self.structure.sites]
        except AttributeError:
            output = [site.specie.symbol for site in self.structure.sites]
        if lattice_params:
            params = self.get_lattice_parameters(decimal_places=decimal_places)
            params[3:] = [str(int(float(i))) for i in params[3:]]
            output.extend(params)

        return " ".join(output)

    def updated_zmatrix_rep(self, zmatrix, decimal_places=1):
        lines = zmatrix.split("\n")
        main_part = []
        variables_part = []

        # Determine the main part and the variables part of the Z-matrix
        for line in lines:
            if "=" in line:
                variables_part.append(line)
            else:
                if line.strip():  # Skip empty lines
                    main_part.append(line)

        # Extract variables from the variables part
        variable_dict = {}
        for var_line in variables_part:
            var, value = var_line.split("=")
            if var.startswith("B"):
                rounded_value = round(float(value.strip()), decimal_places)
            else:
                rounded_value = int(round(float(value.strip())))
            variable_dict[var] = (
                f"{rounded_value}"
                if var.startswith(("A", "D"))
                else f"{rounded_value:.{decimal_places}f}"
            )

        # Replace variables in the main part
        replaced_lines = []
        for line in main_part:
            parts = line.split()
            # atom = parts[0]
            replaced_line = line
            for i in range(1, len(parts)):
                var = parts[i]
                if var in variable_dict:
                    replaced_line = replaced_line.replace(var, variable_dict[var])
            replaced_lines.append(replaced_line)

        return "\n".join(replaced_lines)

    def get_zmatrix_rep(self, decimal_places=1):
        species = [s.element for s in self.structure.species]
        coords = [c for c in self.structure.cart_coords]
        molecule_ = Molecule(
            species,
            coords,
        )
        zmatrix = molecule_.get_zmatrix()
        return self.updated_zmatrix_rep(zmatrix, decimal_places)

    def get_all_text_reps(self, decimal_places: int = 2):
        """
        Returns all the Text representations of the crystal structure in a dictionary.
        """

        return {
            "cif_p1": self._safe_call(
                self.get_cif_string, format="p1", decimal_places=decimal_places
            ),
            "cif_symmetrized": self._safe_call(
                self.get_cif_string, format="symmetrized", decimal_places=decimal_places
            ),
            "cif_bonding": None,
            "slice": self._safe_call(self.get_slice),
            "composition": self._safe_call(self.get_composition),
            "crystal_llm_rep": self._safe_call(self.get_crystal_llm_rep),
            "robocrys_rep": self._safe_call(self.get_robocrys_rep),
            "wycoff_rep": None,
            "atoms": self._safe_call(
                self.get_atoms_params_rep,
                lattice_params=False,
                decimal_places=decimal_places,
            ),
            "atoms_params": self._safe_call(
                self.get_atoms_params_rep,
                lattice_params=True,
                decimal_places=decimal_places,
            ),
            "zmatrix": self._safe_call(self.get_zmatrix_rep),
        }

    def get_requested_text_reps(
        self, requested_reps: List[str], decimal_places: int = 2
    ):
        """
        Returns the requested Text representations of the crystal structure in a dictionary.

        Parameters:
            requested_reps : List of representations to return.
            decimal_places : Number of decimal places for the cif strings.

        Returns:
            Dictionary of requested representations.
        """

        all_reps = {
            "cif_p1": lambda: self._safe_call(
                self.get_cif_string, format="p1", decimal_places=decimal_places
            ),
            "cif_symmetrized": lambda: self._safe_call(
                self.get_cif_string, format="symmetrized", decimal_places=decimal_places
            ),
            "cif_bonding": lambda: None,
            "slice": lambda: self._safe_call(self.get_slice),
            "composition": lambda: self._safe_call(self.get_composition),
            "crystal_llm_rep": lambda: self._safe_call(self.get_crystal_llm_rep),
            "robocrys_rep": lambda: self._safe_call(self.get_robocrys_rep),
            "wycoff_rep": lambda: None,
            "atoms": lambda: self._safe_call(
                self.get_atoms_params_rep,
                lattice_params=False,
                decimal_places=decimal_places,
            ),
            "atoms_params": lambda: self._safe_call(
                self.get_atoms_params_rep,
                lattice_params=True,
                decimal_places=decimal_places,
            ),
            "zmatrix": lambda: self._safe_call(self.get_zmatrix_rep, decimal_places=1),
        }

        return {rep: all_reps[rep]() for rep in requested_reps if rep in all_reps}
