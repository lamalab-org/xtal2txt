import re
from typing import List, Union
from pathlib import Path

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
from invcryrep.invcryrep import InvCryRep
from robocrys import StructureCondenser, StructureDescriber


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

    def __init__(self, structure: Structure):
        self.structure = structure

    @classmethod
    def from_input(cls, input_data: Union[str, Path, Structure]) -> "TextRep":
        """
        Instantiate the TextRep class object with the pymatgen structure from a cif file, a cif string, or a pymatgen Structure object.

        Parameters:
            input_data : cif file of a crystal structure, a cif string, or a pymatgen Structure object.

        Returns:
            TextRep
        """
        if isinstance(input_data, Structure):
            structure = input_data

        elif isinstance(input_data, (str, Path)) and Path(input_data).is_file():
            structure = Structure.from_file(str(input_data))

        else:
            structure = Structure.from_str(str(input_data), "cif")
        return cls(structure)
    
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
        new_string = re.sub(pattern, lambda x: str(rounded_numbers.pop(0)), original_string)
        return new_string

    def get_cif_string(self, format: str = "symmetrized", decimal_places: int = 3) -> str:
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
                    symmetrized_structure, symprec=0.1, significant_figures=decimal_places
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
        return [str(round(i, decimal_places)) for i in self.structure.lattice.parameters]

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
        backend = InvCryRep()
        if primitive:
            primitive_structure = (
                self.structure.get_primitive_structure()
            )  # convert to primitive structure
            return backend.structure2SLICES(primitive_structure)
        return backend.structure2SLICES(self.structure)

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

    def get_crystal_llm_rep(self):
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
        # Randomly translate within the unit cell
        # self.structure.translate_sites(
        #     indices=range(len(self.structure.sites)), vector=np.random.uniform(size=(3,))
        # )

        lengths = self.structure.lattice.parameters[:3]
        angles = self.structure.lattice.parameters[3:]
        atom_ids = self.structure.species
        frac_coords = self.structure.frac_coords

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
        condenser = StructureCondenser()
        describer = StructureDescriber()

        condensed_structure = condenser.condense_structure(self.structure)
        return describer.describe(condensed_structure)


    def get_atom_params(self, lattice_params: bool = False, decimal_places: int = 2):
        """
        Generating a string with the elements of composition inside the crystal lattice with the option to...
        get the lattice parameters as angles (int) and lengths (float) in a string with a space...
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
        
        output = ""
        for site in self.structure.sites:
            output += site.specie.element.symbol + " "

        if lattice_params:
            params = self.get_lattice_parameters(decimal_places=decimal_places)
            params[3:] = [str(int(float(i))) for i in params[3:]]
            for i in [3, 4, 5, 0, 1, 2]:
                output += params[i] + " "

        return output[:-1]
    
    
    def get_all_text_reps(self, decimal_places: int = 2):
        """
        Returns all the Text representations of the crystal structure in a dictionary.
        """

        return {
            "cif_p1": self._safe_call(self.get_cif_string, format="p1", decimal_places=decimal_places),
            "cif_symmetrized": self._safe_call(self.get_cif_string, format="symmetrized", decimal_places=decimal_places),
            "cif_bonding": None,
            "slice": self._safe_call(self.get_slice),
            "composition": self._safe_call(self.get_composition),
            "crystal_llm_rep": self._safe_call(self.get_crystal_llm_rep),
            "robocrys_rep": self._safe_call(self.get_robocrys_rep),
            "wycoff_rep": None,
        }

