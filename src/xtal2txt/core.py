import re
from typing import List, Tuple

from pymatgen.core import Lattice, Structure


class Textrep:
    """
    Generate text representations of crystal structure from pymatgen structure object.

    Attributes:
        structure : pymatgen structure

    Methods:
        from_file : a classmethod
        get_cif_string(n=3)
        get_parameters(n=3)
        get_coords(name, n=3)
        get_cartesian(n=3)
        get_fractional(n=3)
    """

    def __init__(self, structure: Structure):
        self.structure = structure

    @classmethod
    def from_file(cls, filepath: str) -> "Textrep":
        """
        Read cif files as pymatgen structure object. Instantiate the class with the structure object.

        Parameters:
            filepath : cif file of a crystal structure.

        Returns:
            Textrep
        """
        structure = Structure.from_file(filepath)
        return cls(structure)

    @staticmethod
    def round_numbers_in_string(original_string: str, decimal_places: int) -> str:
        """
        Rounds float numbers in the given string to the specified number of decimal places.

        Parameters:
            original_string : str, the input string containing float numbers.
            decimal_places : int, the number of decimal places to round float numbers.

        Returns:
            A new string with rounded float numbers.
        """
        pattern = r"[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)"
        matches = re.findall(pattern, original_string)
        rounded_numbers = [round(float(match), decimal_places) for match in matches]
        new_string = re.sub(pattern, lambda x: str(rounded_numbers.pop(0)), original_string)
        return new_string

    def get_cif_string(self, decimal_places: int = 3) -> str:
        """
        Generate CIF as string in multi-line format.

        All float numbers can be rounded to the specified number (decimal_places).
        Using a Regex pattern to do this on the whole string as we only want to round the float numbers, not integers.
        The Regex pattern detects any float number (negative or positive) in any display method (scientific, normal, ...).

        Parameters:
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A multi-line string
        """
        original_string = "\n".join(self.structure.to(fmt="cif").split("\n")[1:])
        return self.round_numbers_in_string(original_string, decimal_places)

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

    def get_cartesian(self, decimal_places: int = 3) -> Tuple[str, ...]:
        """
        Return the lattice parameters with atoms of the unit cell and their position in Cartesian coordinate with optional rounding.

        Parameters:
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A tuple of strings with the mentioned properties.
        """
        parameters = self.get_lattice_parameters(decimal_places)
        elements = self.get_coords("cartesian", decimal_places)
        return tuple(parameters + elements)

    def get_fractional(self, decimal_places: int = 3) -> Tuple[str, ...]:
        """
        Returning the lattice parameters with the particles of the unit cell and their position in fractional coordinate with optional rounding.

        Parameters:
            decimal_places : int, optional, to specify the rounding digit for float numbers.
                            Defaults to 3

        Returns:
            A tuple of strings with the mentioned properties.
        """
        parameters = self.get_lattice_parameters(decimal_places)
        elements = self.get_coords("fractional", decimal_places)
        return tuple(parameters + elements)

    def get_crystaltuple(
        self, name: str, param_decimal_places: int = 1, coord_decimal_places: int = 2
    ) -> str:
        """
        https://openreview.net/pdf?id=0r5DE2ZSwJ
        Returns the representation as per the above citation,  lattice length( the lengths with one decimal place), angles (as integers), atoms and their coordinates with line breaks...
        Fractional coordinates are always represented with two digits
        3D coordinates are combined with spaces and all other crystal components are combined with newlines
        in the line after that, then the element following with its Cartesian and fractional...
        coordinates as floats in a separate line.

        Parameters:
            name: str
                To specify the type of coordinate direction.
            param_decimal_places: int
                The decimal point to round the lattice parameters.
                Defaults to 1.
            coord_decimal_places: int
                The decimal point to show the coordinate digits.
                Defaults to 2.

        Returns:
            A multi-line string with the mentioned properties of the unit cell.
        """
        parameters = self.get_lattice_parameters(param_decimal_places)
        angles = [str(int(float(angle))) for angle in parameters[3:6]]
        output = " ".join(parameters[:3]) + "\n" + " ".join(angles)
        elements = self.get_coords(name, coord_decimal_places)
        for i in range(0, len(elements), 4):
            output += "\n" + elements[i] + "\n" + " ".join(elements[i + 1 : i + 4])
        return output

    def get_slice():
        pass

    def get_wycryst():
        pass
