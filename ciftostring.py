from pymatgen.core import Lattice, Structure, Molecule
import re


class CifToString:
    """
    A class to get information from a CIF file for a crystal with using pymatgen library.

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
    def __init__(self, structure):
        self.structure = structure


    @classmethod
    def from_file(cls, filepath):
        '''
        Getting the information from a cif file and transform it to a pymatgen object and store it in an object of the class.

        Parameters:
            filepath : cif file
                containing the information of a crystal structure in .cif extension.
        
        Returns:
            pymatgen strucure
        '''
        structure = Structure.from_file(filepath)
        return cls(structure)


    def get_cif_string(self, n=3):
        """
        Return the contents of a CIF file as a multi-line string with...
        removing the first line of the file.

        All float numbers can be rounded to specified number (n).
        Using a Regex pattern to do this on the whole string as we only want to round the float numbers not integers.
        The Regex pattern detect any float number (negative or positive) in any display method (scientific, normal, ...).

        Parameters: 
            n : integer, optional, to specify the rounding digit for float numbers.
                default to 3

        Returns:
            A multi-line string
        """
        original_string = "\n".join(self.structure.to(fmt="cif").split("\n")[1:])

        pattern = r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?'

        matches = re.findall(pattern, original_string)
        rounded_numbers = [round(float(match), n) for match in matches]
        new_string = re.sub(pattern, lambda x: str(rounded_numbers.pop(0)), original_string)
        return new_string

    
    def get_parameters(self, n=3):
        """
        Returning 6 lattice parameters of unit cells in a crystal lattice: the lengths of the cell edges (a, b, and c) in angstrom and the angles between them (alpha, beta, and gamma) in degrees.

        All float numbers can be rounded to a specific number (n).

        Parameters:
            n : integer, optional, to specify the rounding   digit for float numbers.
                default to 3

        Returns:
            A list of strings of the mentioned parameters.
        """
        return [str(round(i, n)) for i in self.structure.lattice.parameters]


    def get_coords(self, name, n=3):
        """
        Returning the particles inside unit cells for a crystal lattice along with their positions...
        that can be described in cartesian or fractional coordinates.

        Parameters:
            name : string
                   specifies the name of coordinate to extract the positions of the particles.

        Returns:
            A list of the strings containing the particle with their positions inside the unit cell.
        """
        elements = list()
        for site in self.structure.sites:
            elements.append(str(site.specie))
            if name == "cartesian":
                coord = [str(x) for x in site.coords.round(n)]
                for el in coord:
                    elements.append(el)
            elif name == "fractional":
                coord = [str(x) for x in site.frac_coords.round(n)]
                for el in coord:
                    elements.append(el)
        return elements


    def get_cartesian(self, n=3) -> tuple:
        """
        Returning the lattice parameters with the particles of the unit cell and their position in cartesian coordinate.

        All float numbers can be rounded.

        Paramaters:
            n : integer, optional, to specify the rounding   digit for float numbers.
                default to 3

        Returns:
            A tuple of strings with the mentioned properties.
        """
        a, b, c, alpha, beta, gamma = self.get_parameters(n)
        elements = self.get_coords("cartesian", n)
        return a, b, c, alpha, beta, gamma, *elements


    def get_fractional(self, n=3) -> tuple:
        """
        Returning the lattice parameters with the particles of the unit cell and their position in fractional coordinate.

        All float numbers can be rounded.

        Paramaters:
            n : integer, optional, to specify the rounding   digit for float numbers.
                default to 3

        Returns:
            A tuple of strings with the mentioned properties.
        """
        a, b, c, alpha, beta, gamma = self.get_parameters(n)
        elements = self.get_coords("fractional", n)
        return a, b, c, alpha, beta, gamma, *elements

