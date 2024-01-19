from pymatgen.core import Lattice, Structure, Molecule
import re


class CifToString:

    def __init__(self, structure):
        self.structure = structure

    @classmethod
    def from_file(cls, filepath):
        structure = Structure.from_file(filepath)
        return cls(structure)

    def get_cif_string(self, n=3):
        original_string = "\n".join(self.structure.to(fmt="cif").split("\n")[1:])

        pattern = r'[+-]?(?=\d*[.eE])(?=\.?\d)\d*\.?\d*(?:[eE][+-]?\d+)?'

        def round_match(match):
            original_f = float(match.group(0))
            rounded_f = round(original_f, n)
            return str(rounded_f)
        
        new_string = re.sub(pattern, round_match, original_string)
        return new_string

    
    def get_parameters(self, n=3):
        return [str(round(i, n)) for i in self.structure.lattice.parameters]

    def get_coords(self, name, n=3):
        elements = list()
        for site in self.structure.sites:
            elements.append(str(site.specie.element.symbol))
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
        a, b, c, alpha, beta, gamma = self.get_parameters(n)
        elements = self.get_coords("cartesian", n)
        return a, b, c, alpha, beta, gamma, *elements

    def get_fractional(self, n=3) -> tuple:
        a, b, c, alpha, beta, gamma = self.get_parameters(n)
        elements = self.get_coords("fractional", n)
        return a, b, c, alpha, beta, gamma, *elements

s1 = "N2.cif"
N2 = CifToString.from_file(s1)
print(N2.get_cif_string())