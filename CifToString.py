from pymatgen.core import Lattice, Structure, Molecule


class CifToString:
    def __init__(self, filepath):
        self.structure = Structure.from_file(filepath)

    def get_cif_string(self):
        return "\n".join(self.structure.to(fmt="cif").split("\n")[1:])
    
    def get_tuple_cartesian(self):
        a, b, c, alpha, beta, gamma = [str(round(i, 3)) for i in self.structure.lattice.parameters]

        elements = list()

        for site in self.structure.sites:
            elements.append(str(site.specie))
            coord = [str(x) for x in site.coords.round(3)]
            for el in coord:
                elements.append(el)

        return a, b, c, alpha, beta, gamma, *elements

    def get_tuple_fractional(self):
        a, b, c, alpha, beta, gamma = [str(round(i, 3)) for i in self.structure.lattice.parameters]

        elements = list()

        for site in self.structure.sites:
            elements.append(str(site.specie))
            coord = [str(x) for x in site.frac_coords.round(3)]
            for el in coord:
                elements.append(el)

        return a, b, c, alpha, beta, gamma, *elements

    
# sample1 = "TlCr5Se8.cif"
# s1 = CifToString(sample1)
# print(s1.get_tuple_cartesian())
# print(s1.get_tuple_fractional())
