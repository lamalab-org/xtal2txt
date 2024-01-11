from pymatgen.core import Lattice, Structure, Molecule


def cif_to_tuple_cartesian(filepath) -> tuple:
    """
    Giving the content of a crystal in Crystallographic Information File format...
    and obtaining the output in a tuple of strings by using pymatgen package.
    The coordinates are cartesian.
    
    All the digits are rounded to 3 decimal points.
    """
    structure = Structure.from_file(filepath)

    a, b, c, alpha, beta, gamma = [str(round(i, 3)) for i in structure.lattice.parameters]

    elements = list()

    for site in structure.sites:
        elements.append(str(site.specie))
        coord = [str(x) for x in site.coords.round(3)]
        for el in coord:
            elements.append(el)

    return a, b, c, alpha, beta, gamma, *elements


# sample1 = "TlCr5Se8.cif"

# print(cif_to_tuple_cartesian(sample1))


