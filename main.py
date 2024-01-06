from pymatgen.core import Lattice, Structure, Molecule


def cif_to_tuples(filepath) -> tuple:
    """
    Giving the content of a crystal in Crystallographic Information File format...
    and obtaining the output in a tuple of strings by using pymatgen package.
    
    All the digits are rounded to 3 decimal points.
    """
    structure = Structure.from_file(filepath)

    a = round(structure.lattice.a, 3)
    b = round(structure.lattice.b, 3)
    c = round(structure.lattice.c, 3)

    alpha = round(structure.lattice.angles[0], 3)
    beta = round(structure.lattice.angles[1], 3)
    gamma = round(structure.lattice.angles[2], 3)

    elements = list()

    for i in range(len(structure.species)):
        elements.append(str(structure.species[i]))
        for j in range(len(structure.cart_coords[i])):
            elements.append(str(round(structure.cart_coords[i][j], 3)))

    return str(a), str(b), str(c), str(alpha), str(beta), str(gamma), *elements


sample1 = "TlCr5Se8.cif"
sample2 = "Ba(PdS2)2.cif"
sample3 = "Cr5S8.cif"

print(cif_to_tuples(sample1))
print(cif_to_tuples(sample2))
print(cif_to_tuples(sample3))

