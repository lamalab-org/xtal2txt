from pymatgen.core import Lattice, Structure, Molecule


def cif_to_string(pathfile): 
    """
    Convert the content of a cif file to a multi-line string with removing the first line.
    """

    structure = Structure.from_file(pathfile)

    return "\n".join(structure.to(fmt="cif").split("\n")[1:])


# sample1 = "TlCr5Se8.cif"


# print(cif_to_string(sample1))


