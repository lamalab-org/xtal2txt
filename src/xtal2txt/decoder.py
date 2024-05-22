from pyxtal import pyxtal
from pyxtal.lattice import Lattice as pyLattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice


class DecodeTextRep:
    def __init__(self, text):
        self.text = text

    def decode(self):
        return self.text

    def wyckoff_decoder(self, input: str, lattice_params: bool = False):
        """
        Generating a pymatgen object from the output of the get_wyckoff_rep() method by using...
        pyxtal package. In this method, all data are extracted from the multi-line string of the...
        mentioned method.
        In pyxtal package, a 3D crystal is produced by specifying the dimensions, elements,
        composition of elements, space group, and sites as wyckoff positions of the elements.

        Params:
            lattice_params: boolean
                To specify whether use lattice parameters in generating crystal structure.

        Returns:
            pmg_struc: pymatgen.core.structure.Structure
        """

        # Always dimension is 3.
        dimensions = 3

        entities = input.split("\n")[:-1]
        elements = entities[0]
        spg = int(entities[1])
        wyckoff_sites = entities[2:]
        elements = elements.split(" ")

        atoms = []
        composition = []
        for el in elements:
            atom = el.rstrip("0123456789")
            number = el[len(atom) :]
            atoms.append(atom)
            composition.append(int(number))

        sites = []
        for atom in atoms:
            sub_site = []
            for site in wyckoff_sites:
                if atom in site:
                    sub_site.append(site.split()[1])

            sites.append(sub_site)

        xtal_struc = pyxtal()

        if lattice_params:
            a, b, c, alpha, beta, gamma = self.get_lattice_parameters()
            cell = pyLattice.from_para(
                float(a), float(b), float(c), float(alpha), float(beta), float(gamma)
            )
            xtal_struc.from_random(dimensions, spg, atoms, composition, sites=sites, lattice=cell)
        else:
            xtal_struc.from_random(dimensions, spg, atoms, composition, sites=sites)

        pmg_struc = xtal_struc.to_pymatgen()

        return pmg_struc

    def llm_decoder(self, input: str):
        """
        Returning pymatgen structure out of multi-line representation.

        Params:
            input: str
                String to obtain the items needed for the structure.

        Returns:
            pymatgen.core.structure.Structure
        """
        entities = input.split("\n")
        lengths = entities[0].split(" ")
        angles = entities[1].split(" ")
        lattice = Lattice.from_parameters(
            a=float(lengths[0]),
            b=float(lengths[1]),
            c=float(lengths[2]),
            alpha=float(angles[0]),
            beta=float(angles[1]),
            gamma=float(angles[2]),
        )

        elements = entities[2::2]
        coordinates = entities[3::2]
        m_coord = []
        for i in coordinates:
            s = [float(j) for j in i.split(" ")]
            m_coord.append(s)

        return Structure(lattice, elements, m_coord)

    def cif_string_decoder_p1(self, input: str):
        """
        Returning a pymatgen structure out of a string format of a cif file.

        Params:
            input: str
                String to obtain the items needed for the structure.

        Returns:
            pymatgen.core.structure.Structure
        """
        entities = input.split("\n")[:-1]

        params = []
        for i in range(2, 8):
            params.append(entities[i].split("   ")[1])

        lattice = Lattice.from_parameters(
            a=float(params[0]),
            b=float(params[1]),
            c=float(params[2]),
            alpha=float(params[3]),
            beta=float(params[4]),
            gamma=float(params[5]),
        )

        elements = []
        m_coord = []
        atoms = entities[entities.index(" _atom_site_occupancy") + 1 :]
        for atom in atoms:
            ls = atom.split("  ")
            elements.append(ls[1])
            m_coord.append([float(ls[4]), float(ls[5]), float(ls[6])])

        return Structure(lattice, elements, m_coord)

    def cif_string_decoder_sym(self, input: str):
        """
        Returning a pymatgen structure out of a string format of a symmetrized cif file.

        Params:
            input: str
                String to obtain the items needed for the structure.

        Returns:
            pymatgen.core.structure.Structure
        """
        entities = input.split("\n")[:-1]

        params = []
        for i in range(1, 8):
            params.append(entities[i].split("   ")[1])

        spg = params[0]
        params = params[1:]
        lattice = Lattice.from_parameters(
            a=float(params[0]),
            b=float(params[1]),
            c=float(params[2]),
            alpha=float(params[3]),
            beta=float(params[4]),
            gamma=float(params[5]),
        )

        elements = []
        m_coord = []
        atoms = entities[entities.index(" _atom_site_occupancy") + 1 :]
        for atom in atoms:
            ls = atom.split("  ")
            elements.append(ls[1])
            m_coord.append([float(ls[4]), float(ls[5]), float(ls[6])])

        # print(atoms)

        return Structure.from_spacegroup(spg, lattice, elements, m_coord)


class MatchRep:
    def __init__(self, textrep, structure):
        self.text = textrep
        self.structure = structure

    def wyckoff_matcher(
        self,
        ltol=0.2,
        stol=0.5,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        allow_subset=True,
        attempt_supercell=True,
        lattice_params: bool = False,
    ):
        """
        To check if pymatgen object from the original cif file match with the generated...
        pymatgen structure from wyckoff_decoder method out of wyckoff representation...
        using fit() method of StructureMatcher module in pymatgen package.

        Params:
            StructureMatcher module can be access in below link with its parameters:
                https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.get_mapping
            lattice_params: bool
                To specify using lattice parameters in the wyckoff_decoder method.

        Returns:
            StructureMatcher().fit_anonymous(): bool
        """

        original_struct = self.structure

        # output_struct = self.wyckoff_decoder(input, lattice_params)
        output_struct = DecodeTextRep(self.text).wyckoff_decoder(self.text, lattice_params=True)

        return StructureMatcher(
            ltol,
            stol,
            angle_tol,
            primitive_cell,
            scale,
            allow_subset,
            attempt_supercell,
        ).fit_anonymous(output_struct, original_struct)

    def llm_matcher(
        self,
        ltol=0.2,
        stol=0.5,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        allow_subset=True,
        attempt_supercell=True,
    ):
        """
        To check if pymatgen object from the original cif file match with the generated
        pymatgen structure from llm_decoder method out of llm representation
        using fit() method of StructureMatcher module in pymatgen package.

        Params:
            input: str
                String to obtain the items needed for the structure.

            StructureMatcher module can be access in below link with its parameters:
                https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.get_mapping

        Returns:
            StructureMatcher().fit(): bool
        """

        original_struct = self.structure
        output_struct = DecodeTextRep(self.text).llm_decoder(self.text)

        return StructureMatcher(
            ltol,
            stol,
            angle_tol,
            primitive_cell,
            scale,
            allow_subset,
            attempt_supercell,
        ).fit(output_struct, original_struct)

    def cif_string_matcher_sym(
        self,
        #        input: str,
        ltol=0.2,
        stol=0.5,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        allow_subset=True,
        attempt_supercell=True,
    ):
        """
        To check if pymatgen object from the original cif file match with the generated
        pymatgen structure from cif_string_decoder_sym method out of string cif representation.
        using fit() method of StructureMatcher module in pymatgen package.

        Params:
            input: str
                String to obtain the items needed for the structure.

            StructureMatcher module can be access in below link with its parameters:
                https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.get_mapping

        Returns:
            StructureMatcher().fit(): bool
        """

        original_struct = self.structure
        output_struct = DecodeTextRep(self.text).cif_string_decoder_sym(self.text)

        return StructureMatcher(
            ltol,
            stol,
            angle_tol,
            primitive_cell,
            scale,
            allow_subset,
            attempt_supercell,
        ).fit(output_struct, original_struct)

    def cif_string_matcher_p1(
        self,
        ltol=0.2,
        stol=0.5,
        angle_tol=5,
        primitive_cell=True,
        scale=True,
        allow_subset=True,
        attempt_supercell=True,
    ):
        """
        To check if pymatgen object from the original cif file match with the generated
        pymatgen structure from cif_string_decoder_p1 method out of string cif representation
        using fit() method of StructureMatcher module in pymatgen package.

        Params:
            input: str
                String to obtain the items needed for the structure.

            StructureMatcher module can be access in below link with its parameters:
                https://pymatgen.org/pymatgen.analysis.html#pymatgen.analysis.structure_matcher.StructureMatcher.get_mapping

        Returns:
            StructureMatcher().fit(): bool
        """

        original_struct = self.structure
        output_struct = DecodeTextRep(self.text).cif_string_decoder_p1(self.text)

        return StructureMatcher(
            ltol,
            stol,
            angle_tol,
            primitive_cell,
            scale,
            allow_subset,
            attempt_supercell,
        ).fit(output_struct, original_struct)
