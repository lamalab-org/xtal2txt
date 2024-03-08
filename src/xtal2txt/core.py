import re
from typing import List, Union
from pathlib import Path
from collections import Counter

from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.cif import CifWriter
from invcryrep.invcryrep import InvCryRep
from robocrys import StructureCondenser, StructureDescriber
from pyxtal import pyxtal
from pymatgen.analysis.structure_matcher import StructureMatcher
from pyxtal.lattice import Lattice as pyLattice


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
            sub_data = element_symbols[i], wyckoff_sites["wyckoffs"][i], wyckoff_sites["equivalent_atoms"][i]
            data.append(sub_data)
        
        a = dict(Counter(data))

        output = ""
        for i, j in a.items():
            output += str(i[0]) + " " +  str(j) + str(i[1]) + "\n"

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

    
    def wyckoff_decoder(self, input: str, lattice_params: bool = False): 
        """
        Generating a pymatgen object from the output of the get_wyckoff_rep() method by using...
        pyxtal package. In this method, all data are extracted from the multi-line string of the...
        mentioned method.
        In pyxtal package, a 3D crystal is produced by specifying the dimensions, elements,...
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
            atom = el.rstrip('0123456789')
            number = el[len(atom):]
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
            cell = pyLattice.from_para(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
            xtal_struc.from_random(dimensions,
                                spg,
                                atoms,
                                composition,
                                sites=sites,
                                lattice=cell)
        else:
            xtal_struc.from_random(dimensions,
                                spg,
                                atoms,
                                composition,
                                sites=sites)

        pmg_struc = xtal_struc.to_pymatgen()

        return pmg_struc

    
    def wyckoff_matcher(self, input: str, ltol = 0.2, stol = 0.5, angle_tol = 5, primitive_cell = True, 
                    scale = True, allow_subset = True, attempt_supercell = True, lattice_params: bool = False):
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
        
        output_struct = self.wyckoff_decoder(input, lattice_params)

        return StructureMatcher(ltol, stol, angle_tol, primitive_cell, scale, allow_subset, attempt_supercell).fit_anonymous(output_struct, original_struct)


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
        lattice = Lattice.from_parameters(a=float(lengths[0]),
                                        b=float(lengths[1]),
                                        c=float(lengths[2]),
                                        alpha=float(angles[0]),
                                        beta=float(angles[1]),
                                        gamma=float(angles[2]))
        
        elements = entities[2::2]
        coordinates = entities[3::2]
        m_coord = []
        for i in coordinates:
            s = [float(j) for j in i.split(" ")]
            m_coord.append(s)

        return Structure(lattice, elements, m_coord)


    def llm_matcher(self, input: str, ltol = 0.2, stol = 0.5, angle_tol = 5, primitive_cell = True, 
                    scale = True, allow_subset = True, attempt_supercell = True):
        """
        To check if pymatgen object from the original cif file match with the generated...
        pymatgen structure from llm_decoder method out of llm representation...
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
        
        output_struct = self.llm_decoder(input)

        return StructureMatcher(ltol, stol, angle_tol, primitive_cell, scale, allow_subset, attempt_supercell).fit(output_struct, original_struct)


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

        lattice = Lattice.from_parameters(a=float(params[0]),
                                        b=float(params[1]),
                                        c=float(params[2]),
                                        alpha=float(params[3]),
                                        beta=float(params[4]),
                                        gamma=float(params[5]))
        
        elements = []
        m_coord = []
        atoms = entities[entities.index(' _atom_site_occupancy') + 1:]
        for atom in atoms:
            ls = atom.split("  ")
            elements.append(ls[1])
            m_coord.append([float(ls[4]), float(ls[5]), float(ls[6])])
        
        return Structure(lattice, elements, m_coord)

    
    def cif_string_matcher_p1(self, input: str, ltol = 0.2, stol = 0.5, angle_tol = 5, primitive_cell = True, 
                    scale = True, allow_subset = True, attempt_supercell = True):
        """
        To check if pymatgen object from the original cif file match with the generated...
        pymatgen structure from cif_string_decoder_p1 method out of string cif representation...
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
        
        output_struct = self.cif_string_decoder_p1(input)

        return StructureMatcher(ltol, stol, angle_tol, primitive_cell, scale, allow_subset, attempt_supercell).fit(output_struct, original_struct)

    
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
        lattice = Lattice.from_parameters(a=float(params[0]),
                                        b=float(params[1]),
                                        c=float(params[2]),
                                        alpha=float(params[3]),
                                        beta=float(params[4]),
                                        gamma=float(params[5]))

        elements = []
        m_coord = []
        atoms = entities[entities.index(' _atom_site_occupancy') + 1:]
        for atom in atoms:
            ls = atom.split("  ")
            elements.append(ls[1])
            m_coord.append([float(ls[4]), float(ls[5]), float(ls[6])])

        # print(atoms)

        return Structure.from_spacegroup(spg, lattice, elements, m_coord)


    def cif_string_matcher_sym(self, input: str, ltol = 0.2, stol = 0.5, angle_tol = 5, primitive_cell  = True, scale = True, allow_subset = True, attempt_supercell = True):
        """
        To check if pymatgen object from the original cif file match with the generated...
        pymatgen structure from cif_string_decoder_sym method out of string cif representation...
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
        
        output_struct = self.cif_string_decoder_sym(input)

        return StructureMatcher(ltol, stol, angle_tol, primitive_cell, scale, allow_subset, attempt_supercell).fit(output_struct, original_struct)


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

