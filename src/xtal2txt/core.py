import logging
import random
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Union, Callable, Any

from pymatgen.core import Structure
from pymatgen.core.structure import Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from robocrys import StructureCondenser, StructureDescriber

from xtal2txt.transforms import TransformationCallback
from xtal2txt.local_env import LocalEnvAnalyzer

logger = logging.getLogger(__name__)


class RepresentationType(Enum):
    """Enumeration of available text representation types."""

    CIF_P1 = "cif_p1"
    CIF_SYMMETRIZED = "cif_symmetrized"
    SLICES = "slices"
    COMPOSITION = "composition"
    CRYSTAL_TEXT_LLM = "crystal_text_llm"
    ROBOCRYS = "robocrys_rep"
    ATOM_SEQUENCES = "atom_sequences"
    ATOM_SEQUENCES_PLUSPLUS = "atom_sequences_plusplus"
    ZMATRIX = "zmatrix"
    LOCAL_ENV = "local_env"


class TextRep:
    """
    Generate text representations of crystal structure for Language modelling.

    Attributes:
        structure : pymatgen structure
        transformations : list of transformations to apply

    Methods:
        from_input : classmethod to create TextRep from various inputs
        get_available_representations : get list of available representation types
        get_cif_string : generate CIF string representation
        get_lattice_parameters : get lattice parameters
        get_coords : get atomic coordinates
        get_slices : get SLICES representation
        get_composition : get composition
        get_local_env_rep : get local environment representation
        get_crystal_text_llm : get Crystal-LLM representation
        get_robocrys_rep : get Robocrystallographer representation
        get_wyckoff_positions : get Wyckoff positions
        get_wycryst : get Wycryst representation
        get_atom_sequences_plusplus : get atom sequences
        get_zmatrix_rep : get Z-matrix representation
        get_all_text_reps : get all available representations
        get_requested_text_reps : get specific representation(s)
    """

    def __init__(
        self,
        structure: Structure,
        transformations: list[tuple[str, dict]] = None,
        enable_logging: bool = False,
    ) -> None:
        """
        Initialize TextRep instance.

        Args:
            structure: Pymatgen Structure object.
            transformations: list of (transformation_name, params) tuples to apply.
            enable_logging: Whether to log errors when representations fail.
        """
        self.structure = structure
        self.transformations = transformations or []
        self.enable_logging = enable_logging

        # SLICES backend is lazy-loaded as versions keep changing
        self._backend = None
        self.condenser = StructureCondenser()
        self.describer = StructureDescriber()

        # Build registry of representation generators
        self._build_registry()

        self.apply_transformations()

    @property
    def backend(self):
        """Lazy-load SLICES backend as versions keep changing."""
        if self._backend is None:
            try:
                from slices.core import SLICES

                self._backend = SLICES()
            except ImportError as e:
                if self.enable_logging:
                    logger.error(f"Failed to import SLICES backend: {e}")
                raise ImportError(
                    "SLICES backend is not available. Please install slices package with compatible dependencies."
                ) from e
        return self._backend

    def _build_registry(self) -> None:
        """Build registry mapping representation names to their generator functions."""
        self._rep_registry: dict[str, Callable] = {
            RepresentationType.CIF_P1.value: lambda dp: self.get_cif_string(
                format="p1", decimal_places=dp
            ),
            RepresentationType.CIF_SYMMETRIZED.value: lambda dp: self.get_cif_string(
                format="symmetrized", decimal_places=dp
            ),
            RepresentationType.SLICES.value: lambda dp: self.get_slices(),
            RepresentationType.COMPOSITION.value: lambda dp: self.get_composition(),
            RepresentationType.CRYSTAL_TEXT_LLM.value: lambda dp: (
                self.get_crystal_text_llm()
            ),
            RepresentationType.ROBOCRYS.value: lambda dp: self.get_robocrys_rep(),
            RepresentationType.ATOM_SEQUENCES.value: lambda dp: (
                self.get_atom_sequences_plusplus(
                    lattice_params=False, decimal_places=dp
                )
            ),
            RepresentationType.ATOM_SEQUENCES_PLUSPLUS.value: lambda dp: (
                self.get_atom_sequences_plusplus(lattice_params=True, decimal_places=dp)
            ),
            RepresentationType.ZMATRIX.value: lambda dp: self.get_zmatrix_rep(
                decimal_places=dp
            ),
            RepresentationType.LOCAL_ENV.value: lambda dp: self.get_local_env_rep(
                local_env_kwargs=None
            ),
        }

    @classmethod
    def get_available_representations(cls) -> list[str]:
        """
        Get list of available representation types.

        Returns:
            list[str]: list of representation type names.
        """
        return [rep.value for rep in RepresentationType]

    @classmethod
    def from_input(
        cls,
        input_data: Union[str, Path, Structure],
        transformations: list[tuple[str, dict]] = None,
        enable_logging: bool = False,
    ) -> "TextRep":
        """
        Instantiate the TextRep class object with the pymatgen structure from a cif file, a cif string, or a pymatgen Structure object.

        Args:
            input_data: A cif file of a crystal structure, a cif string,
                or a pymatgen Structure object.
            transformations: list of transformations to apply.
            enable_logging: Whether to log errors when representations fail.

        Returns:
            TextRep: A TextRep object.
        """
        if isinstance(input_data, Structure):
            structure = input_data

        elif isinstance(input_data, (str, Path)):
            try:
                if Path(input_data).is_file():
                    structure = Structure.from_file(str(input_data))
                else:
                    raise ValueError
            except (OSError, ValueError):
                structure = Structure.from_str(str(input_data), "cif")

        else:
            structure = Structure.from_str(str(input_data), "cif")

        return cls(structure, transformations, enable_logging)

    def apply_transformations(self) -> None:
        """
        Apply transformations to the structure.
        """
        for transformation, params in self.transformations:
            transform_func = getattr(TransformationCallback, transformation)
            self.structure = transform_func(self.structure, **params)

    def _safe_call(self, func: Callable, *args, **kwargs) -> Any | None:
        """
        Safely call a function and return None if it fails.

        Args:
            func: Function to call.
            *args: Positional arguments to pass to func.
            **kwargs: Keyword arguments to pass to func.

        Returns:
            Result of func or None if exception occurs.
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if self.enable_logging:
                logger.warning(
                    f"Failed to generate representation using {func.__name__}: {e}"
                )
            return None

    @staticmethod
    def round_numbers_in_string(original_string: str, decimal_places: int) -> str:
        """
        Rounds float numbers in the given string to the specified number of decimal places using regex.

        Args:
            original_string (str): The input string.
            decimal_places (int): The number of decimal places to round to.

        Returns:
            str: The string with the float numbers rounded to the specified number of decimal places.
        """
        pattern = r"\b\d+\.\d+\b"
        matches = re.findall(pattern, original_string)
        rounded_numbers = [round(float(match), decimal_places) for match in matches]
        new_string = re.sub(
            pattern, lambda x: str(rounded_numbers.pop(0)), original_string
        )
        return new_string

    def get_cif_string(
        self, format: str = "symmetrized", decimal_places: int = 3
    ) -> str:
        """
        Generate CIF as string in multi-line format.

        All float numbers can be rounded to the specified number (decimal_places).
        Currently supports two formats. Symmetrized (cif with symmetry operations and the least symmetric basis) ...
        and P1 (conventional unit cell , with all the atoms listed and only identity as symmetry operation).

        Args:
            format (str): The format of the CIF file. Can be "symmetrized" or "p1".
            decimal_places (int): The number of decimal places to round to.

        Returns:
            str: The CIF string.
        """

        if format == "symmetrized":
            symmetry_analyzer = SpacegroupAnalyzer(self.structure)
            symmetrized_structure = symmetry_analyzer.get_symmetrized_structure()
            cif_string = str(
                CifWriter(
                    symmetrized_structure,
                    symprec=0.1,
                    significant_figures=decimal_places,
                ).cif_file
            )
            cif = "\n".join(cif_string.split("\n")[1:])
            return self.round_numbers_in_string(cif, decimal_places)

        elif format == "p1":
            cif_string = "\n".join(self.structure.to(fmt="cif").split("\n")[1:])
            return self.round_numbers_in_string(cif_string, decimal_places)

    def get_lattice_parameters(self, decimal_places: int = 3) -> list[str]:
        """
        Return lattice parameters of unit cells in a crystal lattice:
        the lengths of the cell edges (a, b, and c) in angstrom and the angles between them (alpha, beta, and gamma) in degrees.

        All float numbers can be rounded to a specific number (decimal_places).

        Args:
            decimal_places (int): The number of decimal places to round to.

        Returns:
            list[str]: The lattice parameters.
        """
        return [
            str(round(i, decimal_places)) for i in self.structure.lattice.parameters
        ]

    def get_coords(self, name: str = "cartesian", decimal_places: int = 3) -> list[str]:
        """
        Return list of atoms in unit cell for with their positions in Cartesian or fractional coordinates as per choice.

        Args:
            name (str): The name of the coordinates. Can be "cartesian" or "fractional".
            decimal_places (int): The number of decimal places to round to.

        Returns:
            list[str]: The list of atoms with their positions.
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

    def get_slices(self, primitive: bool = True) -> str:
        """Returns SLICES representation of the crystal structure.
        https://www.nature.com/articles/s41467-023-42870-7

        Args:
            primitive (bool): Whether to use the primitive structure or not.

        Returns:
            str: The SLICE representation of the crystal structure.
        """

        if primitive:
            primitive_structure = (
                self.structure.get_primitive_structure()
            )  # convert to primitive structure
            return self.backend.structure2SLICES(primitive_structure)
        return self.backend.structure2SLICES(self.structure)

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

    def get_local_env_rep(self, local_env_kwargs: dict | None = None) -> str:
        """
        Get the local environment representation of the crystal structure.

        The local environment representation is a string that contains
        the space group symbol and the local environment of each atom in the unit cell.
        The local environment of each atom is represented as SMILES string and the
        Wyckoff symbol of the local environment.

        Args:
            local_env_kwargs (dict): Keyword arguments to pass to the LocalEnvAnalyzer.

        Returns:
            str: The local environment representation of the crystal structure.
        """
        if not local_env_kwargs:
            local_env_kwargs = {}
        analyzer = LocalEnvAnalyzer(**local_env_kwargs)
        return analyzer.structure_to_local_env_string(self.structure)

    def get_crystal_text_llm(
        self,
        permute_atoms: bool = False,
    ) -> str:
        """
        Code adopted from https://github.com/facebookresearch/crystal-llm/blob/main/llama_finetune.py
        https://openreview.net/pdf?id=0r5DE2ZSwJ

        Returns the representation as per the above citation.

        Args:
            permute_atoms (bool): Whether to permute the atoms in the unit cell.

        Returns:
            str: The crystal-llm representation of the crystal structure.
        """

        lengths = self.structure.lattice.parameters[:3]
        angles = self.structure.lattice.parameters[3:]
        atom_ids = self.structure.species
        frac_coords = self.structure.frac_coords

        if permute_atoms:
            atom_coord_pairs = list(zip(atom_ids, frac_coords))
            random.shuffle(atom_coord_pairs)
            atom_ids, frac_coords = zip(*atom_coord_pairs)

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
        """

        condensed_structure = self.condenser.condense_structure(self.structure)
        return self.describer.describe(condensed_structure)

    def get_wyckoff_positions(self):
        """
        Getting wyckoff positions of the elements in the unit cell as the combination of...
        number and letter.

        Returns:
            str:  A multi-line string that contain elements of the unit cell along with their wyckoff position in each line.

        Hint:
            At the end of the string, there is an additional newline character.
        """

        spacegroup_analyzer = SpacegroupAnalyzer(self.structure)
        wyckoff_sites = spacegroup_analyzer.get_symmetry_dataset()
        element_symbols = [site.specie.element.symbol for site in self.structure.sites]

        data = []

        for i in range(len(wyckoff_sites["wyckoffs"])):
            sub_data = (
                element_symbols[i],
                wyckoff_sites["wyckoffs"][i],
                wyckoff_sites["equivalent_atoms"][i],
            )
            data.append(sub_data)

        a = dict(Counter(data))

        output = ""
        for i, j in a.items():
            output += str(i[0]) + " " + str(j) + " " + str(i[1]) + "\n"

        return output

    def get_wycryst(self):
        """
        Obtaining the wyckoff representation for crystal structures that include:
            chemical formula
            space group number
            elements of the unit cell with their wyckoff positions.

        Returns:
            str: A multi-line string that contains the chemical formula, space group number,
                and the elements of the unit cell with their wyckoff positions.
        """
        output = ""
        chemical_formula = self.structure.composition.formula
        output += chemical_formula
        output += "\n" + str(self.structure.get_space_group_info()[1])
        output += "\n" + self.get_wyckoff_positions()

        return output

    def get_atom_sequences_plusplus(
        self, lattice_params: bool = False, decimal_places: int = 1
    ) -> str:
        """
        Generating a string with the elements of composition inside the crystal lattice with the option to
        get the lattice parameters as angles (int) and lengths (float) in a string with a space
        between them

        Args:
            lattice_params (bool): Whether to include lattice parameters or not.
            decimal_places (int): The number of decimal places to round to.

        Returns:
            str: The string representation of the crystal structure.
        """

        try:
            output = [site.specie.element.symbol for site in self.structure.sites]
        except AttributeError:
            output = [site.specie.symbol for site in self.structure.sites]
        if lattice_params:
            params = self.get_lattice_parameters(decimal_places=decimal_places)
            params[3:] = [str(int(float(i))) for i in params[3:]]
            output.extend(params)

        return " ".join(output)

    def updated_zmatrix_rep(self, zmatrix, decimal_places=1):
        """
        Replace the variables in the Z-matrix with their values and return the updated Z-matrix.
        for eg: z-matrix from pymatgen
        'N\nN 1 B1\nN 1 B2 2 A2\nN 1 B3 2 A3 3 D3\n
        B1=3.79
        B2=6.54
        ....
        is replaced to
        'N\nN 1 3.79\nN 1 6.54 2 90\nN 1 6.54 2 90 3 120\n'

        Args:
            Zmatrix (bool): zmatrix multi line string as implemented in pymatgen.
            decimal_places (int): The number of decimal places to round to.

        Returns:
            str: The updated Z-matrix representation of the crystal structure.
        """
        lines = zmatrix.split("\n")
        main_part = []
        variables_part = []

        # Determine the main part and the variables part of the Z-matrix
        for line in lines:
            if "=" in line:
                variables_part.append(line)
            else:
                if line.strip():  # Skip empty lines
                    main_part.append(line)

        # Extract variables from the variables part
        variable_dict = {}
        for var_line in variables_part:
            var, value = var_line.split("=")
            if var.startswith("B"):
                rounded_value = round(float(value.strip()), decimal_places)
            else:
                rounded_value = int(round(float(value.strip())))
            variable_dict[var] = (
                f"{rounded_value}"
                if var.startswith(("A", "D"))
                else f"{rounded_value:.{decimal_places}f}"
            )

        # Replace variables in the main part
        replaced_lines = []
        for line in main_part:
            parts = line.split()
            # atom = parts[0]
            replaced_line = line
            for i in range(1, len(parts)):
                var = parts[i]
                if var in variable_dict:
                    replaced_line = replaced_line.replace(var, variable_dict[var])
            replaced_lines.append(replaced_line)

        return "\n".join(replaced_lines)

    def get_zmatrix_rep(self, decimal_places=1):
        """
        Generate the Z-matrix representation of the crystal structure.
        It provides a description of each atom in terms of its atomic number,
        bond length, bond angle, and dihedral angle, the so-called internal coordinates.

        Disclaimer: The Z-matrix is meant for molecules, current implementation converts atoms within unit cell to molecule.
        Hence the current implentation might overlook bonds acrosse unit cells.
        """
        species = [
            s.element if hasattr(s, "element") else s for s in self.structure.species
        ]
        coords = [c for c in self.structure.cart_coords]
        molecule_ = Molecule(
            species,
            coords,
        )
        zmatrix = molecule_.get_zmatrix()
        return self.updated_zmatrix_rep(zmatrix, decimal_places)

    def get_all_text_reps(
        self, decimal_places: int = 2, include_none: bool = False
    ) -> dict[str, str | None]:
        """
        Returns all the Text representations of the crystal structure in a dictionary.

        Args:
            decimal_places: Number of decimal places to round to.
            include_none: Whether to include None values for unimplemented representations.

        Returns:
            dictionary mapping representation names to their values.
        """
        results = {}

        # Generate all registered representations
        for rep_name, rep_func in self._rep_registry.items():
            results[rep_name] = self._safe_call(rep_func, decimal_places)

        # Add deprecated/unimplemented representations if requested
        if include_none:
            results["cif_bonding"] = None
            results["wycoff_rep"] = None

        return results

    def get_requested_text_reps(
        self, requested_reps: Union[str, list[str]], decimal_places: int = 2
    ) -> Union[str | None, dict[str, str | None]]:
        """
        Returns the requested Text representation(s) of the crystal structure.

        Args:
            requested_reps: Single representation name or list of representation names.
            decimal_places: The number of decimal places to round to.

        Returns:
            If requested_reps is a string: the representation value (or None if failed).
            If requested_reps is a list: dictionary mapping names to values.

        Raises:
            ValueError: If requested representation is not available.
        """
        # Handle single string request (backward compatibility)
        if isinstance(requested_reps, str):
            if requested_reps not in self._rep_registry:
                available = ", ".join(self.get_available_representations())
                raise ValueError(
                    f"Unknown representation '{requested_reps}'. "
                    f"Available representations: {available}"
                )
            rep_func = self._rep_registry[requested_reps]
            return self._safe_call(rep_func, decimal_places)

        # Handle list of requests
        results = {}
        for rep_name in requested_reps:
            if rep_name not in self._rep_registry:
                if self.enable_logging:
                    logger.warning(f"Skipping unknown representation: {rep_name}")
                results[rep_name] = None
                continue
            rep_func = self._rep_registry[rep_name]
            results[rep_name] = self._safe_call(rep_func, decimal_places)

        return results
