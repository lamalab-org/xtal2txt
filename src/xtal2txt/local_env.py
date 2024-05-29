"""Analyze the local environment of atoms in a structure.

This module requires installation of Openbabel, e.g. via conda:

.. code-block:: bash

        conda install -c conda-forge openbabel
"""

from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import (
    SimplestChemenvStrategy,
)
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import (
    LocalGeometryFinder,
)
from pymatgen.analysis.chemenv.coordination_environments.structure_environments import (
    LightStructureEnvironments,
)
from pymatgen.core import Structure, Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import (
    AllCoordinationGeometries,
)
from typing import Tuple, List

strategy = SimplestChemenvStrategy()

lgf = LocalGeometryFinder()

from pymatgen.io.babel import BabelMolAdaptor


class LocalEnvAnalyzer:
    """A class to analyze the local environment of atoms in a structure."""

    def __init__(self, distance_cutoff: float = 1.4, angle_cutoff: float = 0.3):
        """
        Args:
            distance_cutoff: The distance cutoff to use for determining the nearest neighbors of each atom.
            angle_cutoff: The angle cutoff to use for determining the nearest neighbors of each atom.
        """
        self.distance_cutoff = distance_cutoff
        self.angle_cutoff = angle_cutoff

    def get_local_environments(
        self, structure: Structure
    ) -> Tuple[List[dict], List[dict], str]:
        """Get the local environments of the atoms in a structure.

        Args:
            structure: pymatgen Structure object

        Returns:
            Tuple[List[dict], List[dict]]: A list of dictionaries containing the local environments of the atoms in the structure,
                and a list of dictionaries containing the unknown sites.
        """
        # since we do not want all chemical environments, but only the ones that are unique
        # we need to get the symmetrized structure
        sga = SpacegroupAnalyzer(structure)
        symm_struct = sga.get_symmetrized_structure()

        inequivalent_indices = [
            indices[0] for indices in symm_struct.equivalent_indices
        ]
        wyckoffs = symm_struct.wyckoff_symbols

        # a Voronoi tessellation is used to determine the local environment of each atom
        # that is, the nearest neighbors of each atom
        # this has been proposed in O’Keeffe, M. (1979). Acta Cryst. A35, 772–775.
        # and modified for the ChemEnv paper
        # The SimplestChemenvStrategy is a strategy uses fixed distance and angle cutoffs
        # to determine the coordination environment of each atom
        # according to the tutorial (https://matgenb.materialsvirtuallab.org/2018/01/01/ChemEnv-How-to-automatically-identify-coordination-environments-in-a-structure.html)
        # "The strategy is correct in about 85% of the cases if one uses distance_cutoff=1.4 and angle_cutoff=0.3"
        lgf = LocalGeometryFinder()
        lgf.setup_structure(structure=structure)
        se = lgf.compute_structure_environments(
            maximum_distance_factor=self.distance_cutoff + 0.01,
            only_indices=inequivalent_indices,
        )
        strategy = SimplestChemenvStrategy(
            distance_cutoff=self.distance_cutoff, angle_cutoff=self.angle_cutoff
        )
        lse = LightStructureEnvironments.from_structure_environments(
            strategy=strategy, structure_environments=se
        )

        all_ce = AllCoordinationGeometries()

        envs = []
        unknown_sites = []
        for index, wyckoff in zip(inequivalent_indices, wyckoffs):
            if not lse.neighbors_sets[index]:
                unknown_sites.append(f"{structure[index].species_string} ({wyckoff})")
                continue

            # represent the local environment as a molecule
            mol = Molecule.from_sites(
                [structure[index]] + lse.neighbors_sets[index][0].neighb_sites
            )
            mol = mol.get_centered_molecule()
            mg = MoleculeGraph.with_empty_graph(molecule=mol)
            for i in range(1, len(mol)):
                mg.add_edge(0, i)

            env = lse.coordination_environments[index]
            try:
                co = all_ce.get_geometry_from_mp_symbol(env[0]["ce_symbol"])
            except KeyError:
                co = "Unknown"
            moladapter = BabelMolAdaptor.from_molecule_graph(mg)
            smiles = moladapter.pybel_mol.write("can").strip()
            envs.append(
                {
                    "Site": structure[index].species_string,
                    "Wyckoff Label": wyckoff,
                    "Environment": co,
                    "Molecule": mg,
                    "SMILES": smiles,
                }
            )

        return envs, unknown_sites, symm_struct.spacegroup.int_symbol

    def structure_to_local_env_string(
        self, structure: Structure, add_space_group: bool = True
    ) -> str:
        """Convert a structure to a string representation of its local environments.

        The text string might look like

        "I-42d\nS2- (8d) [Cu]S([In])([In])[Cu]\nCu+ (4a) [S][Cu]([S])([S])[S]\nIn3+ (4b) [S][In]([S])[S].[S]"

        Args:
            structure (Structure): pymatgen Structure object
            add_space_group (bool): Whether to add the space group to the string. Defaults to True.

        Returns:
            str: A string representation of the local environments of the atoms in the structure.
        """
        envs, unknown_sites, spacegroup = self.get_local_environments(structure)
        env_str = []

        if add_space_group:
            env_str.append(f"{spacegroup}")
        # sort the environments by the coordination environment and SMILES
        envs = sorted(envs, key=lambda x: (x["Wyckoff Label"], x["SMILES"]))
        for env in envs:
            env_str.append(f"{env['Site']} ({env['Wyckoff Label']}) {env['SMILES']}")

        if unknown_sites:
            for site in unknown_sites:
                env_str.append(str(site))
        return "\n".join(env_str)
