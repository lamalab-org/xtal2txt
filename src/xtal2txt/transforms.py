import random
import numpy as np
from pymatgen.core.structure import Structure
from typing import Union, List


def set_seed(seed: int):
    """
    Set the random seed for both random and numpy.random.

    Parameters:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)


class TransformationCallback:
    @staticmethod
    def permute_structure(structure: Structure, seed: int = 42) -> Structure:
        """
        Randomly permute the order of atoms in a structure.
        """
        set_seed(seed)
        shuffled_structure = structure.copy()
        sites = shuffled_structure.sites
        random.shuffle(sites)
        shuffled_structure.sites = sites
        return shuffled_structure

    @staticmethod
    def translate_structure(
        structure: Structure,
        vector: Union[List[float], None] = None,
        seed: int = 42,
        **kwargs,
    ) -> Structure:
        """
        Randomly translate the atoms in a structure.
        """
        set_seed(seed)

        if vector is None:
            vector = np.random.uniform(size=(3,))

        structure.translate_sites(
            indices=range(len(structure.sites)),
            vector=vector,
            # frac_coords=True,
            **kwargs,
        )
        return structure

    @staticmethod
    def translate_single_atom(
        structure: Structure,
        max_indices: int = 1,
        vector: List[float] = [0.25, 0.25, 0.25],
        seed: int = 42,
        **kwargs,
    ) -> Structure:
        """
        Randomly translate one or more atoms in a structure.

        Args:
            structure (Structure): The input structure.
            max_indices (int): The maximum number of atoms to translate. Defaults to 1.
            vector (List[float]): The translation vector. Defaults to [0.25, 0.25, 0.25].
            seed (int): The seed for random number generation. Defaults to 42.

        Returns:
            Structure: The transformed structure.
        """
        set_seed(seed)
        indices = random.sample(
            range(len(structure.sites)), min(max_indices, len(structure.sites))
        )  # ensures that we select at most max_indices from the available sites
        structure.translate_sites(indices=indices, vector=vector, frac_coords=True, **kwargs)
        return structure

    @staticmethod
    def perturb_structure(
        structure: Structure, max_distance: float, seed: int = 42, **kwargs
    ) -> Structure:
        """
        Randomly perturb atoms in a structure.
        """
        set_seed(seed)
        distance = random.uniform(0, max_distance)
        structure.perturb(distance=distance, **kwargs)
        return structure
