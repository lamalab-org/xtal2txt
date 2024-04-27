import os
from pymatgen.core import Structure
import pytest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture
def get_incus2():
    return Structure.from_file(os.path.join(THIS_DIR, "data", "InCuS2_p1.cif"))
