import pytest
from xtal2txt.core import TextRep
from pymatgen.core.structure import Structure as pyStructure
import os


THIS_DIR = os.path.dirname(os.path.abspath(__file__))

N2 = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "N2_p1.cif"))
Sr = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "SrTiO3_p1.cif"))
In = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "InCuS2_p1.cif"))
Tl = TextRep.from_input(os.path.join(THIS_DIR, "..", "data", "TlCr5Se8_p1.cif"))


def test_get_crystal_llm_rep_struc() -> None:
    N2_str = "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    assert type(N2.llm_decoder(N2_str)) == pyStructure

    Sr_str = "3.9 3.9 3.9\n90 90 90\nSr2+\n0.00 0.00 0.00\nTi4+\n0.50 0.50 0.50\nO2-\n0.50 0.50 0.50\nO2-\n0.50 0.00 0.50\nO2-\n0.50 0.50 0.00\nO2-\n0.00 0.50 0.50"
    assert type(Sr.llm_decoder(Sr_str)) == pyStructure

    In_str = "5.5 5.5 11.1\n90 90 90\nIn3+\n0.50 0.50 0.00\nIn3+\n0.00 0.50 0.75\nIn3+\n0.00 0.00 0.50\nIn3+\n0.50 0.00 0.25\nCu+\n0.50 0.00 0.75\nCu+\n0.00 0.00 0.00\nCu+\n0.00 0.50 0.25\nCu+\n0.50 0.50 0.50\nS2-\n0.28 0.25 0.62\nS2-\n0.75 0.22 0.88\nS2-\n0.72 0.75 0.62\nS2-\n0.25 0.78 0.88\nS2-\n0.78 0.75 0.12\nS2-\n0.25 0.72 0.38\nS2-\n0.22 0.25 0.12\nS2-\n0.75 0.28 0.38"
    assert type(In.llm_decoder(In_str)) == pyStructure

    Tl_str = "18.9 3.7 9.1\n90 105 90\nTl+\n0.50 0.50 0.50\nTl+\n0.00 0.00 0.50\nCr3+\n0.00 0.50 0.00\nCr3+\n0.34 0.50 0.03\nCr3+\n0.30 0.50 0.67\nCr3+\n0.20 0.00 0.33\nCr3+\n0.16 0.00 0.97\nCr3+\n0.50 0.00 0.00\nCr3+\n0.84 0.00 0.03\nCr3+\n0.80 0.00 0.67\nCr3+\n0.70 0.50 0.33\nCr3+\n0.66 0.50 0.97\nSe2-\n0.08 0.00 0.15\nSe2-\n0.17 0.50 0.49\nSe2-\n0.33 0.00 0.51\nSe2-\n0.08 0.50 0.82\nSe2-\n0.42 0.00 0.18\nSe2-\n0.26 0.00 0.84\nSe2-\n0.24 0.50 0.16\nSe2-\n0.42 0.50 0.85\nSe2-\n0.58 0.50 0.15\nSe2-\n0.67 0.00 0.49\nSe2-\n0.83 0.50 0.51\nSe2-\n0.58 0.00 0.82\nSe2-\n0.92 0.50 0.18\nSe2-\n0.76 0.50 0.84\nSe2-\n0.74 0.00 0.16\nSe2-\n0.92 0.00 0.85"
    assert type(Tl.llm_decoder(Tl_str)) == pyStructure


def test_llm_matcher() -> None:
    N2_str = "5.6 5.6 5.6\n90 90 90\nN0+\n0.48 0.98 0.52\nN0+\n0.98 0.52 0.48\nN0+\n0.02 0.02 0.02\nN0+\n0.52 0.48 0.98"
    assert N2.llm_matcher(N2_str) == True

    Sr_str = "3.9 3.9 3.9\n90 90 90\nSr2+\n0.00 0.00 0.00\nTi4+\n0.50 0.50 0.50\nO2-\n0.50 0.50 0.50\nO2-\n0.50 0.00 0.50\nO2-\n0.50 0.50 0.00\nO2-\n0.00 0.50 0.50"
    assert Sr.llm_matcher(Sr_str) == True

    In_str = "5.5 5.5 11.1\n90 90 90\nIn3+\n0.50 0.50 0.00\nIn3+\n0.00 0.50 0.75\nIn3+\n0.00 0.00 0.50\nIn3+\n0.50 0.00 0.25\nCu+\n0.50 0.00 0.75\nCu+\n0.00 0.00 0.00\nCu+\n0.00 0.50 0.25\nCu+\n0.50 0.50 0.50\nS2-\n0.28 0.25 0.62\nS2-\n0.75 0.22 0.88\nS2-\n0.72 0.75 0.62\nS2-\n0.25 0.78 0.88\nS2-\n0.78 0.75 0.12\nS2-\n0.25 0.72 0.38\nS2-\n0.22 0.25 0.12\nS2-\n0.75 0.28 0.38"
    assert In.llm_matcher(In_str) == True
    Tl_str = "18.9 3.7 9.1\n90 105 90\nTl+\n0.50 0.50 0.50\nTl+\n0.00 0.00 0.50\nCr3+\n0.00 0.50 0.00\nCr3+\n0.34 0.50 0.03\nCr3+\n0.30 0.50 0.67\nCr3+\n0.20 0.00 0.33\nCr3+\n0.16 0.00 0.97\nCr3+\n0.50 0.00 0.00\nCr3+\n0.84 0.00 0.03\nCr3+\n0.80 0.00 0.67\nCr3+\n0.70 0.50 0.33\nCr3+\n0.66 0.50 0.97\nSe2-\n0.08 0.00 0.15\nSe2-\n0.17 0.50 0.49\nSe2-\n0.33 0.00 0.51\nSe2-\n0.08 0.50 0.82\nSe2-\n0.42 0.00 0.18\nSe2-\n0.26 0.00 0.84\nSe2-\n0.24 0.50 0.16\nSe2-\n0.42 0.50 0.85\nSe2-\n0.58 0.50 0.15\nSe2-\n0.67 0.00 0.49\nSe2-\n0.83 0.50 0.51\nSe2-\n0.58 0.00 0.82\nSe2-\n0.92 0.50 0.18\nSe2-\n0.76 0.50 0.84\nSe2-\n0.74 0.00 0.16\nSe2-\n0.92 0.00 0.85"
    assert Tl.llm_matcher(Tl_str) == True
