#!/usr/bin/env python

from pyscf import gto
from NECSCF.Intgr import integral
import numpy

def reduced_mass(mol, DisVector):
    mass = 0.
    for i in range(mol.natm):
        for j in range(3):
            mass += DisVector[i][j] * DisVector[i][j] * integral.nuc_mass(mol.atom_symbol(i))*integral.u_to_au()
    return mass

def geom_gen(Q, DisVector, N, NQ, step):
    X_k = list(numpy.array(range(N)) - int(NQ - 1))
    Geom = [None]*N
    for i in range(N):
        Geom[i] = Q + DisVector * X_k[i]*step
    return Geom


def switch_mol(mol, Q):
    for i in range(mol.natm):
        mol.atom[i][1] = Q[i]
    mol.build()


def switch_geom(mol):
    geom = numpy.zeros((mol.natm,3))
    for i in range(mol.natm):
        geom[i] = mol.atom[i][1]
    return geom
