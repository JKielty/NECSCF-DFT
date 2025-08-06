from pyscf import gto
from NECSCF.DFT import u_scfcycle_fgh
from NECSCF.Nuclear import geom
import numpy as np
import matplotlib.pyplot as plt

# Testing F-F PES

mol=gto.Mole()
mol.atom = [['F',(0, 0, -1.1)],
            ['H',(0, 0, 0.0)],
            ['F',(0, 0, 1.1)]]
mol.basis = 'def2-SVP'
mol.charge = -1
mol.spin = 0
mol.build()

N = 100
NQ = 50
Q = geom.switch_geom(mol)
step = 0.004*1.88972612
v = 0
DisVector = np.array([[0.00000, 0.00000,  -1.00000],
                      [0.00000, 0.00000,  0.00000],
                      [0.00000, 0.00000,  1.00000]])


Geom_k = geom.geom_gen(Q, DisVector, N, NQ, step/1.88972612)

print('Molecular Geometries:\n', np.array(Geom_k).real)

aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, en_energy, nuc_wfn, NUC_GRAD = u_scfcycle_fgh.necscf_run(mol, Q, DisVector, N, NQ, step, v, multi = 25, micro=False, egy_tol=1.0e-4, rms_tol=1.0e-3, maxcycle=20, micro_tol=1.0e-6, micro_cycle=20, cptype = 'HF')



