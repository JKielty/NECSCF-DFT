from pyscf import gto, scf, dft
from NECSCF.DFT import u_scfcycle_ho
from NECSCF.Nuclear import geom
import numpy as np
import matplotlib.pyplot as plt


mol=gto.Mole()
mol.atom = [['H',(0, 0,  -0.323965)],
            ['H1',(0, 0,  0.323965)]]
mol.basis = 'dz'
mol.build()
ks = dft.uks.UKS(mol)
ks.xc = 'pbe'

N = 10
NQ = 5
Q = geom.switch_geom(mol)
step = 0.05*1.88972612
v = 0
angular_freq = 4751
DisVector = np.array([[0.00000, 0.00000,   -0.89429],
                         [0.00000, 0.00000,    0.44749]])

X_k = list((np.array(range(N)) - int(NQ - 1))*step/1.88972612)


aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, en_energy, nuc_wfn, NUC_GRAD = u_scfcycle_ho.necscf_run(mol, Q, DisVector, N, NQ, step, v, angular_freq, multi = 25, micro=True, egy_tol=1.0e-5, rms_tol=1.0e-4, maxcycle=50, micro_tol=1.0e-8, micro_cycle=30, cptype = 'KS')
