from pyscf import gto, scf, dft
from NECSCF.DFT import u_scfcycle_fgh
from NECSCF.Nuclear import geom
import numpy as np
import matplotlib.pyplot as plt


mol=gto.Mole()
mol.atom = [['H',(0, 0,  -0.384000)],
            ['H1',(0, 0,  0.384000)]]
mol.basis = 'dz'
mol.build()
ks = dft.uks.UKS(mol)
ks.xc = 'pbe'

N = 30
NQ = 15
Q = geom.switch_geom(mol)
step = 0.003*1.88972612
v = 1
angular_freq = 4751
DisVector = np.array([[0.00000, 0.00000,   -0.89429],
                         [0.00000, 0.00000,    0.44749]])

X_k = list((np.array(range(N)) - int(NQ - 1))*step/1.88972612)


aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, en_energy, nuc_wfn, NUC_GRAD = u_scfcycle_fgh.necscf_run(mol, Q, DisVector, N, NQ, step, v, multi = 25, micro=True, egy_tol=1.0e-5, rms_tol=1.0e-4, maxcycle=50, micro_tol=1.0e-8, micro_cycle=25, cptype = 'HF-Field')

plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--', color = 'red', label='NECSCF-DFT')
plt.xlabel('Displacement / a.u.')
plt.ylabel('Energy / Hartree')
plt.legend()
plt.savefig(f'PESs_v{v}.png')
plt.close()
plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--', color = 'red')
plt.xlabel('Displacement / a.u.')
plt.ylabel('Energy / Hartree')
plt.savefig(f"FinalPES_v{v}.png")