from pyscf import gto
from NECSCF.DFT import u_scfcycle_fgh
from NECSCF.Nuclear import geom
import numpy as np
import matplotlib.pyplot as plt

# NECSCF-HF results for F-H-F anion

mol=gto.Mole()
mol.atom = [['F',(0, 0, -1.35)],
            ['H',(0, 0, -0.0)],
            ['F',(0, 0, 1.35)]]
mol.basis = 'def2-SVP'
mol.charge = -1
mol.spin = 0
mol.build()

N = 100
NQ = 50
Q = geom.switch_geom(mol)
step = 0.01*1.88972612
v = 1
DisVector = np.array([[0.00000, 0.00000, -0.00000],
                      [0.00000, 0.00000,  1.00000],
                      [0.00000, 0.00000,  0.00000]])

X_k = np.array((np.array(range(N)) - int(NQ - 1))*step/1.88972612) + 1.35
Geom_k = geom.geom_gen(Q, DisVector, N, NQ, step/1.88972612)

print('Molecular Geometries:\n', np.array(Geom_k).real)

aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, en_energy, nuc_wfn, NUC_GRAD = u_scfcycle_fgh.necscf_run(mol, Q, DisVector, N, NQ, step, v, multi = 25, micro=False, egy_tol=1.0e-4, rms_tol=1.0e-4, maxcycle=20, micro_tol=1.0e-6, micro_cycle=20, cptype = 'HF')

plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--', color = 'red', label='NECSCF-DFT')
plt.xlabel('Displacement / a.u.')
plt.ylabel('Energy / Hartree')
plt.legend()
plt.savefig(f'PESs_FGHv{v}.png')
plt.close()
plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--', color = 'red')
plt.xlabel('Displacement / a.u.')
plt.ylabel('Energy / Hartree')
plt.savefig(f"FinalPES_FGH.png")


plt.figure()
plt.plot(X_k, (nuc_wfn)**2, 'go', X_k, (nuc_wfn)**2, 'g--', label='Nuclear Density')
plt.plot(X_k, NUC_GRAD, 'go', X_k, NUC_GRAD, 'g--', color = 'red', label = 'Nuclear Derivative Term')
plt.legend()
plt.xlabel('Displacement / a.u.')
plt.savefig(f'Final_NUCv{v}.png')
plt.close()