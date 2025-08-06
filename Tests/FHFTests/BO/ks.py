from pyscf import gto, dft
from NECSCF.Nuclear import geom
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessPool as Pool

# Born Oppenheimer HF results for F-H-F anion

mol=gto.Mole()
mol.atom = [['F',(0, 0, -1.30)],
            ['H',(0, 0, 0)],
            ['F',(0, 0, 1.30)]]
mol.basis = 'def2-SVP'
mol.charge = -1
mol.spin = 0
mol.build()

N = 100
NQ = 50
Q = geom.switch_geom(mol)
step = 0.025*1.88972612
v = 0
DisVector = np.array([[0.00000, 0.00000,  0.00000],
                      [0.00000, 0.00000,  0.44749],
                      [0.00000, 0.00000,  0.00000]])

X_k = np.array(((np.array(range(N)) - int(NQ - 1))*step/1.88972612))
FH_distance = X_k + mol.atom[0][1][2]
FF_length = int(round(mol.atom[2][1][2] - mol.atom[0][1][2], 2))

aCL = [None]*N
bCL = [None]*N
aCR = [None]*N
bCR = [None]*N
aU1 = [None]*N
bU1 = [None]*N
amo_energy = [None]*N
bmo_energy = [None]*N
amo_occ = [None]*N
bmo_occ = [None]*N
en_energy = [None]*N

Geom_k = geom.geom_gen(Q, DisVector, N, NQ, step/1.88972612)

def initial_guess(X):
    geom.switch_mol(mol, Geom_k[X])
    BO_mf = dft.uks.UKS(mol)
    BO_mf.xc = 'PBE'
    BO_mf.conv_tol_grad = 10e-8
    BO_mf.kernel()
    e_elec, mo_energy, mo_coeff, mo_occ = BO_mf.e_tot, BO_mf.mo_energy, BO_mf.mo_coeff, BO_mf.mo_occ
    return mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1], mo_energy[0], mo_energy[1], mo_occ[0], mo_occ[1], e_elec

multi = 25

if N % multi == 0:
    Nround = int(N / multi)
else:
    Nround = int(N / multi + 1)
pool = Pool(multi)

Ncycle = 0
while Ncycle < Nround:
    if Ncycle < int(N / multi):
        results = pool.map(initial_guess,range(Ncycle*multi, \
                Ncycle*multi+multi))
        for i in range(Ncycle*multi, Ncycle*multi+multi):
            aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i]  = results[i-Ncycle*multi]
        Ncycle = Ncycle + 1
    else:
        results = pool.map(initial_guess,range(Ncycle*multi, N))
        for i in range(Ncycle*multi, N):
            aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i] = results[i-Ncycle*multi]
        Ncycle = Ncycle + 1


print('\nUKS PES for FHF anion:\n', np.array(en_energy).real)

PES = [round(float(x), 8) for x in en_energy]
print('\nParsed PES:\n', PES)


plt.figure
plt.plot(FH_distance, en_energy)
plt.title('UKS PES for F-H-F anion')
plt.xlabel('F-H separation / a.u.')
plt.ylabel('Energy / a.u.')
plt.savefig('UKS_260.png')
    






