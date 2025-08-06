#!/usr/bin/env python

from pyscf import gto, scf, dft
from NECSCF.DFT import u_necscf_fgh, ucpks, ucphf, ucphf_field
from NECSCF.Nuclear import geom, fgh
import numpy
import copy
from numpy.linalg import norm
from numpy import sqrt, einsum
from functools import reduce
from pathos.multiprocessing import ProcessPool as Pool
import matplotlib.pyplot as plt


def necscf_run(mol, Q, DisVector, N, NQ, step, v, multi, micro, egy_tol=1.0e-8, rms_tol=1.0e-4, maxcycle=50, micro_tol=1.0e-4, micro_cycle=20, cptype = 'HF'):
    mf = dft.uks.UKS(mol)
    mf.conv_tol_grad = 10e-8
    mf.xc = 'PBE'
    SCF_conv = False
    ncycle = 0

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
    e1 = [0.]*N
    e2 = [0.]*N
    e_coup = [0.]*N
    e_nuc = [0.]*N
    vec_pon = [0.]*N
    NUC_GRAD = [None]*N
    cycles = [None]*N

    mass = geom.reduced_mass(mol, DisVector)
    Geom_k = geom.geom_gen(Q, DisVector, N, NQ, step/1.88972612)
    X_k = numpy.array((numpy.array(range(N)) - int(NQ - 1))*step/1.88972612) + 1.4
    print ('\nNucleus-Electron Coupled Self-Consistent Field method for treating nonadiabatic electron-nuclei coupling effects')
    print ('\nInitial Guess with UKS solutions')
    # Defind number of round
    if N % multi == 0:
        Nround = int(N / multi)
    else:
        Nround = int(N / multi + 1)
    pool = Pool(multi)

    def initial_guess(X):
        geom.switch_mol(mol, Geom_k[X])
        BO_mf = dft.uks.UKS(mol)
        BO_mf.conv_tol_grad = 10e-8
        BO_mf.kernel()
        e_elec, mo_energy, mo_coeff, mo_occ = BO_mf.e_tot, BO_mf.mo_energy, BO_mf.mo_coeff, BO_mf.mo_occ
        e1, e2 = BO_mf.energy_elec()[0] - BO_mf.energy_elec()[1], BO_mf.energy_elec()[1]
        e_nuc = BO_mf.energy_nuc()
        if cptype == 'HF':
            _aU1, _bU1 = ucphf.make_U(mol, mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1], mo_energy[0], mo_energy[1], mo_occ[0], mo_occ[1])
        elif cptype == 'HF-Field':
            _aU1, _bU1 = ucphf_field.make_U(mol, mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1], mo_energy[0], mo_energy[1], mo_occ[0], mo_occ[1])
        elif cptype == 'KS':
            _aU1, _bU1 = ucpks.make_U(BO_mf, mol, mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1], mo_energy[0], mo_energy[1], mo_occ[0], mo_occ[1])
        return mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1], mo_energy[0], mo_energy[1], mo_occ[0], mo_occ[1], e_elec, e1, e2, e_coup, e_nuc, _aU1, _bU1
    

    Ncycle = 0
    while Ncycle < Nround:
        if Ncycle < int(N / multi):
            results = pool.map(initial_guess,range(Ncycle*multi, \
                 Ncycle*multi+multi))
            for i in range(Ncycle*multi, Ncycle*multi+multi):
                aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                       e1[i], e2[i], e_coup[i], e_nuc[i], aU1[i], bU1[i] = results[i-Ncycle*multi]
            Ncycle = Ncycle + 1
        else:
            results = pool.map(initial_guess,range(Ncycle*multi, N))
            for i in range(Ncycle*multi, N):
                aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                       e1[i], e2[i], e_coup[i], e_nuc[i], aU1[i], bU1[i] = results[i-Ncycle*multi]
            Ncycle = Ncycle + 1


    print ('\nInitial UKS PES')
    print ('Potential Energy Surface')
    print (numpy.array(en_energy).real)
    print("Contribution from Core Hamiltonian")
    print (numpy.array(e1).real)
    print("########  Contribution from Two-Electron Integrals  ########")
    print (numpy.array(e2).real)
    print("########  Contribution from nuclear repulsion  ########")
    print (numpy.array(e_nuc).real)

    nuc_energy, nuc_wfn, NUC_GRAD, harm, anharm = fgh.NUC_iter(N*step, N, v, en_energy, vec_pon, mass, freq_cal=True)
    nuc_kinegy = nuc_wfn.T.conj().dot(fgh.secgrad(N*step, N)).dot(nuc_wfn)*(-1./2./mass)*step
    nuc_poten = (nuc_wfn.T.conj()*en_energy).dot(nuc_wfn)*step
    nuc_density = nuc_wfn.T.conj()*nuc_wfn


    for i in range(N):
        if abs(NUC_GRAD[i]) > 1e2:
            NUC_GRAD[i] = numpy.sign(NUC_GRAD[i]) * 1e2
            print('Very large nuclear term found near node - Set to +/-1e2')

    print ('Nuclear Kinetic Energy')
    print (nuc_kinegy)
    print ('Nuclear Potential Energy')
    print (nuc_poten)
    print ('Total Energy')
    print (nuc_energy)
    print ('Nuclear gradient at grid points')
    print (NUC_GRAD.real)
    print ('Harmonic fundamental frequency [cm-1]')
    print (harm)
    print ('Anharmonic fundamental frequency [cm-1]')
    print (anharm)

    
    plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--')
    plt.xlabel('Displacement / a.u.')
    plt.ylabel('Energy / Hartree')
    plt.savefig(f'InitialPESTightv{v}.png')
    plt.close()
    plt.figure()
    plt.plot(X_k, nuc_density, 'go', X_k, nuc_density, 'g--', label='Nuclear Density')
    plt.plot(X_k, nuc_wfn, 'go', X_k, nuc_wfn, 'g--', color = 'blue', label = 'Nuclear Wavefunction')
    plt.plot(X_k, NUC_GRAD, 'go', X_k, NUC_GRAD, 'g--', color = 'red', label = 'Nuclear Derivative Term')
    plt.legend()
    plt.xlabel('Displacement / a.u.')
    plt.savefig(f'InitialNUCTightv{v}.png')
    plt.close()
    plt.figure()
    plt.plot(X_k, en_energy, 'ko', X_k, en_energy, 'k--', label = 'BO')

    print ('\n******** Entering NECSCF Cycle *********')

    vec_pon = [0.000001 * 1.0j] * N
    print('Introducing vector potential:', numpy.array(vec_pon))
    nuc_energy, nuc_wfn, NUC_GRAD, harm, anharm = fgh.NUC_iter(N*step, N, v, en_energy, vec_pon, mass, freq_cal=True)
    nuc_kinegy = nuc_wfn.T.conj().dot(fgh.secgrad(N*step, N)).dot(nuc_wfn)*(-1./2./mass)*step
    nuc_poten = (nuc_wfn.T.conj()*en_energy).dot(nuc_wfn)*step
    nuc_density = nuc_wfn.T.conj()*nuc_wfn

    for i in range(N):
        if abs(NUC_GRAD[i]) > 1e2:
            NUC_GRAD[i] = numpy.sign(NUC_GRAD[i]) * 1e2
            print('Very large nuclear term found near node - Set to +/-1e2')

    print('Initial Nuclear Data w/ non-zero vector potential')
    print ('Nuclear Kinetic Energy')
    print (nuc_kinegy)
    print ('Nuclear Potential Energy')
    print (nuc_poten)
    print ('Total Energy')
    print (nuc_energy)
    print ('Nuclear gradient at grid points')
    print (NUC_GRAD.real)
    print ('Harmonic fundamental frequency [cm-1]')
    print (harm)
    print ('Anharmonic fundamental frequency [cm-1]')
    print (anharm)

    Ncycle = 0
    def itera(x):
        geom.switch_mol(mol, Geom_k[x])
        ks = dft.uks.UKS(mol)
        if cptype == 'HF':
            _aU1, _bU1 = ucphf.make_U(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        elif cptype == 'HF-Field':
            _aU1, _bU1 = ucphf_field.make_U(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        elif cptype == 'KS':
            aU1, bU1 = ucpks.make_U(ks, mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        _aCL, _bCL, _aCR, _bCR, _amo_energy, _bmo_energy, _amo_occ, _bmo_occ, _energy, _e1, _e2, _e_coup, e_nuc  \
                = u_necscf_fgh.SCF_iter(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_occ[x], bmo_occ[x], aU1[x], bU1[x], NUC_GRAD[x], DisVector, mass)
        return _aCL, _bCL, _aCR, _bCR, _amo_energy, _bmo_energy, _amo_occ, _bmo_occ, _energy, _e1, _e2, _e_coup, e_nuc, _aU1, _bU1

    def iteration(x):
        geom.switch_mol(mol, Geom_k[x])
        ks = dft.uks.UKS(mol)
        if cptype == 'HF':
            _aU1, _bU1 = ucphf.make_U(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        elif cptype == 'HF-Field':
            _aU1, _bU1 = ucphf_field.make_U(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        elif cptype == 'KS':
            _aU1, _bU1 = ucpks.make_U(ks, mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_energy[x], bmo_energy[x], amo_occ[x], bmo_occ[x])
        print(f'CPHF converged for point {x}')
        if (micro==False):
            _aCL, _bCL, _aCR, _bCR, _amo_energy, _bmo_energy, _amo_occ, _bmo_occ, _energy, _e1, _e2, _e_coup, e_nuc, vec_pon, cycles \
                = u_necscf_fgh.SCF_iter(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_occ[x], bmo_occ[x], aU1[x], bU1[x], NUC_GRAD[x], DisVector, mass)
        else:
            _aCL, _bCL, _aCR, _bCR, _amo_energy, _bmo_energy, _amo_occ, _bmo_occ, _energy, _e1, _e2, _e_coup, e_nuc, vec_pon, cycles \
                = u_necscf_fgh.SCF_microiter(mol, aCL[x], bCL[x], aCR[x], bCR[x], amo_occ[x], bmo_occ[x], aU1[x], bU1[x], NUC_GRAD[x], DisVector, mass, micro_tol, micro_cycle, cptype)
        return _aCL, _bCL, _aCR, _bCR, _amo_energy, _bmo_energy, _amo_occ, _bmo_occ, _energy, _e1, _e2, _e_coup, e_nuc, vec_pon, _aU1, _bU1, cycles
    

    while Ncycle < Nround:
        if Ncycle < int(N / multi):
            results = pool.map(iteration,range(Ncycle*multi, \
                 Ncycle*multi+multi))
            for i in range(Ncycle*multi, Ncycle*multi+multi):
                aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                       e1[i], e2[i], e_coup[i], e_nuc[i], vec_pon[i], aU1[i], bU1[i], cycles[i] = results[i-Ncycle*multi]
            Ncycle = Ncycle + 1
        else:
            results = pool.map(iteration,range(Ncycle*multi, N))
            for i in range(Ncycle*multi, N):
                aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                       e1[i], e2[i], e_coup[i], e_nuc[i], vec_pon[i], aU1[i], bU1[i], cycles[i] = results[i-Ncycle*multi]
            Ncycle = Ncycle + 1
    
    print ('NECSCF Potential Energy Surface')
    print (numpy.array(en_energy).real)

    print('Number of micro iterations at each grid point:')
    print(numpy.array(cycles))

    nuc_energy, nuc_wfn, NUC_GRAD, harm, anharm = fgh.NUC_iter(N*step, N, v, en_energy, vec_pon, mass, freq_cal=True)
    nuc_kinegy = nuc_wfn.T.conj().dot(fgh.secgrad(N*step, N)).dot(nuc_wfn)*(-1./2./mass)*step
    nuc_poten = (nuc_wfn.T.conj()*en_energy).dot(nuc_wfn)*step


    for i in range(N):
        if abs(NUC_GRAD[i]) > 1e2:
            NUC_GRAD[i] = numpy.sign(NUC_GRAD[i]) * 1e2
            print('Very large nuclear term found near node - Set to +/-1e2')

    print ('Nuclear Kinetic Energy')
    print (nuc_kinegy)
    print ('Nuclear Potential Energy')
    print (nuc_poten)
    print ('Total Energy')
    print (nuc_energy)
    print ('Nuclear gradient at grid points')
    print (NUC_GRAD.real)
    print ('Harmonic fundamental frequency [cm-1]')
    print (harm)
    print ('Anharmonic fundamental frequency [cm-1]')
    print (anharm)
    print ('Vector potential')
    print (vec_pon)

    while not SCF_conv and ncycle <= max(1, maxcycle):
        print ('\n\nIteration %d \n\n' % (ncycle))
        en_energy_last, aU1_last, bU1_last, E_tot_last, aCR_last, bCR_last, aCL_last, bCL_last, amo_occ_last, bmo_occ_last,amo_energy_last,bmo_energy_last \
               = list(en_energy), list(aU1), list(bU1), nuc_energy, list(aCR), list(bCR), list(aCL), list(bCL), list(amo_occ), list(bmo_occ),list(amo_energy),list(bmo_energy)
        Ncycle = 0

        print ('Nuclear Kinetic Energy')
        print (nuc_kinegy)
        print ('Nuclear Potential Energy')
        print (nuc_poten)
        print ('Nuclear gradient at grid points')
        print (NUC_GRAD.real)
        print ('Harmonic fundamental frequency [cm-1]')
        print (harm)
        print ('Anharmonic fundamental frequency [cm-1]')
        print (anharm)
        print ('Vector potential')
        print (vec_pon)
        

        while Ncycle < Nround:
            if Ncycle < int(N / multi):
                results = pool.map(iteration,range(Ncycle*multi, \
                     Ncycle*multi+multi))
                for i in range(Ncycle*multi, Ncycle*multi+multi):
                    aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                           e1[i], e2[i], e_coup[i], e_nuc[i], vec_pon[i], aU1[i], bU1[i], cycles[i] = results[i-Ncycle*multi]
                Ncycle = Ncycle + 1
            else:
                results = pool.map(iteration,range(Ncycle*multi, N))
                for i in range(Ncycle*multi, N):
                    aCL[i], bCL[i], aCR[i], bCR[i], amo_energy[i], bmo_energy[i], amo_occ[i], bmo_occ[i], en_energy[i], \
                        e1[i], e2[i], e_coup[i], e_nuc[i], vec_pon[i], aU1[i], bU1[i], cycles[i] = results[i-Ncycle*multi]
                Ncycle = Ncycle + 1

        print ('Potential Energy Surface')
        print (numpy.array(en_energy).real)
        print('Number of micro iterations at each grid point:')
        print(numpy.array(cycles))

        nuc_energy, nuc_wfn, NUC_GRAD, harm, anharm = fgh.NUC_iter(N*step, N, v, en_energy, vec_pon, mass, freq_cal=True)
        nuc_kinegy = nuc_wfn.T.conj().dot(fgh.secgrad(N*step, N)).dot(nuc_wfn)*(-1./2./mass)*step
        nuc_poten = (nuc_wfn.T.conj()*en_energy).dot(nuc_wfn)*step
        nuc_density = nuc_wfn.T.conj()*nuc_wfn

        # Included to aid numerical stability

        for i in range(N):
            if abs(NUC_GRAD[i]) > 1e2:
                NUC_GRAD[i] = numpy.sign(NUC_GRAD[i]) * 1e2
                print('Very large nuclear term found near node - Set to +/-1e2')

        print ('Nuclear Kinetic Energy')
        print (nuc_kinegy)
        print ('Nuclear Potential Energy')
        print (nuc_poten)
        print ('Nuclear gradient at grid points')
        print (NUC_GRAD.real)
        print ('Harmonic fundamental frequency [cm-1]')
        print (harm)
        print ('Anharmonic fundamental frequency [cm-1]')
        print (anharm)
        print ('Vector potential')
        print (vec_pon)

        RMS_elec = (norm(numpy.array(en_energy) - numpy.array(en_energy_last))/sqrt(N))
        print ('RMS electronic energy change at all grid points: \n %.15f' % (norm(numpy.array(en_energy) - numpy.array(en_energy_last))/sqrt(N)))
        ARMS_DM = 0.
        BRMS_DM = 0.
        for i in range(N):
            #print 'CR',CR[i]
            #print 'CR_last',CR_last[i]
            ARMS_DM = ARMS_DM + norm(abs(2*aCR[i][:,amo_occ[i]>0].dot(aCL[i][amo_occ[i]>0,:])) - abs(2*aCR_last[i][:,amo_occ_last[i]>0].dot(aCL_last[i][amo_occ_last[i]>0,:])))
            BRMS_DM = BRMS_DM + norm(abs(2*bCR[i][:,bmo_occ[i]>0].dot(bCL[i][bmo_occ[i]>0,:])) - abs(2*bCR_last[i][:,bmo_occ_last[i]>0].dot(bCL_last[i][bmo_occ_last[i]>0,:])))
        print ('RMS density matrix change at all grid points: \n %.15f, %.15f' % (ARMS_DM/sqrt(N), BRMS_DM/sqrt(N)))
        #print 'RMS response density change at all grid points: \n %.15f' % (RMS_U1/sqrt(N))
        print ('Total energy change: \n %.15f+%.15fj' % ((nuc_energy - E_tot_last).real, (nuc_energy - E_tot_last).imag))
        if (abs(nuc_energy - E_tot_last) < egy_tol) and (abs(RMS_elec) < rms_tol):
            SCF_conv = True
            print ('\n******************* NECSCF Results Summary *******************')
            print ('NECSCF convergence is %s' % SCF_conv)
            print ('Final Potential Energy Surface')
            print (numpy.array(en_energy).real)
            print("Contribution from Core Hamiltonian")
            print (numpy.array(e1).real)
            print("########  Contribution from Two-Electron Integrals  ########")
            print (numpy.array(e2).real)
            print("########  Contribution from nucleus-electron coupling  ########")
            print (numpy.array(e_coup).real)
            print("########  Contribution from nuclear repulsion  ########")
            print (numpy.array(e_nuc).real)

            nuc_energy, nuc_wfn, NUC_GRAD = fgh.NUC_iter(N*step, N, v, en_energy, vec_pon, mass, freq_cal=False)

            print ('\nNuc wfn')
            print (nuc_wfn[0].real)
            print ('Total Energy of the System')
            print (nuc_energy)
            
        ncycle += 1
    return aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, en_energy, nuc_wfn, NUC_GRAD
