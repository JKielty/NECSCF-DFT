#!/usr/bin/env python

from pyscf import gto, scf, dft
from NECSCF.DFT import integral, mathtool, transgrad, ucpks, ucphf_field
import numpy
from numpy import sqrt, trace
from NECSCF.Cphf import ucphf
from scipy.linalg.blas import zgemm, zgemv
from scipy.linalg import norm
from functools import reduce

def INT1(S, CL, CR, U1, S1, mo_occ):
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1, CL, S])
    return term1 + S1.T.conj()

# JK-like terms in TN
def INT2(mol, S, CL, CR, U1, S1, mo_occ, atm_no, dofdes):
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    S1 = integral.int1e3_dof(mol, S1, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [S1.T.conj(), CR[:,mo_occ>0], CL[mo_occ>0,:]])
    term2 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1[:,mo_occ>0], CL[mo_occ>0,:]])
    return term1 + term2

def INT3(mol, S, CL, CR, U1, S1, mo_occ, atm_no, dofdes):
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    S1 = integral.int1e3_dof(mol, S1, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [CL[mo_occ>0,:], S, CR, U1[:, mo_occ>0]])
    term2 = reduce(lambda x, y: zgemm(1.0, x, y), [CL[mo_occ>0,:], S1.T, CR[:, mo_occ>0]])
    return trace(term1+term2)

def INT4(mol, Saa, atm_no, dofdes):
    S_aa = integral.intd1ed9_dof(mol, Saa, atm_no, dofdes)
    return S_aa

def INT5(mol, S, CL, CR, U1, Sa0, mo_occ, atm_no, dofdes):
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    Sa0 = integral.int1e3_dof(mol, Sa0, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [Sa0.T, CR[:,mo_occ==0], CL[mo_occ==0,:], Sa0.T])
    term2 = reduce(lambda x, y: zgemm(1.0, x, y), [Sa0.T, CR[:,mo_occ==0], CL[mo_occ==0,:], S, CR, U1, CL, S])
    term3 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1[:,mo_occ==0], CL[mo_occ==0,:], S, CR, U1, CL, S])
    term4 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1[:,mo_occ==0], CL[mo_occ==0,:], Sa0.T])
    return term1+term2+term3+term4

def INT6(mol, S, CL, CR, U1, U1T, Sa0, Saa, S0a2, mo_occ, atm_no, dofdes):
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    U1T = integral.c1_dof(U1T, atm_no, dofdes)
    Sa0 = integral.int1e3_dof(mol, Sa0, atm_no, dofdes)
    Saa = integral.intd1ed9_dof(mol, Saa, atm_no, dofdes)
    S0a2 = integral.intdd1e9_dof(mol, S0a2, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1T, U1, CL, S])
    term2 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1T, CL, Sa0.T])
    return term1+term2

def INT7(mol, S, CL, CR, U1, S1, mo_occ, atm_no, dofdes):
    S1 = integral.int1e3_dof(mol, S1, atm_no, dofdes)
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [S, CR, U1, CL, S])
    return term1 + S1.T

def vec_eA(mol, S, C, U1, S1, mo_occ, atm_no, dofdes):
    U1 = integral.c1_dof(U1, atm_no, dofdes)
    S1 = integral.int1e3_dof(mol, S1, atm_no, dofdes)
    term1 = reduce(lambda x, y: zgemm(1.0, x, y), [C.T.conj()[mo_occ>0, :], S, C, U1[:, mo_occ>0]])
    term2 = reduce(lambda x, y: zgemm(1.0, x, y), [C.T.conj()[mo_occ>0, :], S1.T.conj(), C[:, mo_occ>0]])
    return trace((term1 + term2) - (term1 + term2).T.conj()) * 0.5

#***********************************************************************
### Nuclear Kinetic Matrix and Electron-Nuclear Coulping Matrix
def make_necoup(mol, S, S_n, aCL, aCR, aU1, amo_occ, grad_nuc, mass):
    anecoup = 1./mass *  grad_nuc * INT1(S, aCL, aCR, aU1, S_n, amo_occ)
    return anecoup

def make_evec_pon(mol, S, Sn, C, U1, mo_occ):
    atmlst = range(mol.natm)
    nao = mol.nao_nr()
    symbol = mol.atom_symbol
    dofdes = ['x', 'y', 'z']
    vec_pon = 0.
    for i in range(len(dofdes)):
        for j in atmlst:
            vec_pon += -0.5/(integral.nuc_mass(mol.atom_symbol(j))*integral.u_to_au()) * vec_eA(mol, S, C, U1, Sn, mo_occ, j, dofdes[i]) * vec_eA(mol, S, C, U1, Sn, mo_occ, j, dofdes[i])
    return vec_pon

def S_n(mol, S_a0, DisVector):
    natm = mol.natm
    nao = S_a0.shape[1]
    dof_des = ['x', 'y', 'z']
    S1 = numpy.zeros((len(dof_des),natm,nao,nao), dtype='complex128')
    for i in range(len(dof_des)):
        for j in range(natm):
            S1[i][j] = integral.int1e3_dof(mol, S_a0, j, dof_des[i])
    S1_trans = transgrad.trans(mol, S1, DisVector)
    return S1_trans

def make_FKS(ks, mol, aCL, aCR, amo_occ, bCL, bCR, bmo_occ):
    h1e = integral.make_h1e(mol)
    adm = mathtool.odm(aCL, aCR, amo_occ)
    bdm = mathtool.odm(bCL, bCR, bmo_occ)
    vhf = integral.make_vhf(ks, mol, adm, bdm)
    avhf, bvhf = vhf[0], vhf[1]
    return (h1e + avhf), (h1e + bvhf)

def make_FKSeff(ks, mol, aCL, aCR, aU1, aU1_trans, amo_occ, bU1, bU1_trans, bCL, bCR, bmo_occ, grad_nuc, DisVector, mass):
    aFKS, bFKS = make_FKS(ks, mol, aCL, aCR, amo_occ, bCL, bCR, bmo_occ)
    S = integral.make_s1e(mol)
    Sa0 = -integral.S_a0(mol)  
    Sn = S_n(mol, Sa0, DisVector)
    anecoup = make_necoup(mol, S, Sn, aCL, aCR, aU1_trans, amo_occ, grad_nuc, mass)
    bnecoup = make_necoup(mol, S, Sn, bCL, bCR, bU1_trans, amo_occ, grad_nuc, mass)
    aevec_pon = make_evec_pon(mol, S, Sa0, aCR, aU1, amo_occ)
    bevec_pon = make_evec_pon(mol, S, Sa0, bCR, bU1, bmo_occ)
    evec_pon = aevec_pon + bevec_pon
    return (aFKS - anecoup), (bFKS - bnecoup), anecoup, bnecoup, evec_pon

def make_energy(ks, mol, h1e, anecoup, bnecoup, adm, bdm):
    e1 = numpy.trace((adm).dot(h1e)) + numpy.trace((bdm).dot(h1e))
    vhf = integral.make_vhf(ks, mol, adm, bdm)
    e2 = vhf.ecoul.real + vhf.exc.real
    e_nuc = mol.energy_nuc()
    e_coup = numpy.trace(adm.dot(-anecoup)) + numpy.trace(bdm.dot(-bnecoup))
    e_tot = e1 + e2 + e_nuc + e_coup
    return e_tot, e1, e2, e_coup, e_nuc



#***********************************************************************
### SCF Cycles


def SCF_iter(mol, aCL, bCL, aCR, bCR, amo_occ, bmo_occ, aU1, bU1, grad_nuc, DisVector, mass):
    MICRO_cycle = 1
    ks = dft.uks.UKS(mol)
    S = integral.make_s1e(mol)
    h1e = integral.make_h1e(mol)
    aU1_trans = transgrad.trans(mol, aU1, DisVector)
    bU1_trans = transgrad.trans(mol, bU1, DisVector)
    aFKSeff, bFKSeff, anecoup, bnecoup, evec_pon = make_FKSeff(ks, mol, aCL, aCR, aU1, aU1_trans, amo_occ, bU1, bU1_trans, bCL, bCR, bmo_occ, grad_nuc, DisVector, mass)
    amo_energy, aCL, aCR = mathtool.geigen(aFKSeff.real, S)
    bmo_energy, bCL, bCR = mathtool.geigen(bFKSeff.real, S)
    MO_ENERGY = (amo_energy, bmo_energy)
    MO_COEFF = (aCR, bCR)
    amo_occ, bmo_occ = ks.get_occ(MO_ENERGY, MO_COEFF)
    adm = mathtool.odm(aCL, aCR, amo_occ)
    bdm = mathtool.odm(bCL, bCR, bmo_occ)
    e_tot, e1, e2, e_coup, e_nuc = make_energy(ks, mol, h1e, anecoup, bnecoup, adm, bdm)
    e_tot = e_tot + evec_pon
    print('Evec potential:', evec_pon)
    return aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, e_tot, e1, e2, e_coup, e_nuc, MICRO_cycle

def SCF_microiter(mol, aCL, bCL, aCR, bCR, amo_occ, bmo_occ, aU1, bU1, grad_nuc, DisVector, mass, micro_tol=1.0e-6, micro_cycle=20, cptype = 'HF'):
    MICRO_cycle = 0
    MICRO_conv = False
    ks = dft.uks.UKS(mol)
    S = integral.make_s1e(mol)
    h1e = integral.make_h1e(mol)
    aU1_trans = transgrad.trans(mol, aU1, DisVector)
    bU1_trans = transgrad.trans(mol, bU1, DisVector)
    aFKSeff, bFKSeff, anecoup, bnecoup, evec_pon = make_FKSeff(ks, mol, aCL, aCR, aU1, aU1_trans, amo_occ, bU1, bU1_trans, bCL, bCR, bmo_occ, grad_nuc, DisVector, mass)
    amo_energy, aCL, aCR = mathtool.geigen(aFKSeff.real, S)
    bmo_energy, bCL, bCR = mathtool.geigen(bFKSeff.real, S)
    MO_ENERGY = (amo_energy, bmo_energy)
    MO_COEFF = (aCR, bCR)
    amo_occ, bmo_occ = ks.get_occ(MO_ENERGY, MO_COEFF)
    adm = mathtool.odm(aCL, aCR, amo_occ)
    bdm = mathtool.odm(bCL, bCR, bmo_occ)
    e_tot, e1, e2, e_coup, e_nuc = make_energy(ks, mol, h1e, anecoup, bnecoup, adm, bdm)
    e_tot = e_tot + evec_pon
    print('Evec potential:', evec_pon)
    while not MICRO_conv and MICRO_cycle <= max(1, micro_cycle):
        energy_last, adm_last, bdm_last = e_tot.copy(), adm.copy(), bdm.copy()
        if cptype == 'HF':
            aU1, bU1 = ucphf.make_U(mol, aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ)
        elif cptype == 'HF-Field':
            aU1, bU1 = ucphf_field.make_U(mol, aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ)
        elif cptype == 'KS':
            aU1, bU1 = ucpks.make_U(ks, mol, aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ)
        aU1_trans = transgrad.trans(mol, aU1, DisVector)
        bU1_trans = transgrad.trans(mol, bU1, DisVector)
        aFKSeff, bFKSeff, anecoup, bnecoup, evec_pon = make_FKSeff(ks, mol, aCL, aCR, aU1, aU1_trans, amo_occ, bU1, bU1_trans, bCL, bCR, bmo_occ, grad_nuc, DisVector, mass)
        amo_energy, aCL, aCR = mathtool.geigen(aFKSeff.real, S)
        bmo_energy, bCL, bCR = mathtool.geigen(bFKSeff.real, S)
        MO_ENERGY = (amo_energy, bmo_energy)
        MO_COEFF = (aCR, bCR)
        amo_occ, bmo_occ = scf.uhf.UHF(mol).get_occ(MO_ENERGY, MO_COEFF)
        adm = mathtool.odm(aCL, aCR, amo_occ)
        bdm = mathtool.odm(bCL, bCR, bmo_occ)
        e_tot, e1, e2, e_coup, e_nuc = make_energy(ks, mol, h1e, anecoup, bnecoup, adm, bdm)
        e_tot = e_tot + evec_pon
        print('Evec potential:', evec_pon)
        if (abs(e_tot - energy_last) < micro_tol and norm((adm - adm_last)/adm.shape[0]) < sqrt(micro_tol)):
            MICRO_conv = True
        MICRO_cycle += 1
    return aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, e_tot, e1, e2, e_coup, e_nuc, MICRO_cycle

