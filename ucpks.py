#!/usr/bin/env python

import numpy
import pyscf.lib
from pyscf import gto, dft
from pyscf.hessian import uks as uks_hess
from pyscf.grad import uks as uks_grad
from NECSCF.DFT import integral, mathtool
from numpy.linalg import norm
from scipy.linalg.blas import zgemm, zgemv, zdotc
from NECSCF.Cphf import cphf


def make_XC(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_occ, bmo_occ):
    aCL, aCR = numpy.vstack((amocct,amvirt)), numpy.hstack((amocc, amvir))
    bCL, bCR = numpy.vstack((bmocct,bmvirt)), numpy.hstack((bmocc, bmvir))
    adm = mathtool.odm(aCL, aCR, amo_occ)
    bdm = mathtool.odm(bCL, bCR, bmo_occ)
    aVXC, bVXC = ks._numint.nr_uks(mol, ks.grids, ks.xc, (adm, bdm))[2]
    return aVXC, bVXC
    

def make_XC1(mol, amocc, bmocc, amvir, bmvir, amo_occ, bmo_occ, max_memory, atm_no, dofdes):
    hobj = dft.UKS(mol).Hessian()
    mo_coeff = numpy.array([numpy.hstack((amocc,amvir)), numpy.hstack((bmocc,bmvir))], dtype='float64')
    mo_occ = numpy.vstack((amo_occ, bmo_occ))
    mo_occ = numpy.array(mo_occ, dtype='float64')
    aXC1, bXC1 = uks_hess._get_vxc_deriv1(hobj, mo_coeff, mo_occ, max_memory)
    dict = {'x':0, 'y':1, 'z':2}
    aXC1, bXC1 = aXC1[atm_no][dict[dofdes]], bXC1[atm_no][dict[dofdes]]
    return aXC1, bXC1


def make_f1(mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_occ, bmo_occ, atm_no, dofdes):
    '''
    H^(1) + partial<ak||ik>^alpha over partial y + partial<ak|ik>^beta over partial y
    '''
    nmo, anocc = amocc.shape                                                                 # number of MOs and occupied MOs
    anvir = nmo - anocc                                                                      # number of virtual MOs
    bnocc = bmocc.shape[1]
    bnvir = nmo - bnocc
    # Atomic h^core(1)
    h1a = -(mol.intor('cint1e_ipkin_sph', comp=3) + mol.intor('cint1e_ipnuc_sph', comp=3))   # Get hcore
    h1a = integral.int1e3_dof(mol, h1a, atm_no, dofdes)                                      # Isolate hcore for specific atom and direction
    mol.set_rinv_origin(mol.atom_coord(atm_no))
    h1b = mol.intor('cint1e_iprinv_sph', comp=3)                                             # Get the gradient of nucleus-electron attraction wrt. Nuclear coords
    h1b = integral.intrinv3_dof(mol, h1b, dofdes)                                            # Isolate specific direction
    h1a = h1a - 1*mol.atom_charge(atm_no) * h1b                                              # Combine hcore(0) with hcore(1)
    h1a = h1a + h1a.T.conj()                                                                 # Make hcore
    ah1vo = zgemm(1.0, amvirt, zgemm(1.0, h1a, amocc)) #\mathscr{H^{(1)\alpha}_{ai}}         # H1_{ai}
    ah1ov = zgemm(1.0, amocct, zgemm(1.0, h1a, amvir)) #\mathscr{H^{(1)\alpha}_{ai}}         # H1_{ia}
    bh1vo = zgemm(1.0, bmvirt, zgemm(1.0, h1a, bmocc))
    bh1ov = zgemm(1.0, bmocct, zgemm(1.0, h1a, bmvir))

    g1 = -mol.intor('cint2e_ip1_sph',comp=3).reshape(3,nmo*nmo,nmo*nmo)                      # Gradient of the two-electron integrals between MOs
    g1 = cphf.dint2e_dof(mol, g1, atm_no, dofdes).reshape(nmo,nmo,nmo,nmo)                   ### Reshape LOOK MORE INTO
    #For F^{(1)\alpha}_{vo}
    aaG1vokk = cphf.mo_int2e(g1, (amvirt, amocc, amocct, amocc))                             ### Some matrix multiplication over MO pairs??
    aaJ1vo = numpy.trace(aaG1vokk,0,2,3)                    
    abG1vokk = cphf.mo_int2e(g1, (amvirt, amocc, bmocct, bmocc))
    abJ1vo = numpy.trace(abG1vokk,0,2,3)
    #For F^{(1)\alpha}_{ov}
    aaG1ovkk = cphf.mo_int2e(g1, (amocct, amvir, amocct, amocc))
    aaJ1ov = numpy.trace(aaG1ovkk,0,2,3)
    abG1ovkk = cphf.mo_int2e(g1, (amocct, amvir, bmocct, bmocc))
    abJ1ov = numpy.trace(abG1ovkk,0,2,3)
    #For F^{(1)\beta}_{vo}
    bbG1vokk = cphf.mo_int2e(g1, (bmvirt, bmocc, bmocct, bmocc))
    bbJ1vo = numpy.trace(bbG1vokk,0,2,3)
    baG1vokk = cphf.mo_int2e(g1, (bmvirt, bmocc, amocct, amocc))
    baJ1vo = numpy.trace(baG1vokk,0,2,3)
    #For F^{(1)\beta}_{ov}
    bbG1ovkk = cphf.mo_int2e(g1, (bmocct, bmvir, bmocct, bmocc))
    bbJ1ov = numpy.trace(bbG1ovkk,0,2,3)
    baG1ovkk = cphf.mo_int2e(g1, (bmocct, bmvir, amocct, amocc))
    baJ1ov = numpy.trace(baG1ovkk,0,2,3)
    
    # Get contributions from first order derivative of DFT exchange-correlation 
    aXC1, bXC1 = make_XC1(mol, amocc, bmocc, amvir, bmvir, amo_occ, bmo_occ, 1800, atm_no, dofdes)
    aXC1, bXC1 = numpy.array(aXC1, dtype='complex128'), numpy.array(bXC1, dtype='complex128')
    aaXC1vo = zgemm(1.0, amvirt, zgemm(1.0, aXC1, amocc))
    aaXC1ov = zgemm(1.0, amocct, zgemm(1.0, aXC1, amvir))
    bbXC1vo = zgemm(1.0, bmvirt, zgemm(1.0, bXC1, bmocc))
    bbXC1ov = zgemm(1.0, bmocct, zgemm(1.0, bXC1, bmvir))
    # I believe cross-terms e.g. abXC1vo,... are not needed here?

    af1vo = ah1vo + aaJ1vo + aaXC1vo + abJ1vo                                             # Form Fock matrix from Hcore, J and K
    bf1vo = bh1vo + bbJ1vo + bbXC1vo + baJ1vo
    af1ov = ah1ov + aaJ1ov + aaXC1ov + abJ1ov
    bf1ov = bh1ov + bbJ1ov + bbXC1ov + baJ1ov

    return af1vo, bf1vo, af1ov, bf1ov

def make_B0(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_energy, bmo_energy, amo_occ, bmo_occ, atm_no, dofdes):
    ###  SEE EQ. 38 in Docs  ###
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    bnocc = bmocc.shape[1]
    bnvir = nmo - bnocc
    ae_o = amo_energy[amo_occ>0]                                                          # Occupied alpha MO energies
    ae_v = amo_energy[amo_occ==0]
    be_o = bmo_energy[bmo_occ>0]
    be_v = bmo_energy[bmo_occ==0]
    ae_vo = -1 / pyscf.lib.direct_sum('v-o->vo', ae_v, ae_o)                              # Energy differences between occupied and virtual MOs
    ae_ov = -1 / pyscf.lib.direct_sum('o-v->ov', ae_o, ae_v)
    be_vo = -1 / pyscf.lib.direct_sum('v-o->vo', be_v, be_o)
    be_ov = -1 / pyscf.lib.direct_sum('o-v->ov', be_o, be_v)

    s1 = -integral.S_a0(mol)                                                              # Derivative of overlap matrix wrt. Nuclear coords
    s1 = integral.int1e3_dof(mol, s1, atm_no, dofdes)                                     # Isolate S_a0 for specific atom and direction
    s1 = s1 + s1.T.conj()
    as1vo = zgemm(1.0, amvirt, zgemm(1.0, s1, amocc))                                     # Components of S1 matrix in various MO pairs...
    as1ov = zgemm(1.0, amocct, zgemm(1.0, s1, amvir))
    as1oc = zgemm(1.0, amocct, zgemm(1.0, s1, numpy.hstack((amocc,amvir))))
    as1ocT = zgemm(1.0, numpy.vstack((amocct,amvirt)), zgemm(1.0, s1, amocc))
    bs1vo = zgemm(1.0, bmvirt, zgemm(1.0, s1, bmocc))
    bs1ov = zgemm(1.0, bmocct, zgemm(1.0, s1, bmvir))
    bs1oc = zgemm(1.0, bmocct, zgemm(1.0, s1, numpy.hstack((bmocc,bmvir))))
    bs1ocT = zgemm(1.0, numpy.vstack((bmocct,bmvirt)), zgemm(1.0, s1, bmocc))

    g0 = mol.intor('cint2e_sph').reshape(nmo,nmo,nmo,nmo)                                 # Standard 2-electron integrals
    #For B0^{\alpha}_{vo}
    aaG1vock = cphf.mo_int2e(g0, (amvirt,amocc,numpy.vstack((amocct,amvirt)),amocc))
    aaJ1vo = numpy.einsum('ij,voji->vo', as1oc, aaG1vock)                                                                                     # Add Vxc1 * S
    abG1vock = cphf.mo_int2e(g0, (amvirt,amocc,numpy.vstack((bmocct,bmvirt)),bmocc))
    abJ1vo = numpy.einsum('ij,voji->vo', bs1oc, abG1vock)                                 
    #For B0^{\alpha}_{ov}
    aaG1ovck = cphf.mo_int2e(g0, (amocct,amvir,numpy.vstack((amocct,amvirt)),amocc))
    aaJ1ov = numpy.einsum('ij,ovji->ov', as1oc, aaG1ovck)
    abG1ovck = cphf.mo_int2e(g0, (amocct,amvir,numpy.vstack((bmocct,bmvirt)),bmocc))
    abJ1ov = numpy.einsum('ij,ovji->ov', bs1oc, abG1ovck)
    #For B0^{\beta}_{vo}
    bbG1vock = cphf.mo_int2e(g0, (bmvirt,bmocc,numpy.vstack((bmocct,bmvirt)),bmocc))
    bbJ1vo = numpy.einsum('ij,voji->vo', bs1oc, bbG1vock)
    baG1vock = cphf.mo_int2e(g0, (bmvirt,bmocc,numpy.vstack((amocct,amvirt)),amocc))
    baJ1vo = numpy.einsum('ij,voji->vo', as1oc, baG1vock)
    #For B0^{\beta}_{ov}
    bbG1ovck = cphf.mo_int2e(g0, (bmocct,bmvir,numpy.vstack((bmocct,bmvirt)),bmocc))
    bbJ1ov = numpy.einsum('ij,ovji->ov', bs1oc, bbG1ovck)
    baG1ovck = cphf.mo_int2e(g0, (bmocct,bmvir,numpy.vstack((amocct,amvirt)),amocc))
    baJ1ov = numpy.einsum('ij,ovji->ov', as1oc, baG1ovck)

    # Get contributions from DFT exchange-correlation 
 
    aVXC, bVXC = make_XC(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_occ, bmo_occ)
    aaXC_vm = zgemm(1.0, amvirt, zgemm(1.0, aVXC, numpy.hstack((amocc,amvir))))
    aaXCvo = zgemm(1.0, aaXC_vm, as1ocT)
    aaXC_mv = zgemm(1.0, numpy.vstack((amocct,amvirt)), zgemm(1.0, aVXC, amvir))
    aaXCov = zgemm(1.0, as1oc, aaXC_mv)
    bbXC_vm = zgemm(1.0, bmvirt, zgemm(1.0, bVXC, numpy.hstack((bmocc,bmvir))))
    bbXCvo = zgemm(1.0, bbXC_vm, bs1ocT)
    bbXC_mv = zgemm(1.0, numpy.vstack((bmocct,bmvirt)), zgemm(1.0, bVXC, bmvir))
    bbXCov = zgemm(1.0, as1oc, bbXC_mv)

    as1evo = numpy.einsum('vo,o->vo', as1vo, ae_o)
    as1eov = numpy.einsum('ov,v->ov', as1ov, ae_v)
    bs1evo = numpy.einsum('vo,o->vo', bs1vo, be_o)
    bs1eov = numpy.einsum('ov,v->ov', bs1ov, be_v)

    af1vo, bf1vo, af1ov, bf1ov = make_f1(mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_occ, bmo_occ, atm_no, dofdes)
    af1vo = af1vo - as1evo - aaJ1vo - aaXCvo - abJ1vo                              
    af1ov = af1ov - as1eov - aaJ1ov - aaXCov - abJ1ov
    bf1vo = bf1vo - bs1evo - bbJ1vo - bbXCvo - baJ1vo
    bf1ov = bf1ov - bs1eov - bbJ1ov - bbXCov - baJ1ov
    '''
    print('######  F1 vo  ######')
    print(af1vo)
    print('######  S1e vo  ######')
    print(as1evo)
    print('######  aaJ1 vo  ######')
    print(aaJ1vo)
    print('######  abJ1 vo  ######')
    print(abJ1vo)
    
    print('######  aXC1 vo  ######')
    print(aaXCvo)
    print('######  aXC1 ov  ######')
    print(aaXCov)
    print('######  aXC1 vo  ######')
    print(bbXCvo)
    print('######  aXC1 vo  ######')
    print(bbXCov)
    '''
    return af1vo*ae_vo, af1ov*ae_ov, bf1vo*be_vo, bf1ov*be_ov


####  solve_mo1() is preferred over psolve_mo1()  ####
def psolve_mo1(mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_energy, bmo_energy, amo_occ, bmo_occ, aB0vo, aB0ov, bB0vo, bB0ov, rms_tol, maxcycle):
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    bnocc = bmocc.shape[1]
    bnvir = nmo - bnocc
    ae_o = amo_energy[amo_occ>0]
    ae_v = amo_energy[amo_occ==0]
    be_o = bmo_energy[bmo_occ>0]
    be_v = bmo_energy[bmo_occ==0]
    ae_vo = -1 / pyscf.lib.direct_sum('v-o->vo', ae_v, ae_o)
    ae_ov = -1 / pyscf.lib.direct_sum('o-v->ov', ae_o, ae_v)
    be_vo = -1 / pyscf.lib.direct_sum('v-o->vo', be_v, be_o)
    be_ov = -1 / pyscf.lib.direct_sum('o-v->ov', be_o, be_v)

    G0 = mol.intor('cint2e_sph').reshape(nmo,nmo,nmo,nmo)
    def make_AU(aUvo, aUov, bUvo, bUov):                                   # Obtain A matrix
        admov = -zgemm(1.0, amocc, zgemm(1.0, aUov, amvirt))               # Getting OV Density Matrix
        aJov = numpy.einsum('rs,pqsr->pq', admov, G0)                      # Get coulomb energy
                                                                           # Replace with Vxc1...
        admvo = zgemm(1.0, amvir, zgemm(1.0, aUvo, amocct))
        aJvo = numpy.einsum('rs,pqsr->pq', admvo, G0)
                                                                           # Replace with Vxc1...
        bdmov = -zgemm(1.0, bmocc, zgemm(1.0, bUov, bmvirt))               
        bJov = numpy.einsum('rs,pqsr->pq', bdmov, G0)
        bdmvo = zgemm(1.0, bmvir, zgemm(1.0, bUvo, bmocct))
        bJvo = numpy.einsum('rs,pqsr->pq', bdmvo, G0)
        aAUvo = zgemm(1.0, amvirt, zgemm(1.0, (aJov + aJvo + bJov + bJvo), amocc)) * ae_vo   # Get A_{vo} matrix
        #For AU^{\alpha}_{ov}
        aAUov = zgemm(1.0, amocct, zgemm(1.0, (aJov + aJvo + bJov + bJvo), amvir)) * ae_ov
        #For AU^{\beta}_{vo}
                                                                           # Replace with Vxc1...
                                                                           # Replace with Vxc1...
        bAUvo = zgemm(1.0, bmvirt, zgemm(1.0, (bJov + bJvo + aJov + aJvo), bmocc)) * be_vo
        #For AU^{\beta}_{ov}
        bAUov = zgemm(1.0, bmocct, zgemm(1.0, (bJov + bJvo + aJov + aJvo), bmvir)) * be_ov
        return aAUvo + aB0vo, aAUov + aB0ov, bAUvo + bB0vo, bAUov + bB0ov

    aUvo, aUov, bUvo, bUov = make_AU(aB0vo, aB0ov, bB0vo, bB0ov)
    ncycle = 0
    cphf_conv = False

    #print '******** Entering CPHF Iteration *********'
    while not cphf_conv and ncycle <= max(1, maxcycle):
        aUvo_last, aUov_last, bUvo_last, bUov_last = aUvo, aUov, bUvo, bUov
        aUvo, aUov, bUvo, bUov = make_AU(aUvo_last, aUov_last, bUvo_last, bUov_last)
        if norm(aUvo - aUvo_last) < rms_tol and norm(aUov - aUov_last) < rms_tol and norm(bUvo - bUvo_last) < rms_tol and norm(bUov - bUov_last) < rms_tol:
            cphf_conv = True
        ncycle += 1
    return aUvo, aUov, bUvo, bUov

def solve_mo1(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_energy, bmo_energy, \
                   amo_occ, bmo_occ, aB0vo, aB0ov, bB0vo, bB0ov, rms_tol, maxcycle, lindep):
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    bnocc = bmocc.shape[1]
    bnvir = nmo - bnocc
    ae_o = amo_energy[amo_occ>0]
    ae_v = amo_energy[amo_occ==0]
    be_o = bmo_energy[bmo_occ>0]
    be_v = bmo_energy[bmo_occ==0]
    ae_vo = -1 / pyscf.lib.direct_sum('v-o->vo', ae_v, ae_o)
    ae_ov = -1 / pyscf.lib.direct_sum('o-v->ov', ae_o, ae_v)
    be_vo = -1 / pyscf.lib.direct_sum('v-o->vo', be_v, be_o)
    be_ov = -1 / pyscf.lib.direct_sum('o-v->ov', be_o, be_v)

    g0 = mol.intor('cint2e_sph').reshape(nmo,nmo,nmo,nmo)

    def AU(ks, U):       # Obtain A matrices
        aUvo, aUov, bUvo, bUov = U[:anocc*anvir].reshape(anvir,anocc), U[anocc*anvir:2*anocc*anvir].reshape(anocc,anvir), \
            U[2*anocc*anvir:2*anocc*anvir+bnocc*bnvir].reshape(bnvir,bnocc), U[2*anocc*anvir+bnocc*bnvir:].reshape(bnocc,bnvir)
        
        #A_vo
        aDMvo =  zgemm(1.0, amvir, zgemm(1.0, aUvo, amocct))
        aJvo = numpy.einsum('rs,pqsr->pq', aDMvo, g0) 
        #A_ov
        aDMov = -zgemm(1.0, amocc, zgemm(1.0, aUov, amvirt))
        aJov = numpy.einsum('rs,pqsr->pq', aDMov, g0)
        #B_vo
        bDMvo =  zgemm(1.0, bmvir, zgemm(1.0, bUvo, bmocct))
        bJvo = numpy.einsum('rs,pqsr->pq', bDMvo, g0) 
        #B_ov
        bDMov = -zgemm(1.0, bmocc, zgemm(1.0, bUov, bmvirt))
        bJov = numpy.einsum('rs,pqsr->pq', bDMov, g0)
      
        dm0 = (zgemm(1.0, amocc, amocct), zgemm(1.0, bmocc, bmocct))
        abXCvo, baXCvo = ks._numint.nr_uks_fxc(mol, ks.grids, ks.xc, dm0, dms = (aDMvo.real, bDMvo.real))
        abXCov, baXCov = ks._numint.nr_uks_fxc(mol, ks.grids, ks.xc, dm0, dms = (aDMov.real, bDMov.real))

        '''
        aXCmo = zgemm(1.0, numpy.vstack((amocct, amvirt)), aVXC, numpy.hstack((amocc, amvir)))
        aXCvo = zgemm(1.0, aDMvo, aXCmo)
        aXCov = zgemm(1.0, aDMov, aXCmo)
        bXCmo = zgemm(1.0, numpy.vstack((bmocct, bmvirt)), bVXC, numpy.hstack((bmocc, bmvir)))
        bXCvo = zgemm(1.0, bDMvo, bXCmo)
        bXCov = zgemm(1.0, bDMov, bXCmo)
       
        print('#########  abXCvo  #########')
        print(numpy.linalg.norm(abXCvo))
        print('#########  baXCvo  #########')
        print(numpy.linalg.norm(baXCvo))
        print('#########  abXCov  #########')
        print(numpy.linalg.norm(abXCov))
        print('#########  baXCov  #########')
        print(numpy.linalg.norm(baXCov))
        '''
        # Form AUs
        aAUvo = zgemm(1.0, amvirt, zgemm(1.0, (aJov + abXCov + baXCov + aJvo + abXCvo + baXCvo + bJov + bJvo), amocc)) * ae_vo
        aAUov = zgemm(1.0, amocct, zgemm(1.0, (aJov + abXCov + baXCov + aJvo + abXCvo + baXCvo + bJov + bJvo), amvir)) * ae_ov
        bAUvo = zgemm(1.0, bmvirt, zgemm(1.0, (bJov + baXCov + abXCov + bJvo + abXCvo + baXCvo + aJov + aJvo), bmocc)) * be_vo
        bAUov = zgemm(1.0, bmocct, zgemm(1.0, (bJov + baXCov + abXCov + bJvo + abXCvo + baXCvo + aJov + aJvo), bmvir)) * be_ov
        U = numpy.concatenate((aAUvo.ravel(), aAUov.ravel(), bAUvo.ravel(), bAUov.ravel()))
        return U.ravel()
    B0 = numpy.concatenate((aB0vo.ravel(), aB0ov.ravel(), bB0vo.ravel(), bB0ov.ravel()))
    maxcycle = min(maxcycle, B0.size)
    h = numpy.zeros((maxcycle, maxcycle), dtype='complex128')
    Us = [B0.ravel()]
    AUs = [AU(ks, Us[-1])]
    cycle = 0
    cphf_conv = False


    if abs(zdotc(Us[0].conj(), Us[0])) < rms_tol:
        x = Us[0]
    else:
        print ('******** Entering CPHF Iteration *********')
        while not cphf_conv and cycle <= max(1, maxcycle):
            U = AUs[-1].copy()
            for i in range(cycle+1):
                s12 = h[i, cycle] = -1 * zdotc(Us[i].conj(), AUs[-1])
                U += (s12/zdotc(Us[i].conj(), Us[i])) * Us[i]
            h[cycle, cycle] += zdotc(Us[cycle].conj(), Us[cycle])
            Us.append(U)
            AUs.append(AU(ks, U))
            test = zdotc(Us[-1].conj(), Us[-1])
            #print abs(test)
            if abs(test) < rms_tol**2 or abs(test) < lindep:
                cphf_conv = True
                nd = cycle + 1
                for i in range(nd):
                    for j in range(i):
                        h[i,j] = -1 * zdotc(Us[i].conj(), AUs[j])
                g = numpy.zeros(nd, dtype='complex128')
                g[0] = zdotc(Us[0].conj(), Us[0])
                c = numpy.linalg.solve(h[:nd,:nd], g)
                x = c[0] * Us[0]
                for i in range(1, len(c)):
                    x += c[i] * Us[i]
            print ('Iteration %-3d, CPHF convergence: %.15g, %r.' \
                        % ((cycle + 1), abs(test), cphf_conv))
            cycle += 1
    aUvo, aUov, bUvo, bUov = x[:anocc*anvir].reshape(anvir,anocc), x[anocc*anvir:2*anocc*anvir].reshape(anocc,anvir), \
        x[2*anocc*anvir:2*anocc*anvir+bnocc*bnvir].reshape(bnvir,bnocc), x[2*anocc*anvir+bnocc*bnvir:].reshape(bnocc,bnvir)
    return aUvo, aUov, bUvo, bUov

# bmocc = 0 Special Case 1
def make_f1_1(mol, amocc, amocct, amvir, amvirt, atm_no, dofdes):
    '''
    H^(1) + partial<ak||ik>^alpha over partial y + partial<ak|ik>^beta over partial y
    '''
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    # Atomic h^core(1)
    h1a = -(mol.intor('cint1e_ipkin_sph', comp=3) + mol.intor('cint1e_ipnuc_sph', comp=3))
    h1a = integral.int1e3_dof(mol, h1a, atm_no, dofdes)
    mol.set_rinv_origin(mol.atom_coord(atm_no))
    h1b = mol.intor('cint1e_iprinv_sph', comp=3)
    h1b = integral.intrinv3_dof(mol, h1b, dofdes)
    h1a = h1a - 1*mol.atom_charge(atm_no) * h1b
    h1a = h1a + h1a.T.conj()
    ah1vo = zgemm(1.0, amvirt, zgemm(1.0, h1a, amocc)) #\mathscr{H^{(1)\alpha}_{ai}}
    ah1ov = zgemm(1.0, amocct, zgemm(1.0, h1a, amvir)) #\mathscr{H^{(1)\alpha}_{ai}}

    g1 = -mol.intor('cint2e_ip1_sph',comp=3).reshape(3,nmo*nmo,nmo*nmo)
    g1 = cphf.dint2e_dof(mol, g1, atm_no, dofdes).reshape(nmo,nmo,nmo,nmo)
    #For F^{(1)\alpha}_{vo}
    aaG1vokk = cphf.mo_int2e(g1, (amvirt, amocc, amocct, amocc))
    aaJ1vo = numpy.trace(aaG1vokk,0,2,3)
    aaK1vo = numpy.trace(aaG1vokk,0,1,2)
    #For F^{(1)\alpha}_{ov}
    aaG1ovkk = cphf.mo_int2e(g1, (amocct, amvir, amocct, amocc))
    aaJ1ov = numpy.trace(aaG1ovkk,0,2,3)
    aaG1okkv = cphf.mo_int2e(g1, (amocct, amocc, amocct, amvir))
    aaK1ov = numpy.trace(aaG1okkv,0,1,2)

    af1vo = ah1vo + (aaJ1vo - aaK1vo)
    af1ov = ah1ov + (aaJ1ov - aaK1ov)
    return af1vo, af1ov

def make_B0_1(mol, amocc, amocct, amvir, amvirt, amo_energy, amo_occ, atm_no, dofdes):
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    ae_o = amo_energy[amo_occ>0]
    ae_v = amo_energy[amo_occ==0]
    ae_vo = -1 / pyscf.lib.direct_sum('v-o->vo', ae_v, ae_o)
    ae_ov = -1 / pyscf.lib.direct_sum('o-v->ov', ae_o, ae_v)

    s1 = -integral.S_a0(mol)
    s1 = integral.int1e3_dof(mol, s1, atm_no, dofdes)
    s1 = s1 + s1.T.conj()
    as1vo = zgemm(1.0, amvirt, zgemm(1.0, s1, amocc))
    as1ov = zgemm(1.0, amocct, zgemm(1.0, s1, amvir))
    as1oc = zgemm(1.0, amocct, zgemm(1.0, s1, numpy.hstack((amocc,amvir))))

    g0 = mol.intor('cint2e_sph').reshape(nmo,nmo,nmo,nmo)
    #For B0^{\alpha}_{vo}
    aaG1vock = cphf.mo_int2e(g0, (amvirt,amocc,numpy.vstack((amocct,amvirt)),amocc))
    aaJ1vo = numpy.einsum('ij,voji->vo', as1oc, aaG1vock)
    aaK1vo = numpy.einsum('ij,vijo->vo', as1oc, aaG1vock)
    #For B0^{\alpha}_{ov}
    aaG1ovck = cphf.mo_int2e(g0, (amocct,amvir,numpy.vstack((amocct,amvirt)),amocc))
    aaJ1ov = numpy.einsum('ij,ovji->ov', as1oc, aaG1ovck)
    aaG1okcv = cphf.mo_int2e(g0, (amocct,amocc,numpy.vstack((amocct,amvirt)),amvir))
    aaK1ov = numpy.einsum('ij,oijv->ov', as1oc, aaG1okcv)

    as1evo = numpy.einsum('vo,o->vo', as1vo, ae_o)
    as1eov = numpy.einsum('ov,v->ov', as1ov, ae_v)

    af1vo, af1ov = make_f1_1(mol, amocc, amocct, amvir, amvirt, atm_no, dofdes)
    af1vo = af1vo - as1evo - (aaJ1vo - aaK1vo)
    af1ov = af1ov - as1eov - (aaJ1ov - aaK1ov)
    return af1vo*ae_vo, af1ov*ae_ov

def psolve_mo1_1(mol, amocc, amocct, amvir, amvirt, amo_energy, amo_occ, aB0vo, aB0ov, rms_tol, maxcycle):
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    ae_o = amo_energy[amo_occ>0]
    ae_v = amo_energy[amo_occ==0]
    ae_vo = -1 / pyscf.lib.direct_sum('v-o->vo', ae_v, ae_o)
    ae_ov = -1 / pyscf.lib.direct_sum('o-v->ov', ae_o, ae_v)

    G0 = mol.intor('cint2e_sph').reshape(nmo,nmo,nmo,nmo)
    def make_AU_1(aUvo, aUov):
        admov = -zgemm(1.0, amocc, zgemm(1.0, aUov, amvirt))
        aJov = numpy.einsum('rs,pqsr->pq', admov, G0)
        aKov = numpy.einsum('rs,prsq->pq', admov, G0)
        admvo = zgemm(1.0, amvir, zgemm(1.0, aUvo, amocct))
        aJvo = numpy.einsum('rs,pqsr->pq', admvo, G0)
        aKvo = numpy.einsum('rs,prsq->pq', admvo, G0)
        aAUvo = zgemm(1.0, amvirt, zgemm(1.0, (aJov - aKov + aJvo - aKvo), amocc)) * ae_vo
        #For AU^{\alpha}_{ov}
        aAUov = zgemm(1.0, amocct, zgemm(1.0, (aJov - aKov + aJvo - aKvo), amvir)) * ae_ov
        return aAUvo + aB0vo, aAUov + aB0ov

    aUvo, aUov = make_AU_1(aB0vo, aB0ov)
    ncycle = 0
    cphf_conv = False

    #print '******** Entering CPHF Iteration *********'
    while not cphf_conv and ncycle <= max(1, maxcycle):
        aUvo_last, aUov_last = aUvo, aUov
        aUvo, aUov = make_AU_1(aUvo_last, aUov_last)
        #print 'Iteration %-3d, rms difference = %.15g, %.15g.' \
        #             % (ncycle + 1, norm(aUvo - aUvo_last), norm(aUov - aUov_last))
        if norm(aUvo - aUvo_last) < rms_tol and norm(aUov - aUov_last) < rms_tol:
            cphf_conv = True
        ncycle += 1
    return aUvo, aUov

def make_U(ks, mol, aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, rms_tol=1.0e-15, maxcycle=200):
    lindep=1.0e-15
    natm = mol.natm
    atmlst = range(mol.natm)

    amocc = aCR[:,amo_occ>0]
    amocct = aCL[amo_occ>0,:]
    amvir = aCR[:,amo_occ==0]
    amvirt = aCL[amo_occ==0,:]
    bmocc = bCR[:,bmo_occ>0]
    bmocct = bCL[bmo_occ>0,:]
    bmvir = bCR[:,bmo_occ==0]
    bmvirt = bCL[bmo_occ==0,:]
    nmo, anocc = amocc.shape
    anvir = nmo - anocc
    bnocc = bmocc.shape[1]
    bnvir = nmo - bnocc
    dof_des = ['x', 'y', 'z']
    #Calculate independent pairs
    if bmocc.shape[1] > 0: 
        aUvo = numpy.zeros((len(dof_des),natm,anvir,anocc), dtype='complex128')
        aUov = numpy.zeros((len(dof_des),natm,anocc,anvir), dtype='complex128')
        bUvo = numpy.zeros((len(dof_des),natm,bnvir,bnocc), dtype='complex128')
        bUov = numpy.zeros((len(dof_des),natm,bnocc,bnvir), dtype='complex128')
        for i in range(len(dof_des)):
            for j in atmlst:
                aB0vo, aB0ov, bB0vo, bB0ov = make_B0(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_energy, bmo_energy, amo_occ, bmo_occ, j, dof_des[i])
                auvo, auov, buvo, buov = solve_mo1(ks, mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_energy, bmo_energy, amo_occ, bmo_occ, aB0vo, aB0ov, bB0vo, bB0ov, rms_tol, maxcycle,lindep)
                aUvo[i][j], aUov[i][j], bUvo[i][j], bUov[i][j] = auvo, auov, buvo, buov
        #Full U
        s1 = -integral.S_a0(mol)
        aU = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
        bU = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
        for i in range(len(dof_des)):
            for j in atmlst:
                s1ao = integral.int1e3_dof(mol, s1, j, dof_des[i])
                s1ao = s1ao + s1ao.T
                as1mo = zgemm(1.0, aCL, zgemm(1.0, s1ao, aCR))
                aU[i][j][0:anocc,0:anocc]     = -.5*as1mo[0:anocc,0:anocc]
                aU[i][j][0:anocc,anocc:nmo]   = aUov[i][j]
                aU[i][j][anocc:nmo,0:anocc]   = aUvo[i][j]
                aU[i][j][anocc:nmo,anocc:nmo] = -.5*as1mo[anocc:nmo,anocc:nmo]
                bs1mo = zgemm(1.0, bCL, zgemm(1.0, s1ao, bCR))
                bU[i][j][0:bnocc,0:bnocc]     = -.5*bs1mo[0:bnocc,0:bnocc]
                bU[i][j][0:bnocc,bnocc:nmo]   = bUov[i][j]
                bU[i][j][bnocc:nmo,0:bnocc]   = bUvo[i][j]
                bU[i][j][bnocc:nmo,bnocc:nmo] = -.5*bs1mo[bnocc:nmo,bnocc:nmo]
    if bmocc.shape[1] == 0:
        aUvo = numpy.zeros((len(dof_des),natm,anvir,anocc), dtype='complex128')
        aUov = numpy.zeros((len(dof_des),natm,anocc,anvir), dtype='complex128')
        for i in range(len(dof_des)):
            for j in atmlst:    
                aB0vo, aB0ov = make_B0_1(mol, amocc, amocct, amvir, amvirt, amo_energy, amo_occ, j, dof_des[i])
                auvo, auov = psolve_mo1_1(mol, amocc, amocct, amvir, amvirt, amo_energy, amo_occ, aB0vo, aB0ov, rms_tol, maxcycle)
                aUvo[i][j], aUov[i][j] = auvo, auov
        #Full U
        s1 = -integral.S_a0(mol)
        aU = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
        bU = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
        for i in range(len(dof_des)):
            for j in atmlst:
                s1ao = integral.int1e3_dof(mol, s1, j, dof_des[i])
                s1ao = s1ao + s1ao.T
                as1mo = zgemm(1.0, aCL, zgemm(1.0, s1ao, aCR))
                aU[i][j][0:anocc,0:anocc]     = -.5*as1mo[0:anocc,0:anocc]
                aU[i][j][0:anocc,anocc:nmo]   = aUov[i][j]
                aU[i][j][anocc:nmo,0:anocc]   = aUvo[i][j]
                aU[i][j][anocc:nmo,anocc:nmo] = -.5*as1mo[anocc:nmo,anocc:nmo]
                bs1mo = zgemm(1.0, bCL, zgemm(1.0, s1ao, bCR))
                bU[i][j] = -.5*bs1mo
    return aU, bU

def make_U1T(mol, aCL, bCL, aCR, bCR, aU, bU):
    s1 = -integral.S_a0(mol)
    dof_des = ['x', 'y', 'z']
    nmo = aCL.shape[0]
    atmlst = range(mol.natm)
    aUT = numpy.zeros((len(dof_des),mol.natm,nmo,nmo), dtype='complex128')
    bUT = numpy.zeros((len(dof_des),mol.natm,nmo,nmo), dtype='complex128')
    for i in range(len(dof_des)):
      for j in atmlst:
          s1ao = integral.int1e3_dof(mol, s1, j, dof_des[i])
          s1ao = s1ao + s1ao.T
          as1mo = zgemm(1.0, aCL, zgemm(1.0, s1ao, aCR))
          aUT[i][j] = -aU[i][j] - as1mo
          bs1mo = zgemm(1.0, bCL, zgemm(1.0, s1ao, bCR))
          bUT[i][j] = -bU[i][j] - bs1mo
    return aUT, bUT

def make_C1(mol, aCR, bCR, aU, bU):
    natm = mol.natm
    nmo = aCR.shape[0]
    atmlst = range(mol.natm)
    dof_des = ['x', 'y', 'z']
    aC1 = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
    bC1 = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
    for i in range(len(dof_des)):
        for j in atmlst:
            aC1[i][j] = zgemm(1.0, aCR, aU[i][j])
            bC1[i][j] = zgemm(1.0, bCR, bU[i][j])
    return aC1, bC1

def make_C1T(mol, aCL, bCL, aUT, bUT):
    natm = mol.natm
    nmo = CL.shape[0]
    atmlst = range(mol.natm)
    dof_des = ['x', 'y', 'z']
    aC1T = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
    bC1T = numpy.zeros((len(dof_des),natm,nmo,nmo), dtype='complex128')
    for i in range(len(dof_des)):
        for j in atmlst:
            aC1T[i][j] = zgemm(1.0, aUT[i][j], aCL)
            bC1T[i][j] = zgemm(1.0, bUT[i][j], bCL)
    return aC1T, bC1T

#***********************************************************************
if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1.1'
    mol.basis = '631g'
    mol.build()
    #ks = dft.uks.UKS(mol)
    #ks.xc = 'pbe'
    #ks.kernel()
    '''
    mo_coeff, mo_energy, mo_occ = ks.mo_coeff, ks.mo_energy, ks.mo_occ
    amo_energy, bmo_energy = mo_energy
    aCL, bCL, aCR, bCR = mo_coeff[0].T, mo_coeff[1].T, mo_coeff[0], mo_coeff[1]
    amo_occ, bmo_occ = mo_occ[0], mo_occ[1]
    amocc = aCR[:,amo_occ>0]
    amocct = aCL[amo_occ>0,:]
    amvir = aCR[:,amo_occ==0]
    amvirt = aCL[amo_occ==0,:]
    bmocc = bCR[:,bmo_occ>0]
    bmocct = bCL[bmo_occ>0,:]
    bmvir = bCR[:,bmo_occ==0]
    bmvirt = bCL[bmo_occ==0,:]
    adm = mathtool.odm(aCL, aCR, amo_occ)
    bdm = mathtool.odm(bCL, bCR, bmo_occ)
    
    atm_no, dofdes = 0, 'x'

    vxc = ks._numint.nr_uks(mol, ks.grids, ks.xc, dms = (adm, bdm))[2]

    vxc1 = make_XC(mol, amocc, bmocc, amocct, bmocct, amvir, bmvir, amvirt, bmvirt, amo_occ, bmo_occ)
    
    aXC1, bXC1 = make_XC1(mol, amocc, bmocc, amvir, bmvir, amo_occ, bmo_occ, 1800, atm_no, dofdes)
    aXC1, bXC1 = numpy.array(aXC1, dtype='complex128'), numpy.array(bXC1, dtype='complex128')
    aaXC1vo = zgemm(1.0, amvirt, zgemm(1.0, aXC1, amocc))
    '''
  
    #print(vxc)
    #print('########################################')
    #print(vxc1)

    print(mol.nao)


    #make_U(mol, aCL, bCL, aCR, bCR, amo_energy, bmo_energy, amo_occ, bmo_occ, rms_tol=1.0e-15, maxcycle=200)



    #print(dR_vxc[0][0], dR_vxc[1][0])

