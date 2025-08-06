#!/usr/bin/env python

'''
Math tools for the Electron-Nuclear Coupled Self-Consistent Field
'''

import numpy
import scipy
from numpy.linalg import cholesky
from scipy.linalg import lu, inv, eigh, norm
from scipy.linalg.blas import zgemm, zgemv
from scipy.linalg.lapack import zggev
from pyscf import lib

#def geigen(F, S):
#    d, t = numpy.linalg.eigh(S)
#    s = t / numpy.sqrt(d)
#    xFx = reduce(numpy.dot,(s.T, F, s))
#    aop = lambda xs: [numpy.dot(xFx,x) for x in xs]
#   precond = lambda dx, e, x0: dx/(xFx.diagonal()-e)
#    x0 = xFx.diagonal()
#    n = xFx.shape[0]
#    conv, e, vl, vr = lib.linalg_helper.davidson_nosym1(aop, x0, precond, nroots=n, left=True)
#    CL = numpy.zeros((n,n),dtype='complex128')
#    CR = numpy.zeros((n,n),dtype='complex128')
#    for i in range(n):
#        CL[i,:] = vl[i]
#        CR[:,i] = vr[i]
#    CR = numpy.dot(s, CR)
#    CL = numpy.dot(CL, s.T)
#    return e, CL, CR

#def geigen(F, S):
#    E, L, R = eig(F, S, left=True)
#    L = L.T.conj()
#    idx = E.argsort()
#    E = E[idx]
#    R = R[:,idx]
#    L = L[idx,:]
#    D = zgemm(1.0, L, zgemm(1.0, S, R))
#    ML, MR = lu(D, permute_l=True)
#    NL = zgemm(1.0, inv(ML), L)
#    NR = zgemm(1.0, R, inv(MR))
#    return E, NL, NR

def geigen(F, S):
    L = cholesky(S)
    X = inv(L).dot(F).dot(inv(L).T)
    E, XR = numpy.linalg.eig(X)
    idx = E.argsort()
    E = E[idx]
    XR = XR[:,idx]
    XR = inv(L).T.dot(XR)

    F1 = inv(L).dot(F.T).dot(inv(L).T)
    E1, XL = numpy.linalg.eig(F1)
    idx = E1.conj().argsort()
    XL = XL[:,idx]
    XL = inv(L).T.dot(XL)
    XL = XL.T.conj()

    D = zgemm(1.0, XL, zgemm(1.0, S, XR))
    ML, MR = lu(D, permute_l=True)
    NL = zgemm(1.0, inv(ML), XL)
    NR = zgemm(1.0, XR, inv(MR))
    return E, NL, NR


def geigen_sort(F, S):
    L = cholesky(S)
    X = inv(L).dot(F).dot(inv(L).T)
    E, XR = numpy.linalg.eig(X)
    idx = E.argsort()
    E = E[idx]
    XR = XR[:,idx]
    XR = inv(L).T.dot(XR)

    F1 = inv(L).dot(F.T).dot(inv(L).T)
    E1, XL = numpy.linalg.eig(F1)
    idx = E1.conj().argsort()
    XL = XL[:,idx]
    XL = inv(L).T.dot(XL)
    XL = XL.T.conj()

    D = zgemm(1.0, XL, zgemm(1.0, S, XR))
    ML, MR = lu(D, permute_l=True)
    NL = zgemm(1.0, inv(ML), XL)
    NR = zgemm(1.0, XR, inv(MR))
    return E, NL, NR, idx


def eigen(F):
    E, L, R = scipy.linalg.eig(F, left=True)
    L = L.T.conj()
    idx = E.argsort()
    E = E[idx]
    R = R[:,idx]
    L = L[idx,:]
    D = zgemm(1.0, L, R)
    ML, MR = lu(D, permute_l=True)
    NL = zgemm(1.0, inv(ML), L)
    NR = zgemm(1.0, R, inv(MR))
    return E, NL, NR

def heigen(F, S):
    E, C = eigh(F, S)
    idx = E.argsort()
    C = C[:,idx]
    return E, C

def dm(NL, NR, mo_occ):
    dm = zgemm(2.0, NR[:,mo_occ>0], NL[mo_occ>0,:])
    return dm

def odm(NL, NR, mo_occ):
    dm = zgemm(1.0, NR[:,mo_occ>0], NL[mo_occ>0,:])
    return dm

# For normal UHF cases
def o_vhf(mol, adm, bdm):
    nao = adm.shape[0]
    g0 = mol.intor('cint2e_sph').reshape(nao,nao,nao,nao)
    aJ0 = numpy.einsum('rs,pqsr->pq', adm, g0)
    bJ0 = numpy.einsum('rs,pqsr->pq', bdm, g0)
    aK0 = numpy.einsum('rs,prsq->pq', adm, g0)
    bK0 = numpy.einsum('rs,prsq->pq', bdm, g0)
    av1 = aJ0 + bJ0 - aK0
    bv1 = bJ0 + aJ0 - bK0 
    return av1, bv1

# For RHF cases
def get_vhf(mol, dm):
    nao = dm.shape[0]
    g0 = mol.intor('cint2e_sph').reshape(nao,nao,nao,nao)
    G0 = numpy.zeros(g0.shape, dtype='complex128')
    G0[:,:,:,:] = g0
    J0 = numpy.einsum('rs,pqsr->pq', dm, G0)
    K0 = numpy.einsum('rs,prsq->pq', dm, G0)
    v1 = J0 - .5*K0
    return v1

# Special for H2+
# o_vhf_1 for alpha spin 
def o_vhf_1(mol, dm):
    nao = dm.shape[0]
    g0 = mol.intor('cint2e_sph').reshape(nao,nao,nao,nao)
    aJ0 = numpy.einsum('rs,pqsr->pq', dm, g0)
    aK0 = numpy.einsum('rs,prsq->pq', dm, g0)
    return aJ0 - aK0

# o_vhf_2 for beta spin
def o_vhf_2(mol, dm):
    nao = dm.shape[0]
    g0 = mol.intor('cint2e_sph').reshape(nao,nao,nao,nao)
    aJ0 = numpy.einsum('rs,pqsr->pq', dm, g0)
    return aJ0 
