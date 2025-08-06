#!/usr/bin/env python

'''
Integration and intermediate generator
'''

from pyscf import scf, gto, dft
import numpy
from scipy.linalg import norm

#***********************************************************************
### Nuclear mass
def u_to_au():
    '''
    From wikipedia, https://en.wikipedia.org/wiki/Unified_atomic_mass_unit
    '''
    return 1822.888486192

def nuc_mass(x):
    '''
    From Gaussian 16
    '''
    dict = {'H': 1.00783, 'H1': 2.01410, 'He': 4.00260, 'Be': 9.01218, 'B': 11.00931, 'Li': 7.01600, 'C': 12.00000, 'N': 14.00307, 'O': 15.99491, 'F': 18.99840, 'S': 31.97207}
    return dict[x]

#***********************************************************************
### Integral and derivative integrals
def make_1pdm(mo_coeff, mo_occ):
    mocc = mo_coeff[:,mo_occ>0]
    return numpy.dot(mocc*mo_occ[mo_occ>0], mocc.T.conj())

def make_h1e(mol):
    h = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
    return h

def make_s1e(mol):
    return mol.intor('cint1e_ovlp_sph')

def make_vhf(ks, mol, adm, bdm):
    return ks.get_veff(mol, dm = (adm, bdm))

def S_a0(mol):
    return mol.intor('cint1e_ipovlp_sph', comp=3)

def S_aa(mol):
    return mol.intor('cint1e_ipovlpip_sph', comp=9)

def S_a20(mol):
    return mol.intor('cint1e_ipipovlp_sph', comp=9)

#***********************************************************************
### Derivative integrals and molecular orbital coefficients in a specific
### degree of freedom.
def intrinv3_dof(mol, inte, dofdes):
    dict = {'x':0, 'y':1, 'z':2}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    return mixed_inte

def int1e3_dof(mol, inte, atm_no, dofdes):
    dict = {'x':0, 'y':1, 'z':2}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    integral = numpy.zeros((mixed_inte.shape), dtype='complex256')
    shl0, shl1, p0, p1 = mol.offset_nr_by_atom()[atm_no]
    integral[p0:p1,:] = mixed_inte[p0:p1,:]
    return integral

def intdd1e9_dof(mol, inte, atm_no, dofdes):
    dict = {'x':0,# 'xy':1, 'xz':2,\
            'y':4,# 'yy':4, 'yz':5,\
            'z':8}# 'zy':7, 'zz':8}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    integral = numpy.zeros((mixed_inte.shape), dtype='complex256')
    shl0, shl1, p0, p1 = mol.offset_nr_by_atom()[atm_no]
    integral[p0:p1,:] = mixed_inte[p0:p1,:]
    return integral

def intd1ed9_dof(mol, inte, atm_no, dofdes):
    dict = {'x':0,# 'xy':1, 'xz':2,\
            'y':4,# 'yy':4, 'yz':5,\
            'z':8}# 'zy':7, 'zz':8}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    integral = numpy.zeros((mixed_inte.shape), dtype='complex256')
    shl0, shl1, p0, p1 = mol.offset_nr_by_atom()[atm_no]
    integral[p0:p1,p0:p1] = mixed_inte[p0:p1,p0:p1]
    return integral

'''
Function for second-order kinetic term <a|del2|a>
'''

def c11_dof(C11, dofdes):
    dict = {'x':0,# 'xy':1, 'xz':2,\
            'y':4,# 'yy':4, 'yz':5,\
            'z':8}# 'zy':7, 'zz':8}
    dof = dict[dofdes]
    c11 = C11[dof]
    return c11

def intdd1e_dof(mol, inte, atm_no_bra, dofdes):
    dict = {'x':0,# 'xy':1, 'xz':2,\
            'y':1,# 'yy':4, 'yz':5,\
            'z':2}# 'zy':7, 'zz':8}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    integral = numpy.zeros((mixed_inte.shape), dtype='complex256')
    shl0_bra, shl1_bra, p0_bra, p1_bra = mol.offset_nr_by_atom()[atm_no_bra]
    integral[p0_bra:p1_bra,:] = mixed_inte[p0_bra:p1_bra,:]
    return integral

def intd1ed_dof(mol, inte, atm_no_bra, atm_no_ket, dofdes):
    dict = {'x':0,# 'xy':1, 'xz':2,\
            'y':1,# 'yy':4, 'yz':5,\
            'z':2}# 'zy':7, 'zz':8}
    dof = dict[dofdes]
    mixed_inte = inte[dof]
    integral = numpy.zeros((mixed_inte.shape), dtype='complex256')
    shl0_bra, shl1_bra, p0_bra, p1_bra = mol.offset_nr_by_atom()[atm_no_bra]
    shl0_ket, shl1_ket, p0_ket, p1_ket = mol.offset_nr_by_atom()[atm_no_ket]
    integral[p0_bra:p1_bra,p0_ket:p1_ket] = mixed_inte[p0_bra:p1_bra,p0_ket:p1_ket]
    return integral

def c1_dof(C1, atm_no, dofdes):
    '''Returns C1 in a specific degree of freedom'''
    dict = {'x':0, 'y':1, 'z':2}
    dof = dict[dofdes]
    c1 = C1[dof][atm_no]
    return c1


#***********************************************************************
if __name__ == '__main__':
    mol = gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 1'
    mol.basis = '6-31g'
    mol.build()
