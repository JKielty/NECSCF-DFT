#!/usr/bin/env python

'''
0: COM gradient
1: internal gradient
'''
from NECSCF.Intgr import integral
import numpy

def trans(mol, C1, DisVector):
    dof= C1.shape[0]
    natom = C1.shape[1]
    transC1 = numpy.zeros(C1[0][0].shape, dtype='complex128')
    for i in range(dof):
        for j in range(natom):
            transC1 += C1[i][j] * DisVector[j][i] #numpy.sign(DisVector[j][i]) * abs(DisVector[j][i])**2
    return transC1

#def backtrans(mol, C1prime):
#    MA = integral.nuc_mass(mol.atom_symbol(0))*integral.u_to_au()
#    MB = integral.nuc_mass(mol.atom_symbol(1))*integral.u_to_au()
#    dof= C1prime.shape[0]
#    C1 = numpy.zeros(C1prime.shape, dtype='complex256')
#    for i in range(dof):
#        C1[i][0] = (MA/(MA + MB))*C1[i][0] + C1[i][1]
#        C1[i][1] = (MB/(MA + MB))*C1[i][0] - C1[i][1]
#    return C1

#def trans2(mol,C11): # transform <del alpha|del beta>-like overlop
#    MA = integral.nuc_mass(mol.atom_symbol(0))*integral.u_to_au()
#    MB = integral.nuc_mass(mol.atom_symbol(1))*integral.u_to_au()
#    dof= C11.shape[0]
#    dofdes = ['x', 'y', 'z']
#    transC11 = numpy.zeros((C11.shape[0],2,C11.shape[2],C11.shape[2]), dtype='complex256')
#    for i in range(dof):
#        transC11[i][0] =   integral.intd1ed_dof(mol, C11, 0, 0, dofdes[i]) \
#                         + integral.intd1ed_dof(mol, C11, 0, 1, dofdes[i]) \
#                         + integral.intd1ed_dof(mol, C11, 1, 0, dofdes[i]) \
#                         + integral.intd1ed_dof(mol, C11, 1, 1, dofdes[i])
#        transC11[i][1] =   integral.intd1ed_dof(mol, C11, 0, 0, dofdes[i])*(MB/(MA + MB))*(MB/(MA + MB)) \
#                         - integral.intd1ed_dof(mol, C11, 0, 1, dofdes[i])*(MB/(MA + MB))*(MA/(MA + MB)) \
#                         - integral.intd1ed_dof(mol, C11, 1, 0, dofdes[i])*(MA/(MA + MB))*(MB/(MA + MB)) \
#                         + integral.intd1ed_dof(mol, C11, 1, 1, dofdes[i])*(MA/(MA + MB))*(MA/(MA + MB))
#    return transC11

#def trans3(mol,C11): # transform <del alpha|del beta>-like overlop
#    MA = integral.nuc_mass(mol.atom_symbol(0))*integral.u_to_au()
#    MB = integral.nuc_mass(mol.atom_symbol(1))*integral.u_to_au()
#    dof= C11.shape[0]
#    dofdes = ['x', 'y', 'z']
#    transC11 = numpy.zeros((C11.shape[0],2,C11.shape[2],C11.shape[2]), dtype='complex256')
#    for i in range(dof):
#        transC11[i][0] =   integral.intdd1e_dof(mol, C11, 0, dofdes[i]) \
#                         + integral.intdd1e_dof(mol, C11, 1, dofdes[i])
#        transC11[i][1] =   integral.intdd1e_dof(mol, C11, 0, dofdes[i])*(MB/(MA + MB))*(MB/(MA + MB)) \
#                         + integral.intdd1e_dof(mol, C11, 1, dofdes[i])*(MA/(MA + MB))*(MA/(MA + MB))
#    return transC11
