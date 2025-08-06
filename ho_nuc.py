#!/usr/bin/env python

"""
Module calculates the Harmonic Oscillator nuclear WFs and their derivatives for diatomic molecules.
Also, the sample_displacements function is included to replace the old nuclear.geom.geom_gen function.
"""

import numpy
import math
from NECSCF.Intgr import integral
from NECSCF_DFT.Nuclear import geom
from pyscf import gto
import matplotlib.pyplot as plt


def get_HermP(x, v):
    """
    Returns the Hermite polynomial of order v evaluated at x.
    """
    if v == 0:
        return 1.0
    elif v == 1:
        return 2.0 * x
    elif v == 2:
        return 4.0 * x**2 - 2.0
    elif v == 3:    
        return 8.0 * x**3 - 12.0 * x
    elif v == 4:
        return 16.0 * x**4 - 48.0 * x**2 + 12.0
    elif v == 5:
        return 32.0 * x**5 - 160.0 * x**3 + 120.0 * x
    else:
        raise NotImplementedError("Hermite polynomials are only included up to order 5.")
    
def get_HermP_ddx(x, v):
    """
    Returns the derivative of the Hermite polynomial of order v wrt x, evaluated at x.
    """
    if v == 0:
        return 0.
    elif v == 1:
        return 2.0
    elif v == 2:
        return 8.0 * x
    elif v == 3:    
        return 24.0 * x**2 - 12.0
    elif v == 4:
        return 64 * x**3 - 96.0 * x
    elif v == 5:
        return 160.0 * x**4 - 480.0 * x**2 + 120.0
    else:
        raise NotImplementedError("Hermite polynomials are only included up to order 5.")


def get_nuc_wfn(x, v, alpha):
    '''
    Returns the Harmonic oscillator nuclear wavefunction for a given molecule with vibrational level v.
    Wavefunction is evaluated at x and normalised to 1 over all space.
    alpha: Conversion factor from a.u. to natural units of Harmonic Oscillator
    '''
    norm_constant = numpy.sqrt(alpha) / numpy.sqrt(((2**v)*math.factorial(v))*numpy.sqrt(numpy.pi))
    nuc_wfn = norm_constant * numpy.exp(-0.5 * (alpha * x)**2) * get_HermP(x * alpha, v)  # Calculate the nuclear wavefunction
    return nuc_wfn


def get_nuc_wfn_derivative(x, v, alpha):
    """
    Returns the derivative of the Harmonic oscillator nuclear WF for a given molecule with vibrational QN v wrt
    the internal coordinate (mass-weighted displacement).
    """
    norm_constant = numpy.sqrt(alpha) / numpy.sqrt(((2**v)*math.factorial(v))*numpy.sqrt(numpy.pi))
    nuc_wfn_derivative = alpha * norm_constant * numpy.exp(-0.5 * (alpha * x)**2) * \
    (get_HermP_ddx((alpha * x), v) - alpha * x * get_HermP((alpha * x), v))
    return nuc_wfn_derivative

'''
def get_nuc_term(nuc_wfn, nuc_wfn_derivative, epsilon = 0.12):
    if abs(nuc_wfn) < 0.33 and abs(nuc_wfn_derivative) >= 0.5:
        nuc_wfn =   (nuc_wfn) / ((nuc_wfn)**2 + (epsilon)**2) # replace discontinuity with smoothed function
        return nuc_wfn * nuc_wfn_derivative
    elif abs(nuc_wfn) < 0.33 and abs(nuc_wfn_derivative) < 0.5:
        return 0
    else:
        return nuc_wfn_derivative / nuc_wfn
'''


def get_nuc_term(nuc_wfn, nuc_wfn_derivative, v):
    if v == 0: 
        return nuc_wfn_derivative / nuc_wfn
    elif v == 1:
        epsilon = 0.01
        nuc_wfn =   (nuc_wfn) / ((nuc_wfn)**2 + (epsilon)**2) # replace discontinuity with smoothed function
        return nuc_wfn * nuc_wfn_derivative



def get_nuc_data(mol, N, NQ, step, DisVector, angular_freq, v):
    # angular_freq should be given in cm^-1
    angular_freq = angular_freq * 4.55634e-6  # Convert to a.u.
    mass = geom.reduced_mass(mol)  # Reduced mass in a.u.
    X_k = list((numpy.array(range(N)) - int(NQ - 1))*step/1.88972612)  
    alpha = numpy.sqrt(mass * angular_freq)  
    nuc_wfn = numpy.zeros(N)
    nuc_wfn_derivative = numpy.zeros(N)
    nuc_term = numpy.zeros(N)
    nuc_density = numpy.zeros(N)
    for i in range(N):
        nuc_wfn[i] = get_nuc_wfn(X_k[i], v, alpha)
        nuc_wfn_derivative[i] = get_nuc_wfn_derivative(X_k[i], v, alpha)
        nuc_term[i] = get_nuc_term(nuc_wfn[i], nuc_wfn_derivative[i], v)
        nuc_density[i] = nuc_wfn[i]**2
    nuc_energy = angular_freq * (v + 0.5)
    return nuc_energy, nuc_wfn, nuc_wfn_derivative, nuc_term, nuc_density


def show_nuc_data(mol, N, step, DisVector, k, filename, v, offset = None):
    """
    Returns the nuclear density for a given molecule at the displaced coordinates.
    displaced_coords: 3D array of atomic coordinates for each displacement, preserving COM.
    k: Force constant of the diatomic molecule in a.u. (10^-5 * Newton/meter)
    v: Vibrational quantum number
    """
    Q = geom.switch_geom(mol)
    mass = geom.reduced_mass(mol)
    angular_freq = numpy.sqrt(k/mass)
    if offset is None:
        offset = N/2 - 1
    max_disp = (Q + DisVector * (N - int(offset)) * step)
    Full_Grid = numpy.arange(numpy.min(max_disp), numpy.max(max_disp), step)
    atm_wfn = numpy.zeros(shape=(mol.natm, len(Full_Grid)))
    atm_density = numpy.zeros(shape=(mol.natm, len(Full_Grid)))
    for i in range(mol.natm):
        atom_mass = integral.nuc_mass(mol.atom_symbol(i)) * integral.u_to_au()  # Mass of atom in a.u.
        Q_z = geom.switch_geom(mol)[i][2]  # Equilibrium coordinates
        for j in range(len(Full_Grid)):
            x = (Full_Grid[j] - Q_z) * numpy.sqrt(atom_mass * angular_freq)
            atm_wfn[i][j] = get_nuc_wfn(x, v)
            atm_density[i][j] = (atm_wfn[i][j])**2 
    nuc_wfn = atm_wfn[0] + atm_wfn[1]
    nuc_density = atm_density[0] + atm_density[1]  # Sum over all atoms to get total nuclear density

    plt.figure()
    plt.plot(Full_Grid, nuc_wfn, label='Nuclear Wavefunction')
    plt.plot(Full_Grid, nuc_density, label='Nuclear Density')
    xlabel = "Atomic Coordinates / a.u."
    plt.axvline(x=geom.switch_geom(mol)[0][2], linestyle='--', color='r', label='Equilibrium Position')
    plt.axvline(x=geom.switch_geom(mol)[1][2], linestyle='--', color='r', label='Equilibrium Position')
    plt.savefig(filename)
    plt.close
    



def get_CoM(mol):
    """
    Returns the center of mass of the molecule in atomic units.
    """
    CoM = numpy.zeros(3)
    geom.switch_mol(mol, geom.switch_geom(mol))  # Changes atomic coords from tuple to numpy array
    for i in range(mol.natm):
        CoM += mol.atom[i][1] * integral.nuc_mass(mol.atom_symbol(i))
    CoM /= sum([integral.nuc_mass(mol.atom_symbol(i)) for i in range(mol.natm)])
    return CoM






if __name__ == '__main__':
    mol=gto.Mole()
    mol.atom = [['H',(0, 0,  -0.323965)],
            ['H1',(0, 0,  0.323965)]]
    mol.basis = 'dz'
    mol.build()

    N = 30
    NQ = 15
    Q = geom.switch_geom(mol)
    step = 0.01*1.88972612
    angular_freq = 4751
    DisVector = numpy.array([[0.00000, 0.00000,   -0.89429],
                         [0.00000, 0.00000,    0.44749]])

    X_k = list((numpy.array(range(N)) - int(NQ - 1))*step/1.88972612)



    Grid = geom.sample_displacements(mol, N, step, DisVector)[0]
    Disps = geom.sample_displacements(mol, N, step, DisVector)[1]
    for i in range(2):
        nuc_energy, nuc_wfn, nuc_wfn_derivative, nuc_term, nuc_density = get_nuc_data(mol, N, NQ, step, DisVector, angular_freq, v=i)
        filename = f"nuc_test_data{i}.png"
        plt.figure()
        plt.plot(X_k, nuc_wfn, label='Nuclear Wavefunction', color='blue')
        plt.plot(X_k, nuc_wfn_derivative, label='Nuclear Derivative', color='red')
        plt.plot(X_k, nuc_term, label='Nuclear Term', color = 'Green')
        xlabel = "Atomic Coordinates / a.u."
        plt.savefig(filename)
        plt.close



    #print(reduced_mass(mol))
    #print(geom.sample_displacements(mol, 4, 0.01)[0])
    #print(geom.sample_displacements(mol, 4, 0.01)[1])
    #print(geom.sample_displacements(mol, 4, 0.01)[2])
    #print(nuc_wfn)
    #print(nuc_wfn_derivative)
    #print(Grid)
    #print(Disps)
    #print(nuc_density)
