#!/usr/bin/env python

import numpy
from numpy import exp, pi, cos, sin, sqrt, log, tan
from scipy.linalg import inv, eigh
from scipy.linalg.blas import zgemm
from pyscf import scf, gto, lib
from NECSCF.Nuclear import geom
from NECSCF.Mathtool import mathtool

def x_i(L, N):
    delta_x = L/(N)
    x_i = numpy.arange(N,  dtype='float') + 1.  # Change float128 to float
    x_i = delta_x * x_i
    return x_i

def k_l(L, N):
    delta_k = 2*pi/L
    n = (N - 1)/2
    k_l = numpy.arange(-n, n+1)
    delta_k = 2*pi/L
    k_l = k_l * delta_k
    return k_l

def T(L, N, mass):
    """
    kinetic terms
    T_{ij}.
    i, j: number of grid pts
    """
    n = int((N - 1)/2)
    T = numpy.zeros((N, N))
    delta_x = L/(N)
    for i in range(N):
        for j in range(N):
            for l in range(n):
                T[i][j] = T[i][j] + 2./(N)*cos((l+1)*2.*pi*(i-j)/(N))*2./mass*(pi*(l+1)/L)**2
    return T

def V(PES):
    V = numpy.diag(PES)
    return V

def eigen(H):
    """
    Diagonalization of Hamiltonian
    :param H: Hamiltonian
    :return: Sorted energy state and eigenvectors
    """
    energy, vector = eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    vector = vector[:,idx]  # grid pts : state idx
    return energy, vector

def grad(f, L, N):
    delta_x = L/N
    grad = numpy.zeros(len(f), dtype='complex256')
    for i in range(4, len(f)-5):
        grad[i] = (3*f[i-4]-32*f[i-3]+168*f[i-2]-672*f[i-1]+0*f[i+0]+672*f[i+1]-168*f[i+2]+32*f[i+3]-3*f[i+4])/(840.*delta_x)
    #Left and right formulas for terminal points
    grad[0] = (f[1] - f[0])/delta_x
    grad[len(f)-1] = (f[len(f)-1] - f[len(f)-2])/delta_x
    #Two point formula applies
    grad[1] = (f[2] - f[0])/(2.*delta_x)
    grad[len(f)-2] = (f[len(f)-1] - f[len(f)-3])/(2.*delta_x)
    #Four point formula applies
    grad[2] = (-f[4] + 8.*f[3] - 8.*f[1] + f[0])/(12.*delta_x)
    grad[3] = (-f[5] + 8.*f[4] - 8.*f[2] + f[1])/(12.*delta_x)
    grad[len(f)-3] = (-f[len(f)-1] + 8.*f[len(f)-2] - 8.*f[len(f)-4] + f[len(f)-5])/(12.*delta_x)
    grad[len(f)-4] = (-f[len(f)-2] + 8.*f[len(f)-3] - 8.*f[len(f)-5] + f[len(f)-6])/(12.*delta_x)
    return grad

def eigs(H):
    # davidson diagonalization
    aop = lambda x: numpy.dot(H, x)
    precond = lambda dx, e, x0: dx/(a.diagonal()-e)
    x0 = a[0]
    energy, vector = lib.davidson(aop, x0, precond)
    return energy, vector

#def normalization(vector, L, N):
#    delta_x = L/N
#    print delta_x
#    c = 1./(delta_x * sum(vector*vector))
#    return sqrt(c) * vector

def normalization(CL, CR, L, N):
    delta_x = L/N
    c = 1./(delta_x * sum(CL*CR))
    return sqrt(c) * CL, sqrt(c) * CR

def anagrad(CR, T):
    grad = T.dot(CR)
    return grad

def firstgrad(L, N):
    n = (N - 1)/2
    T = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i==j):
                T[i][j] = 0.
            else:
                T[i][j] = -(2.*pi/L/N)*((sin((n+1)*(i-j)*2.*pi/N)/2./sin((i-j)*pi/N)**2) + \
                             ((-1)**(i-j+1)*(n+1)/(sin((i-j)*pi/N))))
    return T

def secgrad(L, N):
    n = (N - 1)/2
    T = numpy.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if (i==j):
                T[i][j] = -(2.*pi/L)**2*n*(n+1)/3.
            else:
                T[i][j] = -(2.*pi/L)**2/(2*N*sin((j-i)*pi/N)**2)*((-1)**(j-i)*(n+1)*cos((j-i)*pi/N) + \
                             (n+1)*cos((n+1)*(j-i)*2.*pi/N) - \
                             sin((n+1)*(j-i)*2.*pi/N)/tan((j-i)*pi/N))
    return T

def NUC_iter(L, N, v, PES, mass, freq_cal):
    """
    According to nuclear equation, we can only calculate one state of nuclear orbital.
    But we have N geometries and we are solving N nuclear wavefunctions corresponding to each of geometries
    Overall will give a separation of state symmetries and find the excited state
    Module potential replacing E_{el} is not applied
    """
    H = T(L, N, mass) + numpy.diag(PES) #- zgemm(1.0, numpy.diag(A), firstgrad(L, N))
    # H is the Hamiltonian created by FGH
    energy, vector = eigen(H)
    vector_n = normalization(vector[:,int(round(v))], vector[:,int(round(v))], L, N)[1]
    #  vector:   grid pts : state idx
    #print vector_n
    #vector_n = abs(vector_n)
    #gradient = anagrad(vector_n, firstgrad(L, N))
    grad1 = grad(vector_n, L, N)
    #print 'Numerical\n', grad1/CR_n
    #print 'Analytical\n', gradient/CR_n
    if freq_cal == True:
        w1, w2 = freq(energy, v)
        return energy[v], (vector_n, vector_n), grad1, w1, w2
    else:
        return energy[v], (vector_n, vector_n), grad1  #gradient

def freq(energy, v):
    # v1 initial state; v2 final state
    # Finding the vibrational band from the higher state. (more meaningful to compare with exp)
    A = numpy.array([[1, 0.5, -0.5**2, 0.5**3, 0.5**4],
                     [1, 1.5, -1.5**2, 1.5**3, 1.5**4],
                     [1, 2.5, -2.5**2, 2.5**3, 2.5**4],
                     [1, 3.5, -3.5**2, 3.5**3, 3.5**4],
                     [1, 4.5, -4.5**2, 4.5**3, 4.5**4]])
    B = numpy.array([energy[0], energy[1], energy[2], energy[3], energy[4]])
    C = numpy.linalg.solve(A,B)
    #print C
    #print energy
    #print energy[1]-energy[0]
    harm = C[1]/(4.55633E-6) # 1 cm^-1 = 4.55633E-6 u
    # if anharmonic, it is just the energy difference.
    anharm = (energy[v+1] - energy[v])/4.55633E-6
    return harm, anharm

