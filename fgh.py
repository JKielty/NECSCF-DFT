#!/usr/bin/env python

import numpy
from numpy import exp, pi, cos, sin, sqrt, log, tan
from scipy.linalg import inv, eigh
from scipy.linalg.blas import zgemm
from pyscf import scf, gto, lib
from NECSCF.Nuclear import geom

def x_i(L, N):
    delta_x = L/(N)
    x_i = numpy.arange(N,  dtype='float') + 1.  # Change float128 to float
    x_i = delta_x * x_i
    return x_i

def T(L, N, mass):
    if N % 2 == 0:
        n = N / 2
    else:
        n = (N - 1)/2
    n = int(n)
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
    energy, vector = eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    vector = vector[:,idx]
    return energy, vector

def normalization(vector, L, N):
    delta_x = L/N
    c = 1./(delta_x * sum(vector.conj()*vector))
    return sqrt(c) * vector

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
                T[i][j] = -(2.*pi/(L*N))*((sin((n+1)*(i-j)*2.*pi/N)/2./sin((i-j)*pi/N)**2) + \
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


def grad(f, L, N):
    delta_x = L/N
    grad = numpy.zeros(len(f))
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


def NUC_iter(L, N, v, PES, vec_pon, mass, freq_cal):
    H = T(L, N, mass) + numpy.diag(PES) - 0.5*(zgemm(1.0, numpy.diag(vec_pon), firstgrad(L, N)) + zgemm(1.0, firstgrad(L, N), numpy.diag(vec_pon)))
    energy, vector = eigen(H)
    print('Nuclear eigenvalues:', energy)
    vector_n = normalization(vector[:,int(round(v))], L, N)
    print('Normalised Nuclear eigenvector:', vector_n)
    gradient = grad(vector_n, L, N) / vector_n
    #gradient = (vector_n.conj() * anagrad(vector_n, firstgrad(L, N)) - \
    #            vector_n * anagrad(vector_n.conj(), firstgrad(L, N)))/(2*vector_n * vector_n.conj())
    if freq_cal == True:
        w1, w2 = freq(energy)
        return energy[v], vector_n, gradient, w1, w2
    else:
        return energy[v], vector_n, gradient

def freq(energy): # v1 initial state; v2 final state
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
    anharm = (energy[1] - energy[0])/4.55633E-6
    return harm, anharm
