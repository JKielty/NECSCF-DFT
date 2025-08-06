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
    energy, vector = eigh(H)
    idx = energy.argsort()
    energy = energy[idx]
    vector = vector[:,idx]
    return energy, vector

def grad(f, L, N):
    delta_x = L/N
    grad = numpy.zeros(len(f), dtype='float')  # Change float128 to float
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
    aop = lambda x: numpy.dot(H,x)
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

def NUC_iter(L, N, v, PES,A, mass, freq_cal):
    A = numpy.zeros(N)
    H = T(L, N, mass) + numpy.diag(PES)
    energy, vector = eigen(H)
    ENG = numpy.zeros(v)
    VEC = numpy.zeros((v,N))
    for i in range(v):
        vector_n = normalization(vector[:,i], vector[:,i], L, N)[1]
        VEC[i] = vector_n
        ENG[i] = energy[v]
    if freq_cal == True:
        w = freq(energy, v)
        return ENG, VEC, w
    else:
        return ENG, VEC #gradient

def freq(energy, v): # v1 initial state; v2 final state
    anharm = numpy.zeros(v)
    for i in range(v):
        anharm[i] = (energy[i+1] - energy[i])/4.55633E-6
    return anharm
