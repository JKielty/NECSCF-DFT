#!/usr/bin/env python

import numpy
from numpy import exp, pi, tan, cos, sin, sqrt, einsum, ones, eye
from scipy.linalg import eigh
from pyscf import gto, scf, lib

def x_i(L, N):
    delta_x = L/(N-1)
    x_i = numpy.arange(N) - int((N-1)/2)
    x_i = delta_x * x_i
    return x_i

def firstdiv(L, N):
    n = int((N - 1)/2)
    T = numpy.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if (i == j):
                T[i][j] = 0
            else:
                T[i][j] = -(2*pi/L/N)*((sin((n+1)*(i-j)*2*pi/N)/2/sin((i-j)*pi/N)**2) + \
                             ((-1)**(i-j+1)*(n+1)/(sin((i-j)*pi/N))))
    return T

def seconddiv(L,N):
    n = int((N - 1)/2)
    T = numpy.zeros((N,N))
    for i in range(N):
        for j in range(N):            
            if (i == j):
                T[i][j] = -(4.*pi*pi/L/L)*n*(n+1)/3.
            else:
                T[i][j] = -(2.*pi*pi/L/L/N/sin((j-i)*pi/N)**2.)*((-1)**(j-i)*(n+1)*cos((j-i)*pi/N)+\
                        (n+1)*cos((n+1)*(j-i)*2*pi/N) - sin((n+1)*(j-i)*2*pi/N)/tan((j-i)*pi/N))
    return T 

def T(L, N, mass): #D is a 2*N*N dimensional array, G is a 2*2 dimensional array
    #G = numpy.array([[1.05263158, -1.],[-1.,1.05263158]])/(1.82289E3)
    #T1 = einsum('rs,rik,sjl,rs->ijkl',(ones((2,2))-eye(2)),D,D,G)*-0.5
    #T2 = einsum('im,mk,jn,nl->ijkl',D[0],D[0],D[1],D[1])*-0.5*G[1][1]
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    X1 = seconddiv(L1,N1)
    X2 = seconddiv(L2,N2)
    X3 = seconddiv(L3,N3)
    #T = einsum('im,mk,')
    #T = numpy.zeros((N*N*N,N*N*N))
    #T = numpy.zeros((N,N,N,N,N,N))
    T1 = -0.5*einsum('ij,kl,mn->ikmjln',X1,eye(N2),eye(N3))/mass
    T2 = -0.5*einsum('ij,kl,mn->ikmjln',eye(N1),X2,eye(N3))/mass
    T3 = -0.5*einsum('ij,kl,mn->ikmjln',eye(N1),eye(N2),X3)/mass
    #T1 = -0.5*einsum('ij,kl,mn->ikmjln',X1,X2,X3)/mass
    return T1 + T2 + T3

def V(PES):
    V = numpy.diag(PES)
    return V

def PES(mol, L, N):
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    x1 = x_i(L1, N1)/1.88972612
    x2 = x_i(L2, N2)/1.88972612
    x3 = x_i(L3, N3)/1.88972612
    PES = numpy.zeros((N1,N2,N3))
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                mol.atom[1][1] = (x1[i], x2[j], x3[k])
                print (mol.atom)
                mol.build()
                mf = scf.hf.RHF(mol)
                conv, etot, mo_energy, mo_coeff, mo_occ = scf.hf.kernel(mf)
                PES[i][j][k] = etot
    return PES.reshape(N1*N2*N3)

def eigen(H, v):
    #aop = lambda x: numpy.dot(H,x)
    #precond = lambda dx, e, x0: dx/(H.diagonal()-e)
    #x0 = [eye(H.shape[0])[0],eye(H.shape[0])[1],eye(H.shape[0])[2],eye(H.shape[0])[3],eye(H.shape[0])[4]]
    #energy, vector = lib.davidson(aop,x0,precond,nroots=v+2)
    energy, vector = eigh(H)
    return energy, vector

def normalization(C, L, N):
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    X = C.copy().reshape(N1,N2,N3)
    delta_x1,delta_x2,delta_x3 = L1/(N1-1),L2/(N2-1),L3/(N3-1)
    c=0.
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                c = c + X[i][j][k]**2.*delta_x1*delta_x2*delta_x3
    c = 1./sqrt(c)
    return c*C

def grad1D(f, L, N):
    delta_x = L/(N-1)
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

def grad(C, L, N):
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    X = C.reshape(N1,N2,N3)
    DX = numpy.zeros((N1,N2,N3))
    DY = numpy.zeros((N1,N2,N3))
    DZ = numpy.zeros((N1,N2,N3))
    for j in range(N2):
        for k in range(N3):
            DX[:,j,k] = grad1D(X[:,j,k], L1, N1)
    for i in range(N1):
        for k in range(N3):
            DY[i,:,k] = grad1D(X[i,:,k], L2, N2)
    for i in range(N1):
        for j in range(N2):
            DZ[i,j,:] = grad1D(X[i,j,:], L3, N3)
    DX, DY, DZ = DX/X, DY/X, DZ/X
    DD = [None]*N1*N2*N3
    for i in range(N1):
        for j in range(N2):
            for k in range(N3):
                DD[i*N2*N3+j*N3+k] = (DX[i][j][k],DY[i][j][k],DZ[i][j][k])
    return DD

def Nuc_iter(mol, L, N, v):
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    pes = PES(mol,L, N).reshape(N1*N2*N3)
#    #print pes
    mass = 1.82289E3
    H = T(L,N, mass).reshape((N1*N2*N3,N1*N2*N3)) + V(pes)
    energy, vector = eigen(H, v)
    vector_n = normalization(vector[:,v], L, N)
    return energy[v], vector_n

def NUC_iter(L, N, v, pes, mass):
    (L1, L2, L3) = L
    (N1, N2, N3) = N
    H = T(L,N, mass).reshape((N1*N2*N3,N1*N2*N3)) + V(pes)
    energy, vector = eigen(H.real, v)
    vector_n = normalization(vector[:,v], L, N)
    return energy[v], vector_n

#mol = gto.Mole()
#mol.atom = [['F',(0, 0, -1.1)], ['H',(0, 0, 0.)], ['F',(0, 0, 1.1)]]
#mol.charge=-1
#mol.basis = 'sto-3g'
#mol.build()
#mf=scf.hf.RHF(mol)
#mf.kernel()

#N = 11
#v=0
#e,c = Nuc_iter(mol, (1.5*1.88972612,1.5*1.88972612,1.0*1.88972612),(N,N,N),v)

#(DX,DY,DZ) = grad(c, (1.5*1.88972612,1.5*1.88972612,1.0*1.88972612),(N,N,N))

#print (e)
#print c[:,0]

#X = numpy.zeros(N)
#Z = c.reshape(N,N,N)
#print (Z[5,5,:])
#print (DZ[5,5,:])
#print (Z[:,5,5])
#print (Z[5,:,5])
