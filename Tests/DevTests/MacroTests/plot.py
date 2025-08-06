import matplotlib.pyplot as plt
import numpy

e_final =  numpy.array([-1.13334596, -1.12831124, -1.13084892, -1.13235054, -1.13282894, -1.13228316,
 -1.13042143, -1.12864534, -1.1272876,  -1.12677364])

e_bo =  numpy.array([-1.11790512, -1.12479729, -1.12914581, -1.13143832, -1.13206653, -1.13134717,
 -1.12953773, -1.12684843, -1.12345142, -1.11948812])





N = 10
NQ = 5
step = 0.025*1.88972612

X_k = list((numpy.array(range(N)) - int(NQ - 1))*step/1.88972612)


plt.plot(X_k, e_bo, 'go', X_k, e_bo, 'g--', label='BO', color='Black')
plt.plot(X_k, e_final, 'go', X_k, e_final, 'g--', label='NECSCF-DFT', color='Red')
plt.legend()
plt.ylabel('Energy difference / micro au')
plt.xlabel('Displacement / a.u.')
plt.savefig('PESs_v0.png')

