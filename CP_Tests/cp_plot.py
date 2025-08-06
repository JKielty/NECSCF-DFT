import matplotlib.pyplot as plt
import numpy

cphf =  numpy.array([5.35884777e-07, 3.03468744e-07, 1.35172323e-07, 3.37558427e-08,
 0.00000000e+00, 3.34330496e-08, 1.32682383e-07, 2.95598483e-07,
 5.19426711e-07, 7.22378302e-07])

cpks =  numpy.array([5.51870700e-07, 3.03483161e-07, 1.35175005e-07, 3.37559210e-08,
 0.00000000e+00, 3.34337342e-08, 1.32692785e-07, 2.95644230e-07,
 5.19205858e-07, 7.98467572e-07])

cphff = numpy.array([-2.64624114e-05, -1.40326458e-05, -5.98492344e-06, -1.45448899e-06,
  0.00000000e+00, -1.41406168e-06, -5.64099944e-06, -1.27242513e-05,
 -2.29765701e-05, -3.61318968e-05])

e_hf = numpy.array([-0.87723031, -0.9932759,  -1.06128064, -1.10029473, -1.12114976, -1.13033615,
 -1.13191265, -1.12850409, -1.12184754, -1.11310969])

e_hff = numpy.array([-0.87807527, -0.99367972, -1.06143965, -1.10033112, -1.12114976, -1.13036907,
 -1.13204208, -1.12879559, -1.12237301, -1.1135525 ])

e_ks = numpy.array([-0.87722922, -0.99327477, -1.06128008, -1.10029455, -1.12114976, -1.13033601,
 -1.13191228, -1.12850354, -1.1218468,  -1.11310885])

e_bo = numpy.array([-0.87722479, -0.99327304, -1.06127951, -1.10029449, -1.12114976, -1.13033598,
 -1.13191212, -1.12850317, -1.12184632, -1.11310833])

hf_diff = (e_bo - e_hf)*1000
hff_diff = (e_bo - e_hff)*1000
ks_diff = (e_bo - e_ks)*1000

hfks_diff = (e_hf - e_ks)*10000000





N = 10
NQ = 5
step = 0.05*1.88972612

X_k = list((numpy.array(range(N)) - int(NQ - 1))*step/1.88972612)
diff = cphf - cpks

#plt.plot(X_k, cphf, color = 'Black', label = 'CPHF')
#plt.plot(X_k, cpks, color = 'red', label = 'CPKS' )
plt.plot(X_k, hfks_diff, 'go', X_k, hfks_diff, 'g--', label='HF', color='Black')
#plt.plot(X_k, hff_diff, 'go', X_k, hff_diff, 'g--', label='HF-Field', color='Blue')
#plt.plot(X_k, ks_diff, 'go', X_k, ks_diff, 'g--', label='KS', color='Red')
#plt.legend()
plt.ylabel('Energy difference / micro au')
plt.xlabel('Displacement / a.u.')
plt.savefig('hfks_comparison.png')

