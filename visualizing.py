import numpy as np
import matplotlib.pyplot as plot

y_data = np.loadtxt("analogy_data.csv", delimiter=",") 
y_data_no = np.loadtxt("no_analogy_data.csv", delimiter=",") 

plot.figure()
plot.imshow(y_data)
plot.colorbar()
plot.savefig('plots/analogy_data.png')

plot.figure()
plot.imshow(y_data_no)
plot.colorbar()
plot.savefig('plots/no_analogy_data.png')

