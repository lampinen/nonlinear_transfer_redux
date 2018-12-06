import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

rseeds = [66, 80, 104, 107]
relationship_types = ["between_analog", "between_non_analog", "within"]
rep_dists = {analogous: {rtype: {rseed: [] for rseed in rseeds} for rtype in relationship_types} for analogous in [0, 1]}

nlayer = 3
nonlinear = 1
for rseed in rseeds: 
    for analogous in [0, 1]:
        filename_prefix = "results/nlayer_%i_nonlinear_%i_analogous_%i_rseed_%i_" %(nlayer,nonlinear,analogous,rseed)
        for epoch in range(0, 20000, 100): 
            filename = filename_prefix + "penultimate_hidden_reps_epoch_%i.csv" % epoch
            if not os.path.exists(filename): # early stopped
               break
            reps = np.loadtxt(filename, delimiter=',')
            dists = squareform(pdist(reps, metric='euclidean'))
            if analogous: 
               rep_dists[analogous]["between_analog"][rseed].append(dists[[0, 1, 2], [3, 4, 5]])
               rep_dists[analogous]["between_non_analog"][rseed].append(dists[[0, 0, 1, 1, 2, 2], [4, 5, 3, 5, 3, 4]])
            else:
               rep_dists[analogous]["between_non_analog"][rseed].append(dists[:2, 3:])

            rep_dists[analogous]["within"][rseed].append(dists[[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])

for rtype in relationship_types:
    print(rtype)
    for analogous in [0, 1]:
        if rtype == "between_analog" and not analogous:
            continue

        print("\tAnalogous %i:" % analogous)
        final_values = [x for x in rep_dists[analogous][rtype][rseed][-1] for rseed in rseeds] # oh god why
        print("\t\tMean: %.2f, SD: %.2f" % (np.mean(final_values), np.std(final_values)))
