import pypuf 
import pypuf.simulation, pypuf.io
import pypuf.attack
import pypuf.metrics
import numpy as np  # Import numpy
from pypuf.simulation import ArbiterPUF
from pypuf.io import random_inputs

# Create the PUF instance
puf_1 = ArbiterPUF(n=64, seed=1)

# Generate random inputs and evaluate the PUF
response = puf_1.eval(random_inputs(n=64, N=3, seed=2))

# Correct the use of array
response_array = np.array([1, 1, -1], dtype=np.int8)  # Use numpy.array()

print(response)
print(response_array)
############################### ATTACKS ######################

puf = pypuf.simulation.XORArbiterPUF(n=64, k=4, seed=1)
crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=50000, seed=2)

attack = pypuf.attack.LRAttack2021(crps, seed=3, k=4, bs=1000, lr=.001, epochs=100)
attack.fit()  
model = attack.model
print("____________")
print(pypuf.metrics.similarity(puf, model, seed=4))
