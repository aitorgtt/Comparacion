from crear_instancia_linea import bqm_linea
from dwave.system import DWaveSampler
import time



t0 = time.time()
state_updated = DWaveSampler(num_reads=10000).sample(bqm_linea)
t1 = time.time()

print(state_updated.first.energy)
print(t1-t0)