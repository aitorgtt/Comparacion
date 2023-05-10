from dwave.system import DWaveSampler
import dimod

sampler=DWaveSampler()
qubits = sampler.nodelist
couplers = sampler.edgelist


edges_linea = {}
i = qubits.pop(0)
nodes_linea={i:-1}
keep = True
while keep:
    keep = False
    for j in qubits:
        if (i,j) in couplers:
            nodes_linea[j]=-1
            edges_linea[(i,j)]=2
            i=j
            qubits.remove(i)
            keep = True
            break

bqm_linea = dimod.BinaryQuadraticModel(nodes_linea, edges_linea, 0.0, 'BINARY')
print(bqm_linea)