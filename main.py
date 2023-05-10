import hybrid
import dwave.system
from dwave.system import DWaveSampler
import time

from data.tsp_to_bqm import bqm

s = open('salida.txt', 'w')
s.write("Instancia: TSP 10\n")
s.write("Numero de variables: {}\n\n".format(len(bqm)))

initial_state = hybrid.State.from_problem(bqm)


s.write("_____________HIBRIDOS____________\n")
s.write("LeapHybridBQMSampler \n")
t0 = time.time()
sampleset = dwave.system.LeapHybridBQMSampler().sample(bqm, label='LeapHybridBQMSampler')
t1 = time.time()
s.write("energy: {} \n".format(sampleset.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("Kerberos: \n")
t0 = time.time()
sampleset = hybrid.KerberosSampler().sample(bqm, qpu_params={'label': 'Kerberos'})
t1 = time.time()
s.write("energy: {} \n".format(sampleset.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("LOOP: EnergyImpact + QPUSubproblemAutoEmbeddingSampler + Splat: \n")
sub_size = 40
workflow=hybrid.Loop((hybrid.EnergyImpactDecomposer(sub_size, traversal = "pfs") | hybrid.QPUSubproblemAutoEmbeddingSampler() | hybrid.SplatComposer()), max_iter=50)
t0=time.time()
state_updated=workflow.run(initial_state).result()
t1=time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("EnergyImpact + QPUSubproblemAutoEmbeddingSampler + Splat: \n")
sub_size = 200
workflow = (hybrid.EnergyImpactDecomposer(size=sub_size) |
            hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=2000) |
            hybrid.SplatComposer())
t0 = time.time()
state_updated = workflow.run(initial_state, qpu_params={'label': 'Energy-QPU-Splat'}).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))



s.write("EnergyImpact + QPUSubproblemAutoEmbeddingSampler + Splat: \n")
sub_size = 40
workflow = (hybrid.EnergyImpactDecomposer(size=sub_size) |
            hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=2000) |
            hybrid.SplatComposer())
t0 = time.time()
state_updated = workflow.run(initial_state, qpu_params={'label': 'Energy-QPU-Splat'}).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))



s.write("\n\n\n_____________CLASICOS____________\n")
s.write("Full clasical Kerberos: \n")
iteration = hybrid.Race(hybrid.BlockingIdentity(),
                        hybrid.InterruptableTabuSampler(timeout=500),
                        hybrid.InterruptableSimulatedAnnealingProblemSampler(num_reads=1, num_sweeps=10000)) \
            | hybrid.ArgMin()
workflow = hybrid.Loop(iteration, max_iter=100, max_time=None, convergence=3)
t0 = time.time()
state_updated = workflow.run(initial_state).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))
s.write("Tabu: \n")


workflow = hybrid.TabuProblemSampler()
t0 = time.time()
state_updated = workflow.run(initial_state).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("Simulated Annealing: \n")
workflow = hybrid.SimulatedAnnealingProblemSampler()
t0 = time.time()
state_updated = workflow.run(initial_state).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))

s.write("SteepestDescent: \n")
workflow = hybrid.SteepestDescentProblemSampler()
t0 = time.time()
state_updated = workflow.run(initial_state).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


"""sub_size = 40 REVERSE ANNEALING
workflow = (hybrid.EnergyImpactDecomposer(size=sub_size) |
            hybrid.ReverseAnnealingAutoEmbeddingSampler(num_reads=100) |
            hybrid.SplatComposer())
initial_state = hybrid.State.from_problem(bqm)
t0 = time.time()
state_updated = workflow.run(initial_state, qpu_params={'label': 'Energy-ReverseAnnealing-Splat'}).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


"""

s.write("\n\n\n_____________CUANTICOS____________\n")
s.write("QPUSubproblemAutoEmbeddingSampler: \n")
workflow = (hybrid.IdentityDecomposer() |
            hybrid.QPUSubproblemAutoEmbeddingSampler(num_reads=2000) |
            hybrid.SplatComposer())
t0 = time.time()
state_updated = workflow.run(initial_state, qpu_params={"label" : 'Identity-QPU-Splat'}).result()
t1 = time.time()
s.write("energy: {} \n".format(state_updated.samples.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("AutoEmbeddingComposite: \n")
t0 = time.time()
sampleset = dwave.system.AutoEmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=2000, label='AutoEmbeddingComposite')
t1 = time.time()
s.write("energy: {} \n".format(sampleset.first.energy))
s.write("total time: {} \n \n".format(t1-t0))

s.write("EmbeddingDWaveSampler: \n")
t0 = time.time()
sampleset = dwave.system.EmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=2000, label='EmbeddingDWaveSampler')
t1 = time.time()
s.write("energy: {} \n".format(sampleset.first.energy))
s.write("total time: {} \n \n".format(t1-t0))


s.write("LazyFixedEmbeddingDWaveSampler: \n")
t0 = time.time()
sampleset = dwave.system.LazyFixedEmbeddingComposite(DWaveSampler()).sample(bqm, num_reads=2000, label='LazyFixedEmbeddingDWaveSampler') #MODIFICAR PARAMETROS
t1 = time.time()
s.write("energy: {} \n".format(sampleset.first.energy))
s.write("total time: {} \n \n".format(t1-t0))



s.close()
