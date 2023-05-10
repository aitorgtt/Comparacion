import dimod
import utils
from qiskit_optimization.converters import QuadraticProgramToQubo


instance = "./data/TSP/dj10.tsp"

tsp = utils.parse_tsplib_format(instance)
qp = tsp.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
bqm = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)


