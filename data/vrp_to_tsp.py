import dimod
import utils
from qiskit_optimization.converters import QuadraticProgramToQubo


instance = "./data/VRP/P-n4_1.vrp"

vrp = utils.parse_vrplib_format(instance)
qp = vrp.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
bqm = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)