from qiskit_optimization.applications import Maxcut
from qiskit_optimization.converters import QuadraticProgramToQubo
import dimod

instance = "./data/MCP/MaxCut_10.mc"

max_cut = Maxcut(Maxcut.parse_gset_format(instance))
qp = max_cut.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
bqm_binary = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)