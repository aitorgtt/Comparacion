from qiskit_optimization.applications import  BinPacking
from qiskit_optimization.converters import QuadraticProgramToQubo
import dimod
import utils

instance = "./data/BPP/BPP_3.bpp"

weights,max_weight = utils.parse_BPP_format(instance)
binPacking = BinPacking(weights,max_weight)
qp = binPacking.to_quadratic_program()
qp2qubo = QuadraticProgramToQubo()
qubo = qp2qubo.convert(qp)
bqm = dimod.as_bqm(qubo.objective.linear.to_array(), qubo.objective.quadratic.to_array(), dimod.BINARY)