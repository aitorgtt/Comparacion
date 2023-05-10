'''
* Copyright (c) 2023 TECNALIA <esther.villar@tecnalia.com;eneko.osaba@tecnalia.com>
*
* This file is free software: you may copy, redistribute and/or modify it
* under the terms of the GNU General Public License as published by the
* Free Software Foundation, either version 3.
*
* This file is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
from pathlib import Path
import networkx as nx
import random
import matplotlib.pyplot as plt

instance = 'MaxCut_20'
n_reduced_variables = 50
edges = 0

path = Path('generated_data', f'{instance}.mc')

output = ""

for i in range(n_reduced_variables):
    for j in range(n_reduced_variables):
        int = random.randint(0,9)
        if i != j and int<4:
            output = output + str(i+1) + " " + str(j+1) + " " + str(1)
            output = output + "\n"
            edges = edges+1

f = open(path, "x")

f.write(str(n_reduced_variables) + " " + str(edges))
f.write("\n")
f.write(output)
f.close()