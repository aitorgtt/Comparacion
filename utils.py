#                                               UTILS
########################################################################################################################
# THIS CLASS INCLUDES SOME IMPORTANT METHODS FOR THE CORRECT EXECUTION OF THIS QBrick
########################################################################################################################

import numpy as np
import networkx as nx
from qiskit_optimization.applications import VehicleRouting, Tsp
from qiskit_optimization.exceptions import QiskitOptimizationError
import matplotlib.pyplot as plt

########################################################################################################################
# Method: parse_vrplib_format
# Description: this method reads a CVRPLib compliant instance and generates a Qiskit VehicleRouting Instance
########################################################################################################################

def parse_vrplib_format(filename: str) -> "vrp":
    """Read a graph in VRPLIB format from file and return a VRP instance.

    Args:
        filename: the name of the file.

    Raises:
        QiskitOptimizationError: If the type is not "VRP"
        QiskitOptimizationError: If the edge weight type is not "EUC_2D"

    Returns:
        A Tsp instance data.
    """
    name = ""
    coord = []  # type: ignore
    with open(filename, encoding="utf8") as infile:
        coord_section = False
        for line in infile:
            if line.startswith("NAME"):
                name = line.split(":")[1]
                name.strip()
            elif line.startswith("TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "VRP" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "VRP" type. Actual: {typ}'
                    )
            elif line.startswith("DIMENSION"):
                dim = int(line.split(":")[1])
                coord = np.zeros((dim, 2))  # type: ignore
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "EUC_2D" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "EUC_2D" edge weight. Actual: {typ}'
                    )
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
            elif coord_section:
                v = line.split()
                index = int(v[0]) - 1
                coord[index][0] = float(v[1])
                coord[index][1] = float(v[2])

    x_max = max(coord_[0] for coord_ in coord)
    x_min = min(coord_[0] for coord_ in coord)
    y_max = max(coord_[1] for coord_ in coord)
    y_min = min(coord_[1] for coord_ in coord)

    pos = {i: (coord_[0], coord_[1]) for i, coord_ in enumerate(coord)}

    graph = nx.random_geometric_graph(
        len(coord), np.hypot(x_max - x_min, y_max - y_min) + 1, pos=pos
    )

    for w, v in graph.edges:
        delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
        graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))
    return VehicleRouting(graph, num_vehicles=2, depot=0)

########################################################################################################################
# Method: parse_tsplib_format
# Description: this method reads a TSPLib compliant instance and generates a Qiskit TSP Instance
########################################################################################################################

def parse_tsplib_format(filename: str) -> "Tsp":
    """Read a graph in TSPLIB format from file and return a Tsp instance.

    Args:
        filename: the name of the file.

    Raises:
        QiskitOptimizationError: If the type is not "TSP"
        QiskitOptimizationError: If the edge weight type is not "EUC_2D"

    Returns:
        A Tsp instance data.
    """
    name = ""
    coord = []  # type: ignore
    with open(filename, encoding="utf8") as infile:
        coord_section = False
        for line in infile:
            if line.startswith("NAME"):
                name = line.split(":")[1]
                name.strip()
            elif line.startswith("TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "TSP" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "TSP" type. Actual: {typ}'
                    )
            elif line.startswith("DIMENSION"):
                dim = int(line.split(":")[1])
                coord = np.zeros((dim, 2))  # type: ignore
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                typ = line.split(":")[1]
                typ.strip()
                if "EUC_2D" not in typ:
                    raise QiskitOptimizationError(
                        f'This supports only "EUC_2D" edge weight. Actual: {typ}'
                    )
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
            elif coord_section:
                v = line.split()
                index = int(v[0]) - 1
                coord[index][0] = float(v[1])
                coord[index][1] = float(v[2])

    x_max = max(coord_[0] for coord_ in coord)
    x_min = min(coord_[0] for coord_ in coord)
    y_max = max(coord_[1] for coord_ in coord)
    y_min = min(coord_[1] for coord_ in coord)

    pos = {i: (coord_[0], coord_[1]) for i, coord_ in enumerate(coord)}

    graph = nx.random_geometric_graph(
        len(coord), np.hypot(x_max - x_min, y_max - y_min) + 1, pos=pos
    )

    for w, v in graph.edges:
        delta = [graph.nodes[w]["pos"][i] - graph.nodes[v]["pos"][i] for i in range(2)]
        graph.edges[w, v]["weight"] = np.rint(np.hypot(delta[0], delta[1]))
    return Tsp(graph)

########################################################################################################################
# Method: parse_BPP_format
# Description: this method reads a BPP instance and returns the list of weight and the capacity of the bin
########################################################################################################################

def parse_BPP_format(file):
    data = open(file)
    count = 0
    weights = []
    for line in data:
        if count==1:
            max_weight = int(line)
        if count>1:
            weights.append(int(line))
        count=count+1
    data.close()
    return weights,max_weight

########################################################################################################################
# Method: information
# Description: this method prints information about the ising and the formulation of the problem
########################################################################################################################

def information(qp, qubo):
    print(qp.prettyprint())
    qubitOp, offset = qubo.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))

########################################################################################################################
# Method: draw_graph
# Description: this method draws a graph
########################################################################################################################

def draw_TSP_graph(G):
    colors = ["r" for node in G.nodes]
    pos = [G.nodes[node]["pos"] for node in G.nodes]
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

def draw_tsp_solution(G, order):
    colors = ["r" for node in G.nodes]
    pos = [G.nodes[node]["pos"] for node in G.nodes]
    G2 = nx.DiGraph()
    G2.add_nodes_from(G)
    n = len(order)
    for i in range(n):
        j = (i + 1) % n
        G2.add_edge(order[i], order[j], weight=G[order[i]][order[j]]["weight"])
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(
        G2, node_color=colors, edge_color="b", node_size=600, alpha=0.8, ax=default_axes, pos=pos
    )
    edge_labels = nx.get_edge_attributes(G2, "weight")
    nx.draw_networkx_edge_labels(G2, pos, font_color="b", edge_labels=edge_labels)
    plt.show()

def draw_MCP_graph(G):
    colors = ["r" for node in G.nodes()]
    pos = nx.spring_layout(G)
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
    plt.show()

########################################################################################################################
# Method: make_reverse_anneal_schedule
# Description: Build annealing waveform pattern for reverse anneal feature
########################################################################################################################

def make_reverse_anneal_schedule(s_target=0.0, hold_time=10.0, ramp_back_slope=0.2, ramp_up_time=0.0201,
                                 ramp_up_slope=None):
    """Build annealing waveform pattern for reverse anneal feature.
    Waveform starts and ends at s=1.0, descending to a constant value
    s_target in between, following a linear ramp.
      s_target:   s-parameter to descend to (between 0 and 1)
      hold_time:  amount of time (in us) to spend at s_target (must be >= 2.0us)
      ramp_slope: slope of transition region, in units 1/us
    """
    # validate parameters
    if s_target < 0.0 or s_target > 1.0:
        raise ValueError("s_target must be between 0 and 1")
    if hold_time < 0.0:
        raise ValueError("hold_time must be >= 0")
    if ramp_back_slope > 0.2:
        raise ValueError("ramp_back_slope must be <= 0.2")
    if ramp_back_slope <= 0.0:
        raise ValueError("ramp_back_slope must be > 0")

    ramp_time = (1.0 - s_target) / ramp_back_slope

    initial_s = 1.0
    pattern = [[0.0, initial_s]]

    # don't add new points if s_target == 1.0
    if s_target < 1.0:
        pattern.append([round(ramp_time, 4), round(s_target, 4)])
        if hold_time != 0:
            pattern.append([round(ramp_time+hold_time, 4), round(s_target, 4)])

    # add last point
    if ramp_up_slope is not None:
        ramp_up_time = (1.0-s_target)/ramp_up_slope
        pattern.append([round(ramp_time + hold_time + ramp_up_time, 4), round(1.0, 4)])
    else:
        pattern.append([round(ramp_time + hold_time + ramp_up_time, 4), round(1.0, 4)])

    return pattern