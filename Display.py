from graphviz import Graph
import numpy as np


def draw_magnitudes(graph):
    dot = Graph("Magnitude binary")

    for point, coefficient in graph.items():
        variables = np.nonzero(point)[0]
        arity = np.count_nonzero(point)
        if  arity == 1:
            v1 = str(variables[0])
            dot.node(v1, label="<<b>" + v1 + "</b>: " + str(round(coefficient, 1)) + ">")
        elif arity == 2 and abs(coefficient) > 0.1:
            dot.edge(str(variables[0]), str(variables[1]), label=str(round(coefficient, 1)))
    dot.render('test-output/round-table.gv', view=True)
