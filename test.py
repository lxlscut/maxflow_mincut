import maxflow
import numpy as np

if __name__ == '__main__':
    g = maxflow.GraphFloat()
    node_ids = g.add_grid_nodes((3, 3))
    structure = np.array([[0, 0, 0],
                          [0, 0, 1],
                          [0, 0, 0]])
    weights = np.array([[1, 2,1],
                        [4, 5,1],
                        [7, 8,1]])
    g.add_grid_edges(node_ids, weights=weights, structure=structure,
                     symmetric=False)
    g.maxflow()
    a = g.get_segment(8)
    print(a)