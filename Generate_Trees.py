from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def bootstrap(X, Y, prop):
    n = len(Y)
    nb = int(n*prop)
    b_idx = np.random.choice(range(n), size=nb, replace=True)
    Xb = X[b_idx, :]
    Yb = Y[b_idx]
    oob_idx = list(set(range(n)) - set(b_idx))
    Xoob = X[oob_idx, :]
    Yoob = Y[oob_idx]
    return Xb, Yb, Xoob, Yoob


def generate_trees(X, Y, max_depths, prop=None, n_estimators=None):
    if prop is None:
        prop = 0.5
    if n_estimators is None:
        n_estimators = 1000
    List_trees = []
    List_scores = []
    for i in range(n_estimators):
        Xb, Yb, Xoob, Yoob = bootstrap(X, Y, prop)
        clf = DecisionTreeClassifier(random_state=0, max_depth=max_depths)
        tree = clf.fit(Xb, Yb)
        List_scores.append(tree.score(Xoob, Yoob))
        List_trees.append(tree)
    return np.array(List_trees), np.array(List_scores)

def labeling_trees(List_trees, List_scores, max_depths, rho=None):
    if rho is None:
        rho = 0.1

    List_tree_structures = []
    for tree in List_trees:
        nodes = tree.tree_.__getstate__()['nodes']
        tree_structure = -np.ones([2**(max_depths+1)-1, 3])

        tree_structure[:len(nodes), :] = np.array([list(nodes[k])[:3] for k in range(len(nodes))])
        List_tree_structures.append(tree_structure)
    List_labels = np.zeros(len(List_scores))
    List_labels[np.argsort(List_scores)[int((1-rho)*len(List_scores)):]] = 1
    return List_tree_structures, List_labels




if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    max_depths = 3
    List_trees, List_scores = generate_trees(X, Y, max_depths)
    tree_structures, labels = labeling_trees(List_trees, List_scores, max_depths)


