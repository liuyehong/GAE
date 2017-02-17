from Discriminate_Network import *
from Generate_Trees import *

class Generate_Adversarial_Ensemble(object):
    def __init__(self, max_depth=None, rho=None, n_iter=None, nb_epoch=None, n_estimators=None,
                 List_trees=None):

        if max_depth is None:
            max_depth = 3
        if rho is None:
            rho = 0.5
        if n_iter is None:
            n_iter = 10
        if n_estimators is None:
            n_estimators = 1000

        self.max_depth = max_depth
        self.rho = rho
        self.n_iter = n_iter
        self.nb_epoch = nb_epoch
        self.n_estimators = n_estimators
        self.List_trees = List_trees



    def fit(self, X, Y):
        # generate trees and train network
        List_trees, List_scores = generate_trees(X, Y, self.max_depth)
        tree_structures, labels = labeling_trees(List_trees, List_scores, self.max_depth)

        for k in range(self.n_iter):
            # fit and refit net work
            discriminate_network = fit_discriminate_network(tree_structures, labels, self.rho, self.nb_epoch)

            # use trained network to select enough good trees.
            List_trees = []
            List_scores = []
            while len(List_trees) < self.n_estimators:
                # regenerate new trees.
                List_new_trees, List_new_scores = generate_trees(X, Y, self.max_depth)
                new_tree_structures, new_labels = labeling_trees(List_new_trees, List_new_scores, self.max_depth)
                # discriminate good trees from new trees.
                X_train = np.array(new_tree_structures)
                X_train = X_train.reshape(X_train.shape[0], 1, 2 ** (self.max_depth + 1) - 1, 3).astype('float32')
                X_train = (X_train - np.mean(X_train)) / np.std(X_train)
                discriminate_probs = discriminate_network.predict(X_train)
                discriminate_idx = np.argsort(discriminate_probs[:, 1])[int(len(Y) * (1 - rho)):]
                # memorize selected good trees
                List_trees.extend(List_new_trees[discriminate_idx])
                List_scores.extend(List_new_scores[discriminate_idx])
            # selected new batch of trees
            List_trees = List_trees[:self.n_estimators + 1]
            List_scores = List_scores[:self.n_estimators + 1]
            tree_structures, labels = labeling_trees(List_trees, List_scores, self.max_depth)

        self.List_trees = List_trees

    def predict_proba(self, X):
        Y_predict = np.zeros([np.size(X, axis=0), self.n_estimators])
        for k in range(self.n_estimators):
            tree = self.List_trees[k]
            Y_predict[:, k] = tree.predict_proba(X)[:, 0]
        return np.average(Y_predict, axis=1)

    def predict(self, X):
        return self.predict_proba(X) < 0.5

    def score(self, X, Y):
        return np.sum(self.predict(X) == Y)/float(len(Y))


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.ensemble import BaggingClassifier
    X, Y = make_classification(n_samples=500, n_features=20, n_informative=4)
    X_train = X[:300, :]
    Y_train = Y[:300]
    X_test = X[300:, :]
    Y_test = Y[300:]

    max_depth = 5
    rho = 0.2
    nb_epoch = 2
    n_iter = 5
    n_estimators = 1000

    gae = Generate_Adversarial_Ensemble(max_depth, rho, n_iter, nb_epoch, n_estimators)
    gae.fit(X_train, Y_train)
    print gae.predict_proba(X_test)
    print gae.predict(X_test)
    print gae.score(X_test, Y_test)

    #Bagging
    bc = BaggingClassifier(n_estimators=1000)
    bc.fit(X_train, Y_train)
    print bc.score(X_test, Y_test)
