"""Pure Python Naive Bayes classifier

Simple nb_classifier.

Initial Author: Kevin Molloy and Patrick Campfield And Thomas Lane
"""

import numpy as np
import math
import scipy.stats as stats

class NBClassifier:
    """
    A naive bayes classifier for use with categorical and real-valued attributes/features.

    Attributes:
        classes (list): The set of integer classes this tree can classify.
        smoothing_flag (boolean): Indicator whether or not to perform
                                  Laplace smoothing
        feature_dists (list):  A placeholder for each feature/column in X
                               that holds the distributions for that feature.
    """

    def __init__(self, smoothing_flag=False):
        """
        NBClassifier constructor.

        :param smoothing: for discrete elements only
        """
        if smoothing_flag:
            self.smoothing = 1
        else:
            self.smoothing = 0

        """
        feature_dists is a list of dictionaries, one for each feature in X.
        The dictionary is envisioned to be for each class label, and
       the key might be:
          -- for continuous features, a tuple with the distribution
          parameters for a Gaussian (mu, std) 
          -- for discrete features, another dictionary where the keys 
             are the individual domain values for the feature
             and the value is the computed probability from the training data 
        """
        self.feature_dists = []
        self.ALPHA = 1.0


    def get_smoothing(self):
        if self.smoothing:
            return True
        else:
            return False




    def fit(self, X, X_categorical, y):
        """
        Construct the NB using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features).
                  This is the training data.
        :param X_categorical: numpy boolean array with length num_features.
                              True values indicate that the feature is discrete.
                              False values indicate that the feature is continuous.
        :param y: Numpy integer array with length num_samples
                  These are the training labels.

        :return: Stores results in class variables, nothing returned.

        An example of how my dictionary looked after running fit on the
        loan classification problem in the textbook without smoothing:
        [{0: {'No': 0.5714285714285714, 'Yes': 0.42857142857142855},
         1: {'No': 1.0}   },
        {0: {'Divorced': 0.14285714285714285, 'Married': 0.5714285714285714, 'Single': 0.2857142857142857},
         1: {'Divorced': 0.3333333333333333, 'Single': 0.6666666666666666}   },
        {0: (110.0, 54.543560573178574, 2975.0000000000005),
         1: (90.0, 5.0, 25.0)}]
        """

        ## Need a category for each column in X
        assert(X.shape[1] == X_categorical.shape[0])
        ## each row in training data needs a label
        assert(X.shape[0] == y.shape[0])

        self.classes = list(set(y))
        self.priors = {}
        self.X_categorical = X_categorical
        X_class = np.array([ X[y == c] for c in self.classes], dtype=object)

        if self.smoothing:
            for col, j in enumerate(X_categorical):
                self.priors = {}
                if j:
                    unqInCol = list(set(X[:,col]))
                    for i in self.classes:
                        unq, count = np.unique(X_class[i][:,col], return_counts=True)
                        self.priors[i] = {unq[k]: (count[k] + self.ALPHA)/(np.sum(count) + len(unqInCol)) for k in range(len(unq))}
                        if len(unq) != len(unqInCol):
                            extra = {}
                            for k in unqInCol:
                                if k not in unq:
                                    extra[i] = {k: (self.ALPHA)/(np.sum(count) + len(unq)+self.ALPHA)}
                            self.priors[i].update(extra[i])
                    self.feature_dists =  np.append(self.feature_dists, self.priors)
                else:
                    for i in self.classes:              
                        mean = np.mean(X_class[i][:,col].astype(np.float64))
                        std = np.std(X_class[i][:,col].astype(np.float64), ddof=1)
                        var = np.var(X_class[i][:,col].astype(np.float64), ddof=1)
                        self.priors[i] = (mean, std, var)
                    self.feature_dists =  np.append(self.feature_dists, self.priors)
        else:
            for col, j in enumerate(X_categorical):
                self.priors = {}

                if j:
                    unqInCol = list(set(X[:,col]))
                    for i in self.classes:
                        unq, count = np.unique(X_class[i][:,col], return_counts=True)
                        self.priors[i] = {unq[k]: (count[k])/(np.sum(count)) for k in range(len(unq))}
                        if len(unq) != len(unqInCol):
                            extra = {}
                            for k in unqInCol:
                                if k not in unq:
                                    extra[i] = {k: np.divide(0,np.sum(count) + len(unqInCol))}
                            self.priors[i].update(extra[i])
                else:
                    for i in self.classes:
                        mean = np.mean(X_class[i][:,col].astype(np.double))
                        std = np.std(X_class[i][:,col].astype(np.double), ddof=1)
                        var = np.var(X_class[i][:,col].astype(np.double), ddof=1)
                        self.priors[i] = (mean, std, var)
                self.feature_dists =  np.append(self.feature_dists, self.priors)


    def feature_class_prob(self,feature_index, class_label, x):
        """
        Compute a single conditional probability.  You can call
        this function in your predict function if you wish.

        Example: For the loan default problem:
            feature_class_prob(1, 0, 'Single') returns 0.5714

        :param feature_index:  index into the feature set (column of X)
        :param class_label: the label used in the probability (see return below)
        :param x: the data value

        :return: P(class_label | feature(fi) = x) the probability
        """
        feature_dist = self.feature_dists[feature_index]
        # validate feature_index
        assert feature_index < self.X_categorical.shape[0], \
            'Invalid feature index passed to feature_class_prob'

        # validate class_label
        assert class_label < len(self.classes), \
            'invalid class label passed to feature_class_prob'

        if self.X_categorical[feature_index]:
            return feature_dist[class_label][x]
        else:
            return stats.norm.pdf(x, feature_dist[class_label][0], feature_dist[class_label][1])


    def predict(self, X):
        """
        Predict labels for test matrix X

        Parameters/returns
        ----------
        :param X:  Numpy array with shape (num_samples, num_features)
        :return: Numpy array with shape (num_samples, )
            Predicted labels for each entry/row in X.
        """

        ## validate that x contains exactly the number of features
        assert(X.shape[1] == self.X_categorical.shape[0])
        predicted_labels = np.array([])
        for x in X:
            label_percents = np.array([])
            class_labels = np.array([])
            for class_label in self.classes:  
                for feature_index in range(self.X_categorical.size):         
                    label_percents = np.append(label_percents, 
                        self.feature_class_prob(feature_index=feature_index, class_label=class_label, x=x[feature_index]))
                    class_labels = np.append(class_labels, class_label)
            predicted_labels = np.append(predicted_labels, class_labels[np.argmax(label_percents)])
            
                    
        return predicted_labels


def nb_demo():
    ## data from table Figure 4.8 in the textbook

    X = np.array([['Yes', 'Single',125],
                  ['No', 'Married', 100],
                  ['No', 'Single', 70],
                  ['Yes', 'Married', 120],
                  ['No', 'Divorced', 95],
                  ['No', 'Married', 60],
                  ['Yes', 'Divorced', 220],
                  ['No', 'Single', 85],
                  ['No', 'Married', 75],
                  ['No', 'Single', 90]
                 ])

    ## first two features are categorical and 3rd is continuous
    X_categorical = np.array([True, True, False])

    ## class labels (default borrower)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

    nb = NBClassifier(smoothing_flag=True)

    nb.fit(X, X_categorical, y)
    # test_pt = np.array([['No', 'Married', 120]])
    # yhat = nb.predict(test_pt)
    # x_pt = 120.
    # class_label = 0
    # p = nb.feature_class_prob(feature_index=2, class_label=class_label, x=x_pt)

    # # the book computes this as 0.0016 * alpha
    # print('Predicted value for someone who does not a homeowner,')
    # print('is married, and earns 120K a year is:', yhat)


def main():
    nb_demo()

if __name__ == "__main__":
    main()
