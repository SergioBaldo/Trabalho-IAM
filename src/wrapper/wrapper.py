class Wrapper:
    """
    A class that acts as a wrapper for different selection methods.

    Attributes:
        clf (object): The classifier or model to be used.
        sm (object): The selection method to be used.
        X (array-like): The feature data.
        y (array-like): The target data.
    """

    def __init__(self, clf, sm, X, y):
        """
        Initializes a new SelectionWrapper instance.

        Args:
            clf (object): An object representing the classifier or model.
            sm (object): An object representing the selection method.
            X (array-like): The feature data.
            y (array-like): The target data.
        """
        self.clf = clf
        self.sm = sm
        self.X = X
        self.y = y

    def select(self, X, y):
        """
        Perform the selection using the specified selection method.

        Calls the `search` method of the selection method to initiate the selection process.
        """
        self.sm.search(self.clf, self.X, self.y)
