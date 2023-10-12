class Wrapper:
    """
    A class that acts as a wrapper for different selection methods.

    Attributes:
        selection_method (object): The selection method to be used.
    """

    def __init__(self, selection_method):
        """
        Initializes a new Wrapper instance.

        Args:
            selection_method (object): An object representing the selection method.
        """
        self.selection_method = selection_method

    def select(self):
        """
        Perform the selection using the specified selection method.

        Calls the `search` method of the selection method to initiate the selection process.
        """
        self.selection_method.search()
