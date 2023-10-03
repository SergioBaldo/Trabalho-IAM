class Wrapper:
    def __init__(self, sm, df_train, df_test, cwd):
        self.sm = sm
        self.df_train = df_train
        self.df_test = df_test
        self.cwd = cwd

    def select(self, df_train, df_test, cwd):
        self.sm.search(self.df_train, self.df_test, self.cwd)
