import pandas as pd

# Class which compares timing
class TimingComparator:

    # Initialise by creating arrays containing each midi value for each potential note
    def __init__(self):
        super().__init__()

    # Return the mean difference between the primer and generated timing values
    def getMeanDifference(self, timing_in: pd.Series, timing_out: pd.Series):
        mean_in = timing_in.mean()
        mean_out = timing_out.mean()

        return max(mean_in - mean_out, mean_out - mean_in)

    # Return the standard deviation difference between the primer and generated timing values
    def getStdDifference(self, timing_in: pd.Series, timing_out: pd.Series):
        std_in = timing_in.std()
        std_out = timing_out.std()

        return max(std_in - std_out, std_out - std_in)

    # Return the variance difference between the primer and generated timing values
    def getVarDifference(self, timing_in: pd.Series, timing_out: pd.Series):
        var_in = timing_in.var()
        var_out = timing_out.var()

        return max(var_in - var_out, var_out - var_in)
