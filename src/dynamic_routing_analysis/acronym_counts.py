import pandas as pd

def plot_units_areas_histogram_for_session(units:pd.DataFrame) -> None:
    """
    Plots a histogram of the areas for units from a session
    
    Parameters
    ----------
    units: pd.DataFrame
        The units table from the nwb
    """
    
    print(units['location'].value_counts())
    units['location'].value_counts().plot(kind='bar')