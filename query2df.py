import itertools
import numpy as np
import pandas as pd
from tabulate import tabulate

def query2df(query, variables=None, groupby=None, verbose=3):
    """Convert query from inference model to a dataframe.

    Parameters
    ----------
    query : Object from the inference model.
        Convert query object to a dataframe.
    variables : list
        Order or select variables.
    groupby: list of strings (default: None)
        The query is grouped on the variable name by taking the maximum P value for each catagory.

    Returns
    -------
    df : pd.DataFrame()
        Dataframe with inferences.

    """
    if ((groupby is not None) and np.any(np.isin(groupby, variables))):
        # Needs to be set to true.
        groupby = list(np.array(groupby)[np.isin(groupby, variables)])
    else:
        if verbose>=2: print('[bnlearn] >Warning: variable(s) [%s] does not exists in DAG.' %(groupby))
        groupby=None

    states = []
    getP = []
    for value_index, prob in enumerate(itertools.product(*[range(card) for card in query.cardinality])):
        states.append(prob)
        getP.append(query.values.ravel()[value_index])

    df = pd.DataFrame(data=states, columns=query.scope())
    df['p'] = getP

    # Convert the numbers into variable names
    for col in query.scope():
        df[col] = np.array(query.state_names[col])[df[col].values.astype(int)]

    # Order or filter on input variables
    if variables is not None:
        # Add Pvalue column
        variables = variables + ['p']
        df = df[variables]

    # groupby
    if groupby is not None:
        df = df.groupby(groupby).apply(lambda x: x.loc[x['p'].idxmax()])
        df.reset_index(drop=True, inplace=True)

    # Print table to screen
    if verbose>=3:
        print('[bnlearn] >Data is stored in [query.df]')
        print(tabulate(df, tablefmt="grid", headers="keys"))

    return df