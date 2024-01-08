from pgmpy.factors.discrete.CPD import TabularCPD
import scipy.stats as st
import numpy as np
import pandas as pd

def from_counts_to_mean_and_variance(counts_table):
    # Counts table has a TabularCPD format
    alpha_j_list = counts_table.values
    alpha = sum(alpha_j_list)
    means = []
    vars = []
    for j in range(len(alpha_j_list)):
        alpha_j = alpha_j_list[j]
        means.append(alpha_j / alpha)
        vars.append( (alpha_j * (alpha - alpha_j)) / ((alpha**2)*(alpha + 1)) )

    return means, vars



def from_counts_to_quantiles(counts_table):
    ppf_05 = []
    ppf_95 = []
    for j in range(len(counts_table.values)):

        alpha = counts_table.values[j]
        beta = sum(counts_table.values) - counts_table.values[j]

        ppf_05.append(st.beta.ppf(0.05, alpha, beta))
        ppf_95.append(st.beta.ppf(0.95, alpha, beta))

    ppf_95 = np.array(ppf_95)
    ppf_05 = np.array(ppf_05)
    
    return ppf_95, ppf_05



def csv_quantiles(model_bn, counts_per_year):
    years = counts_per_year.keys()

    nodes_list = list(model_bn.nodes())

    for year in years:
        for j in range(len(nodes_list)):
            node = nodes_list[j]
            num_states = len(model_bn.states[node])
            parents = model_bn.get_parents(node)
            parents_card = [len(model_bn.states[par]) for par in parents]
            ppf_95, ppf_05 = from_counts_to_quantiles(counts_per_year[year][j])
            
            tab_95 = TabularCPD(node, num_states, ppf_95.reshape(num_states,-1), evidence = parents, evidence_card= parents_card)
            tab_05 = TabularCPD(node, num_states, ppf_05.reshape(num_states,-1), evidence = parents, evidence_card= parents_card)

            tab_95.to_csv(f"bounds/{node}_{year}_95.csv")
            tab_05.to_csv(f"bounds/{node}_{year}_05.csv")

            df_95 = pd.DataFrame(ppf_95.reshape(num_states,-1))
            df_05 = pd.DataFrame(ppf_05.reshape(num_states,-1))

            df_range = pd.DataFrame(columns=df_95.columns, index=df_95.index)

            for row in range(df_95.shape[0]):
                for column in range(df_95.shape[1]):
                    df_range[column][row] = f"[{np.round(df_05[column][row],5)}, {np.round(df_95[column][row],5)}]"

            df_range.to_csv(f"bounds/{node}_{year}_range.csv")