import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

from pgmpy.models import BayesianNetwork

from preprocessing import preprocessing


def create_pscount_dict_from_model(model_bn, card_dict, prior_weight, size_prior_dataset):
    model_infer = VariableElimination(model_bn)
    dict_ps = {}
    for variable in list(model_infer.variables):
        dict_ps[variable] = (model_infer.model.get_cpds(variable).values.reshape(model_bn.get_cardinality(variable), card_dict[variable]) * size_prior_dataset*50).tolist()
    
    return dict_ps


def prior_update_iteration(model_bn, card_dict, pscount_dict, size_prior_dataset, folder_path = "data"):
    years = [2012,2013,2014,2015]
    counts_per_year = dict.fromkeys(years, None)
    prior_weight_list = [1, 50, 100, 500]
    for i in range(len(years)):
        df_aux = pd.read_csv(f"{folder_path}/df_{years[i]}.csv", index_col = None).copy()
        df_aux = preprocessing(df_aux)
        
        counts_tables = model_bn.fit(df_aux, estimator=BayesianEstimator,weighted = False, prior_type = 'dirichlet', pseudo_counts = pscount_dict, n_jobs = 4)

        pscount_dict = create_pscount_dict_from_model(model_bn, card_dict, prior_weight_list[i], size_prior_dataset)

        print(f"Year {years[i]} update completed")
        counts_per_year[years[i]] = counts_tables

    model_infer = VariableElimination(model_bn)
    return model_infer, counts_per_year