import os
import pandas as pd
import numpy as np

from pgmpy.models import BayesianNetwork

from pgmpy.estimators import HillClimbSearch, BDsScore
from pgmpy.factors.discrete import State

from preprocessing import preprocessing
import config


# ---- Read CSV and short preprocessing ---------------------------------
dir = os.getcwd()
file_path = os.path.join(dir, "data/df_2012.csv")

df = pd.read_csv(file_path, index_col = None)
df = preprocessing(df)

print("Successful data read")
# -----------------------------------------------------------------------


# ---- Structure Learning -----------------------------------------------
target = config.inputs["target"]
blck_lst = config.structure["black_list"]
fxd_edges = config.structure["fixed_edges"]

from pgmpy.estimators import HillClimbSearch, BDsScore

est = HillClimbSearch(data = df)
model = est.estimate(scoring_method=BDsScore(df, equivalent_sample_size = 5), fixed_edges=fxd_edges, black_list=blck_lst)
print("Successful structure learning")
# -----------------------------------------------------------------------


# ----- Save learned model ----------------------------------------------

if not os.path.exists("images"):
        os.mkdir("images")

if not os.path.exists("riskmap_datasets"):
        os.mkdir("riskmap_datasets")

# PRIOR NET
import pyAgrum as gum
import pyAgrum.lib.image as gumimage
import matplotlib.pyplot as plt

bn_gum = gum.BayesNet()
bn_gum.addVariables(list(df.columns))
bn_gum.addArcs(list(fxd_edges))

path = "images/"
file_name = str('cancer_colorrectal_prior') + '.png'
file_path = os.path.join(path,file_name)

gumimage.export(bn_gum, file_path, size = "20!",
                nodeColor = config.node_color,
                            )

# POSTERIOR NET
bn_gum_2 = gum.BayesNet()
bn_gum_2.addVariables(list(df.columns))
bn_gum_2.addArcs(list(model.edges))

arcColor_mine = dict.fromkeys(bn_gum_2.arcs(), 0.3)
for elem in list(bn_gum.arcs()):
    arcColor_mine[elem] = 1

path = "images/"
file_name = str('cancer_colorrectal_learned_bds') + '.png'
file_path = os.path.join(path,file_name)

gumimage.export(bn_gum_2, file_path, size = "20!",
                 nodeColor = config.node_color,
              
                cmapArc =  plt.get_cmap("hot"),
                arcColor= arcColor_mine )

print("Successful graphic models save")
# -----------------------------------------------------------------------


# ---- Parameter estimation ---------------------------------------------

from pgmpy.models import BayesianNetwork
model_bn = BayesianNetwork(model)


card_dict = dict.fromkeys(model.nodes, 1)
for node in card_dict.keys():
    for parent in model.get_parents(node): 
        card_dict[node] *= len(set(df[parent]))


# Prior parameters
size_prior_dataset = len(df) / 10000
pscount_dict = {
    "Sex": [[df["Sex"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset],[df["Sex"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]], 
    "Age": [[df["Age"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset],[ df["Age"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset],[df["Age"].value_counts(normalize=True).sort_index()[2] *size_prior_dataset],[df["Age"].value_counts(normalize=True).sort_index()[3] *size_prior_dataset]], #,[1.65 * size_prior_dataset]], 
    'BMI': [[df["BMI"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict["BMI"], [df["BMI"].value_counts(normalize=True).sort_index()[1] *size_prior_dataset]*card_dict['BMI'], [df["BMI"].value_counts(normalize=True).sort_index()[2] *size_prior_dataset]*card_dict['BMI'], [df["BMI"].value_counts(normalize=True).sort_index()[3] *size_prior_dataset]*card_dict['BMI']],
    'Alcohol': [[df["Alcohol"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict["Alcohol"], [df["Alcohol"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict["Alcohol"]],
    'Smoking': [[df["Smoking"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict["Smoking"], [df["Smoking"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict["Smoking"], [df["Smoking"].value_counts(normalize=True).sort_index()[2] * size_prior_dataset]*card_dict["Smoking"]], 
    'PA': [[df["PA"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict["PA"], [df["PA"].value_counts(normalize=True).sort_index()[2] * size_prior_dataset]*card_dict["PA"]],
    'SD': [[df["SD"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict["SD"], [df["SD"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict["SD"],[df["SD"].value_counts(normalize=True).sort_index()[2] * size_prior_dataset]*card_dict["SD"]],
    'SES': [[df["SES"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict["SES"], [df["SES"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict["SES"], [df["SES"].value_counts(normalize=True).sort_index()[2] * size_prior_dataset]*card_dict["SES"]],
    'Depression': [[df["Depression"].value_counts(normalize=True).sort_index()[0] *size_prior_dataset]*card_dict['Depression'], [df["Depression"].value_counts(normalize=True).sort_index()[1]*size_prior_dataset]*card_dict['Depression']], 
    'Anxiety': [[df["Anxiety"].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]* card_dict['Anxiety'], [df["Anxiety"].value_counts(normalize=True).sort_index()[1]* size_prior_dataset]* card_dict['Anxiety']] , 
    'Diabetes': [[df["Diabetes"].value_counts(normalize=True).sort_index()[0]* size_prior_dataset]*card_dict['Diabetes'], [df["Diabetes"].value_counts(normalize=True).sort_index()[1]* size_prior_dataset]*card_dict['Diabetes']], 
    'Hypertension': [[df["Hypertension"].value_counts(normalize=True).sort_index()[0]* size_prior_dataset]*card_dict['Hypertension'], [df["Hypertension"].value_counts(normalize=True).sort_index()[1]* size_prior_dataset]*card_dict['Hypertension']] ,
    'Hyperchol.': [[df["Hyperchol."].value_counts(normalize=True).sort_index()[0] * size_prior_dataset]*card_dict['Hyperchol.'], [df["Hyperchol."].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict['Hyperchol.']] ,  
    'CRC': [[df["CRC"].value_counts(normalize=True).sort_index()[0]* size_prior_dataset]*card_dict['CRC'], [df["CRC"].value_counts(normalize=True).sort_index()[1] * size_prior_dataset]*card_dict['CRC']] , 
    }


model_bn = BayesianNetwork(model)

from parameter_estimation import prior_update_iteration

model_infer, counts_per_year = prior_update_iteration(model_bn, card_dict, pscount_dict = pscount_dict, size_prior_dataset=size_prior_dataset)

print("Successful parameter estimation")
# ----------------------------------------------------------------------


# ---- Save model statistics of interest (90% prediction interval) -----
from table_statistics import from_counts_to_mean_and_variance, csv_quantiles

if not os.path.exists("bounds"):
        os.mkdir("bounds")

mean, var = from_counts_to_mean_and_variance( counts_per_year[2012][0] )

csv_quantiles(model_bn, counts_per_year=counts_per_year)

print("Successful statistics save")
# -----------------------------------------------------------------------


# ---- Risk mapping -----------------------------------------------------
from risk_mapping import heatmap_plot_and_save
from prediction_interval import prediction_interval

col_var = config.pointwise_risk_mapping["col_var"]
row_var = config.pointwise_risk_mapping["row_var"]

heatmap_plot_and_save(df, model_bn, col_var, row_var)


# If calculate interval = True, an approximation of the prediction intervals will be 
# by sampling. However, it is a task that requires relatively large computation and time 
# resources, so we encourage to use the example case available.

calculate_interval = config.inputs["calculate_interval"]
if calculate_interval:
    prediction_interval(model_bn, col_var, row_var, path_to_data = "interval_df/")

col_var = config.interval_risk_mapping["col_var"]
row_var = config.interval_risk_mapping["row_var"]

heatmap_plot_and_save(df, model_bn, col_var, row_var, interval = True)

print("Successful risk mapping")

# -----------------------------------------------------------------------



# ---- Influential variables --------------------------------------------
from influential_variables import influential_variables

df_pos = df[df[target] == True].copy()

# Increase the n_random_trials to get meaningful results.
heatmap_data = influential_variables(data=df_pos, target=target, model_bn = model_bn, n_random_trials = config.inputs["n_random_trials"])

print("Successful influential variables")
# -----------------------------------------------------------------------



# ---- Evaluation of the model ------------------------------------------
from evaluation_classification import evaluation_classification

df_remaining = pd.read_csv("data/df_2016.csv")
df_remaining = preprocessing(df_remaining)

evaluation_classification(df_remaining, model_bn)

print("Successful evaluation of the model")
# -----------------------------------------------------------------------