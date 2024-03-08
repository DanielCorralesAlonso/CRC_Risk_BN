import pandas as pd
import numpy as np
# import plotly.express as px
import math
import seaborn as sns
# import plotly.io as pio
from pgmpy.inference import VariableElimination
from query2df import query2df
from heatmap import heatmap

def pointwise_risk_mapping(model_bn, var1, var2):

    model_infer = VariableElimination(model_bn)
    
    mod_columns = model_bn.states[var1]
    mod_rows = model_bn.states[var2]

    df_hom = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_muj = pd.DataFrame(columns = mod_columns, index = mod_rows )

    A = model_infer.query(variables = ["CRC"], evidence = {})
    A_hom = model_infer.query(variables = ["CRC"], evidence={"Sex": "M"})
    A_muj = model_infer.query(variables = ["CRC"], evidence={"Sex": "W"})


    for Sex in model_bn.states["Sex"]:
        for row in mod_rows:
            for column in mod_columns:

                q = model_infer.query(variables=["CRC"], evidence={"Sex": Sex, var1: column, var2: row})
                
                if Sex == "M":
                    df_hom[column][row] = round( np.log(1 - query2df(q, verbose = 0)["p"][0]) - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )
                else:
                    df_muj[column][row] = round( np.log(1 - query2df(q, verbose = 0)["p"][0])  - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )
                    
    return df_hom, df_muj



def heatmap_plot_and_save(df, model_bn, col_var, row_var, q_length = 40, n_samples = 25000, path_to_data = "interval_df", interval = False, save = True):

    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.color_palette("RdBu_r", n_colors = 256) #sns.diverging_palette(220, 20, center = "dark", n=n_colors)

    if interval:
        df_hom = pd.read_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}.csv", index_col=[0])
        df_hom_interval = pd.read_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv", index_col=[0])

        df_muj = pd.read_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}.csv", index_col=[0])
        df_muj_interval = pd.read_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv", index_col=[0])
        
    else:   
        df_hom, df_muj = pointwise_risk_mapping(model_bn, col_var, row_var)


    magnitude_data_hom = df[df["Sex"] == "M"][[col_var, row_var]].value_counts(sort=False)
    magnitude_data_muj = df[df["Sex"] == "W"][[col_var, row_var]].value_counts(sort=False)

    corr = pd.melt(df_hom.reset_index(), id_vars='index') 
    corr.columns = ['x', 'y', 'value']

    if interval:
        corr_int = pd.melt(df_hom_interval.reset_index(), id_vars='index') 
        corr_int.columns = ['x', 'y', 'value']
    else:
        corr_int = corr

    fig = heatmap(
        x=corr['y'],
        y=corr['x'],
        text = corr_int["value"],
        title = f"{col_var} vs {row_var} risk map for men",
        size = magnitude_data_hom, # pd.to_numeric(corr['value']).abs(),
        color = pd.to_numeric(corr["value"]),
        color_range = [-2, 2], 
        palette = palette,
        size_scale = 5000,
        
    )

    if save:
        fig.savefig(f"images/point_risk_map_men_{col_var}_{row_var}.png")

    corr = pd.melt(df_muj.reset_index(), id_vars='index') 
    corr.columns = ['y', 'x', 'value']

    if interval:
        corr_int = pd.melt(df_muj_interval.reset_index(), id_vars='index') 
        corr_int.columns = ['x', 'y', 'value']
    else:
        corr_int = corr

    fig = heatmap(
        x=corr['x'],
        y=corr['y'],
        text = corr_int["value"],
        size = magnitude_data_muj, # pd.to_numeric(corr['value']).abs(),
        color = pd.to_numeric(corr["value"]),
        color_range = [-2, 2], 
        palette = palette,
        size_scale = 5000,
        title = f"{col_var} vs {row_var} risk map for women",
    )

    if save:
        fig.savefig(f"images/point_risk_map_women_{col_var}_{row_var}.png")


        df_hom.to_csv(f"riskmap_datasets/men_pointwise_est_risk_map_{col_var}_{row_var}.csv")
        df_muj.to_csv(f"riskmap_datasets/women_pointwise_est_risk_map_{col_var}_{row_var}.csv")

        magnitude_data_hom.to_csv(f"riskmap_datasets/men_magnitudes_est_risk_map_{col_var}_{row_var}.csv")
        magnitude_data_muj.to_csv(f"riskmap_datasets/women_magnitudes_est_risk_map_{col_var}_{row_var}.csv")