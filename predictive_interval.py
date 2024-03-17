import pandas as pd
import numpy as np
import os
os.environ["NUMEXPR_MAX_THREADS"] = "64"


from pgmpy.inference import ApproxInference, VariableElimination
from pgmpy.factors.discrete import State

from query2df import query2df


def predictive_interval(model_bn, col_var, row_var, n_samples = 25000 , q_length = 40, target_variable = "CRC", path_to_data = "interval_df"):
    if not os.path.exists(path_to_data):
        os.mkdir(path_to_data)
    
    model_infer = VariableElimination(model_bn)
    model_approx_infer = ApproxInference(model_bn)

    mod_columns = model_bn.states[col_var]
    mod_rows = model_bn.states[row_var]

    df_hom_inf = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_hom_sup = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_hom= pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_hom_str = pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_muj_inf = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_muj_sup = pd.DataFrame(columns = mod_columns, index = mod_rows )
    df_muj = pd.DataFrame(columns = mod_columns, index = mod_rows )

    df_muj_str = pd.DataFrame(columns = mod_columns, index = mod_rows )

    A_hom = model_infer.query(variables = [target_variable], evidence={"Sex": "M"}, show_progress = False)
    A_muj = model_infer.query(variables = [target_variable], evidence={"Sex": "W"}, show_progress = False)


    for i in range(len(model_bn.states["Sex"])):
        Sex = model_bn.states["Sex"][i]

        for j in range(len(mod_rows)):
            row = mod_rows[j]

            for k in range(len(mod_columns)):
                column = mod_columns[k]

                ev = [State("Sex", Sex), State(col_var,column), State(row_var,row)]

                q_point = np.log(1 - query2df(  model_infer.query(variables=[target_variable], 
                                                                            evidence={"Sex": Sex, col_var: column, row_var: row},
                                                                            show_progress = False)  ,   verbose = 0)["p"][0])
                q = np.zeros(q_length)

                df_partial_samples = pd.DataFrame(columns = ["Sex" , col_var, row_var])
                for n in range(n_samples): df_partial_samples = pd.concat([df_partial_samples, pd.DataFrame(data = {"Sex":[i], col_var:[k], row_var: [j]})])
                df_partial_samples.reset_index(drop=True, inplace = True)
                df_partial_samples = df_partial_samples.astype("int32")

                import time

                for k in range(len(q)):

                    start_time = time.time()
                    end_time = time.time()
                    print("time taken:", end_time - start_time)

                    start_time = time.time()
                    q[k] = np.log(1 - query2df(model_approx_infer.query(variables=[target_variable], evidence= {"Sex": Sex, col_var: column, row_var: row},n_samples = n_samples, show_progress = False), verbose = 0)["p"][0])
                    end_time = time.time()
                    print("time taken:", end_time - start_time)

                a = np.sort(q)

                if Sex == "M":
                    df_hom_inf[column][row] = round( a[round(q_length*4 / 100)] - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )
                    df_hom_sup[column][row] = round( a[round(q_length*94.9 / 100)] - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )

                    df_hom_str[column][row] = f"[ {df_hom_inf[column][row]}, {df_hom_sup[column][row]}]"

                    print(f'Risk interval for men with {col_var} = {column} and {row_var} = {row} is: {df_hom_str[column][row]} ({n_samples} samples and interval of size {q_length})')

                    df_hom[column][row] = round( q_point - np.log( 1 - query2df(A_hom, verbose = 0)["p"][0]) , 3 )

                    print(f'Pointwise estimation of the risk:', df_hom[column][row])
                    
                else:
                    df_muj_inf[column][row] = round( a[round(q_length*4 / 100)] - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )
                    df_muj_sup[column][row] = round( a[round(q_length*94.9 / 100)] - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )

                    df_muj_str[column][row] = f"[ {df_muj_inf[column][row]}, {df_muj_sup[column][row]}]"

                    print(f'Risk interval for women with {col_var} = {column} and {row_var} = {row} is: {df_muj_str[column][row]} ({n_samples} samples and interval of size {q_length})')

                    df_muj[column][row] = round( q_point - np.log( 1 - query2df(A_muj, verbose = 0)["p"][0]) , 3 )

                    print(f'Pointwise estimation of the risk:', df_muj[column][row])

                df_hom.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}.csv")
                df_muj.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}.csv")

                df_hom_str.to_csv(f"{path_to_data}/df_hom_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")
                df_muj_str.to_csv(f"{path_to_data}/df_muj_{col_var}_{row_var}_{q_length}_{n_samples}_interval.csv")

    return df_hom, df_hom_str, df_muj, df_muj_str
