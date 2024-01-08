import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.inference import VariableElimination
import random
from query2df import query2df

def influential_variables(data, target, model_bn, n_random_trials = 50):
    model_infer = VariableElimination(model_bn)

    data = data.reset_index(drop=True)
    ordered_variables = ["Sex","Age", "SD", "SES", "PA", "Depression", "Smoking", "BMI","Alcohol","Anxiety", "Diabetes", "Hyperchol.", "Hypertension"]

    dict_impact_patient = dict.fromkeys(list(range(len(data))))

    for _ in range(n_random_trials):
        random.shuffle(ordered_variables)

        print(ordered_variables)

        dropped = list(model_bn.nodes())
        for elem in list(model_bn.get_ancestral_graph(target).nodes):
            if elem == target:
                continue
            else:
                dropped.remove(elem)
        
        diff_vect = np.zeros((data.shape[0], len(data.iloc[0].drop(labels = dropped).dropna())))
        crc_sample_aux = np.zeros_like(diff_vect)
        for i in range(data.shape[0]):

            sample = data.iloc[i].drop(labels = dropped).dropna()

            j = 0
            list_elem = []
            def_variables = [x for x in ordered_variables if x not in dropped]

            for elem in [x for x in ordered_variables if x not in dropped]:
                list_elem.append(elem)
                sample_aux = sample[list_elem].copy()
                sample_aux_dict = sample_aux.to_dict()
                q_sample_aux = model_infer.query(variables=[target], evidence = sample_aux_dict)

                crc_sample_aux[i,j] =  np.log(1 - query2df(q_sample_aux, verbose = 0)["p"][0].copy())

                j += 1

            impact_aux = pd.DataFrame(columns=def_variables)
            aux = np.zeros(len(sample))
            
            
            for j in range(len(diff_vect[i])):
                if j == 0:

                    sample_CRC = model_infer.query(variables=["CRC"])
                    aux[j] = (crc_sample_aux[i,j] - np.log(1 - query2df(sample_CRC, verbose = 0)["p"][0].copy()))   / np.abs(np.log(1 - query2df(sample_CRC, verbose = 0)["p"][0].copy())) * 100
                
                    continue

                else:
                    aux[j] = (crc_sample_aux[i,j] - crc_sample_aux[i,j-1])  /  np.abs( crc_sample_aux[i,j-1]) * 100
                                        
                            
            impact_aux = pd.DataFrame([aux], columns = def_variables)

            dict_impact_patient[i] = pd.concat([dict_impact_patient[i], impact_aux], axis = 0)


    for i in range(data.shape[0]):
        if i==0:
            grouped_data = pd.concat([data.iloc[i].rename(index = 'Evidence'), dict_impact_patient[i].replace(0,float('nan')).median(axis = 0).rename('Influence')], axis = 1)
        else:
            grouped_data_aux = pd.concat([data.iloc[i].rename(index = 'Evidence'), dict_impact_patient[i].replace(0,float('nan')).median(axis = 0).rename('Influence')], axis = 1)
            grouped_data = pd.concat([grouped_data, grouped_data_aux], axis = 0)
                
    def combine_categories(row):
                return f"{row.name} = {row['Evidence']}"

    grouped_data['Influential Variable and Reason'] = grouped_data.apply(combine_categories, axis=1)      
        

    heatmap_data = grouped_data[["Influential Variable and Reason", "Influence"]].sort_values(by = ["Influence"], ascending = False).copy().set_index(["Influential Variable and Reason"])

    heatmap_data = heatmap_data.groupby("Influential Variable and Reason").mean().sort_values(by = ["Influence"], ascending = False)

    heatmap_data.dropna(inplace=True) # Remove CRC row

    plt.figure(figsize=(2,8))
    ax = sns.heatmap(heatmap_data, cmap='RdBu_r', annot=True, fmt='.1f', linewidths=1, center = 0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    plt.savefig('images/influential_variables.png')

    return heatmap_data