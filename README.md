# Colorectal Risk Mapping Through Bayesian Networks

- [Colorectal Risk Mapping Through Bayesian Networks](#colorectal-risk-mapping-through-bayesian-networks)
  * [Project description](#project-description)
  * [Directory strucure](#directory-strucure)
    + [File descriptions](#file-descriptions)
  * [Contact](#contact)



## Project description

Only about 14 % of the susceptible EU citizens
 participate in colorectal cancer (CRC) screening programs despite of being the
 third most common type of cancer worldwide. The development of predictive
 models can facilitate personalized CRC predictions which can be embedded in
 decision-support tools that facilitate screening and treatment recommendations.
 This paper, published in _Computer Methods and Programs in Biomedicine_, develops a predictive model that aids in characterizing risk groups
 and assessing the influence of a variety of risk factors on the population. 

[Find the paper in the following link](https://doi.org/10.1016/j.cmpb.2024.108407)

![Bayesian network](./images/cancer_colorectal_learned_bds.png)

## Directory strucure

### File descriptions
- **main.py**: Contains the pipeline of the project which can be summarized in the following steps:

  1. **Data Preprocessing**: It reads a CSV file named `df_2012.csv` from the `data` directory, and applies some preprocessing to the data using a function from the `preprocessing` module.
  
  2. **Structure Learning**: It uses the `HillClimbSearch` and `BDsScore` classes from the `pgmpy` library to estimate the structure of a Bayesian Network from the data. The structure learning process can be influenced by a target variable, a blacklist of edges, and a list of fixed edges, all of which are specified in a `config` module.
  
  3. **Model Visualization**: It visualizes the learned Bayesian Network structure using the `pyAgrum` library, and saves the visualizations as PNG images in the `images` directory. It creates two visualizations: one for the prior network (before learning) and one for the posterior network (after learning).
  
  4. **Parameter Estimation**: It estimates the parameters of the Bayesian Network using a function from the `parameter_estimation` module. This process involves updating the prior parameters of the network based on the data.
  
  5. **Model Statistics**: It calculates and saves some statistics of interest about the model, such as the mean and variance of the counts per year. It also calculates the 90% posterior predictive interval for these counts.
  
  6. **Risk Mapping**: It creates and saves a heatmap of the risk associated with different variables in the model. This is done using a function from the `risk_mapping` module. If specified in the `config` module, it also calculates an approximation of the posterior predictive intervals by sampling.
  
  7. **Influential Variables**: It identifies the variables in the model that have the most influence on a target variable. This is done using a function from the `influential_variables` module.
  
  8. **Model Evaluation**: Finally, it evaluates the performance of the model in classifying a target variable in a separate dataset (`df_2016.csv`). This is done using a function from the `evaluation_classification` module.

- **config.py**: This file contains several Python dictionaries and lists that are used to configure the behavior of a Bayesian Network model:

  1. `inputs`: This dictionary specifies the target variable for the model ("CRC"), whether to calculate intervals (False), and the number of random trials to perform (10).
  
  2. `structure`: This dictionary contains two lists:
     - `black_list`: A list of variable pairs that should not be connected in the Bayesian Network.
     - `fixed_edges`: A list of variable pairs that should always be connected in the Bayesian Network.
  
  3. `node_color`: This dictionary assigns a weight to each variable, which could be used for visual representation or importance ranking. The weights range from 0.1 to 0.4.
  
  4. `pointwise_risk_mapping`: This dictionary specifies the column variable ("Age") and the row variable ("BMI") for the pointwise risk mapping.
  
  5. `interval_risk_mapping`: This dictionary specifies the column variable ("Age") and the row variable ("BMI") for the interval risk mapping.
  
  6. `interval_path`: This dictionary specifies the path ("prueba22nov/") where the interval risk mapping results will be saved.
   
 
- **preprocessing.py**: Add necessary preprocessing steps
 
- **parameter_estimation.py**:
  1. `create_pscount_dict_from_model(model_bn, card_dict, prior_weight, size_prior_dataset)`: This function generates a dictionary of pseudo counts from a Bayesian Network model. It reshapes the conditional probability distributions (CPDs) of each variable in the model and scales them by the size of the prior dataset and a specified factor.
  
  2. `prior_update_iteration(model_bn, card_dict, pscount_dict, size_prior_dataset)`: This function performs a prior update iteration on the Bayesian Network model using data from different years. It reads data, preprocesses it, fits the model using a Bayesian Estimator with a Dirichlet prior, updates the pseudo counts dictionary, and stores the count tables for each year. It returns the updated model and a dictionary of counts per year.
 
- **risk_mapping.py**:
  1. `pointwise_risk_mapping(model_bn, var1, var2)`: This function calculates the pointwise risk mapping for two variables (`var1` and `var2`) in a Bayesian Network model. It queries the model for the probability of "CRC" given different combinations of the two variables and "Sex", and stores the results in two dataframes. The risk is calculated as the logarithm of the difference between the probability of "CRC" given the evidence and the marginal probability of "CRC". The results are rounded to three decimal places.
  
  2. `heatmap_plot_and_save(...)`: This function generates a heatmap based on the provided data and visual parameters. If the `save` flag is set to `True`, it also saves the generated heatmap as a PNG image in a specified directory. The filename of the image is based on the provided title.
 
- **influential_variables.py**:
  1. `influential_variables(data, target, model_bn, n_random_trials = 50)`: Calculates the influence of different variables on a target variable in a Bayesian Network model. It performs multiple random trials, shuffling the variables, identifying non-ancestors of the target, and calculating difference vectors for each row in the dataframe.

- **evaluation_classification.py**:
  1. `evaluation_classification(df_test, model_bn, test_var = "CRC")`: This function evaluates the classification performance of a Bayesian Network model on a test dataset. It initializes a Variable Elimination object with the model, then iterates over the rows of the test dataframe. For each row, it drops the test variable, converts the row to a dictionary, and queries the model for the probability of the test variable given the evidence in the row. It stores the predicted probabilities in a list `y_prob_pred`. Finally, it calculates the false positive rate, true positive rate, and thresholds for the Receiver Operating Characteristic (ROC) curve using the true labels and the predicted probabilities.

The functions `BayesianEstimator.py` and `BayesianNetwork.py` are modified versions of the original functions from `pgmpy` which would need to be replaced in this library for the main code to run properly. The reason behind this is to save the unnormalized tables of counts and used them to calculate the mean and variance of the empirical distributions.

## Contact
For any further consultation please contact _danielcorralesalonso@gmail.com_ 
