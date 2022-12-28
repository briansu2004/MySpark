#!/usr/bin/env python
# coding: utf-8

# # Real-world data

# The section reports the performances that are obtained on real-world data using imbalanced learning techniques. The dataset is the same as in [Chapter 3, Section 5](Baseline_FDS_RealWorldData). Results are reported following the methodology used in the previous sections with simulated data. 
# 
# We first report the performances for cost-sensitive techniques, varying the class weight for decision trees and logistic regression models. We then report the performances for resampling techniques using decision trees, and varying the imbalance ratio with SMOTE, RUS, and a combination of SMOTE and RUS. We finally report the results using ensemble techniques, varying the imbalance ratio with Bagging and Random Forests models, and varying the class weight with an XGBoost model.
# 

# In[1]:


# Initialization: Load shared functions

# Load shared functions
get_ipython().system('curl -O https://raw.githubusercontent.com/Fraud-Detection-Handbook/fraud-detection-handbook/main/Chapter_References/shared_functions.py')
get_ipython().run_line_magic('run', 'shared_functions.py')
#%run ../Chapter_References/shared_functions.ipynb


# ## Cost-sensitive

# We followed the methodology reported in Section 6.2 ([](Cost_Sensitive_Learning_Transaction_Data)), saving the results in a `performances_cost_sensitive_real_world_data.pkl` pickle file. The performances and execution times can be retrieved by loading the pickle file.

# In[2]:


filehandler = open('images/performances_cost_sensitive_real_world_data.pkl', 'rb') 
(performances_df_dictionary, execution_times) = pickle.load(filehandler)


# ### Decision tree
# 
# The results for decision tree models are reported below. The tree depth was set to 6 (providing the best performances as reported in [Chapter 5](Model_Selection_RWD_Decision_Trees)). We varied the class weight in the range 0.01 to 1, with the following set of possible values: $[0.01, 0.05, 0.1, 0.5, 1]$.
# 

# In[3]:


performances_df_dt=performances_df_dictionary['Decision Tree']
summary_performances_dt=get_summary_performances(performances_df_dt, parameter_column_name="Parameters summary")

get_performances_plots(performances_df_dt, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Class weight for the majority class",
                       summary_performances=summary_performances_dt)


# We recall that a class weight of 1 consists in giving the same weight to positive and negative classes, whereas a class weight of 0.01 consists in giving 100 times more weight to the positive class (thus favoring the detection of fraud instances). We also note that a class weight of 1 provides the same results as in [Chapter 5](Model_Selection_Decision_Tree). 
# 
# The results show that decreasing the class weight allows to increase the performances in terms of AUC ROC, but decreases the performances in terms of Average Precision and CP@100, particularly for very low values (close to 0.01).
# 
# The performances as a function of the best parameters are summarized below.

# In[4]:


summary_performances_dt


# These results follow the same trends as those obtained with the [simulated data](Cost_Sensitive_Learning_Transaction_Data): Cost-sensitive learning is effective for improving AUC ROC performances, but detrimental to Average Precision. 

# ### Logistic regression
# 
# The results for logistic regression are reported below. The regularization parameter C was set to 0.1 (providing the best performances as was reported in [Chapter 5](Model_Selection_RWD_Logistic_Regression)). 

# In[5]:


performances_df_lr=performances_df_dictionary['Logistic Regression']
summary_performances_lr=get_summary_performances(performances_df_lr, parameter_column_name="Parameters summary")

get_performances_plots(performances_df_lr, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Class weight for the majority class",
                       summary_performances=summary_performances_lr)


# In[6]:


summary_performances_lr


# Similar to decision trees, lowering the class weight of the majority class provides a boost of performances in terms of AUC ROC. The impact on Average Precision and CP@100 is however mitigated: the only noticeable impact is an improvement of the Average Precision on validation data, which however comes at the cost of a higher variance (as is visible from the large confidence interval).

# ## Resampling
# 
# We followed the methodology reported in Section 6.3 ([](Resampling_Strategies_Transaction_Data)), saving the results in the `performances_resampling_real_world_data.pkl` pickle file. The performances and execution times can be retrieved by loading the file. Performances were assessed for SMOTE, RUS, and a combined resampling with SMOTE and RUS. All the experiments relied on decision tree models, whose tree depth was set to 6 (providing the best performances as was reported in [Chapter 5](Model_Selection_RWD_Decision_Trees)). 
# 

# In[7]:


filehandler = open('images/performances_resampling_real_world_data.pkl', 'rb') 
(performances_df_dictionary, execution_times) = pickle.load(filehandler)


# ### SMOTE
# 
# The results for SMOTE are reported below. The imbalance ratio was varied in the range 0.01 to 1, with the following set of possible values: $[0.01, 0.05, 0.1, 0.5, 1]$. We recall that the higher the imbalance ratio, the stronger the resampling. An imbalance ratio of 0.01 yields a distribution close to the original one (where the percentage of frauds is close to 0.25%). An imbalance ratio of 1 yields a distribution that contains as many positive instances as negative instances.  
# 

# In[8]:


performances_df_SMOTE=performances_df_dictionary['SMOTE']
summary_performances_SMOTE=get_summary_performances(performances_df_SMOTE, parameter_column_name="Parameters summary")

get_performances_plots(performances_df_SMOTE, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Imbalance ratio",
                       summary_performances=summary_performances_SMOTE)


# In[9]:


summary_performances_SMOTE


# The results show that the benefits of SMOTE are mitigated. Creating new synthetic instances of the positive class tends to increase AUC ROC performances (left plot). It however comes with a decrease of performances for both Average Precision and CP@100 metrics. These results are in line with those observed on simulated data (Section 6.3, [](Resampling_Strategies_Transaction_Data_Oversampling)).

# ### Random undersampling
# 
# The results for random undersampling (RUS) are reported below. 

# In[10]:


performances_df_RUS=performances_df_dictionary['RUS']
summary_performances_RUS=get_summary_performances(performances_df_RUS, parameter_column_name="Parameters summary")

get_performances_plots(performances_df_RUS, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Imbalance ratio",
                       summary_performances=summary_performances_RUS)


# In[11]:


summary_performances_RUS


# Similarly, RUS allows to improve performances in terms of AUC ROC, but comes with a noticeable decrease of performances in terms of AP and CP@100. These results are also in line with those observed on simulated data (Section 6.3, [](Resampling_Strategies_Transaction_Data_RUS)).

# ### Combining SMOTE with undersampling
# 
# We finally report the results for combined resampling (SMOTE followed by RUS). 

# In[12]:


performances_df_combined=performances_df_dictionary['Combined']
summary_performances_combined=get_summary_performances(performances_df_combined, parameter_column_name="Parameters summary")

get_performances_plots(performances_df_combined, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Imbalance ratio",
                       summary_performances=summary_performances_combined)


# In[13]:


summary_performances_combined


# Again, we observe that resampling allows to improve performances in terms of AUC ROC, but decreases performances in terms of AP and CP@100. The results are in line with those observed on simulated data (Section 6.3, [](Resampling_Strategies_Transaction_Data_Combining)).

# ## Ensembling
# 
# We followed the methodology reported in Section 6.4 ([](Ensembling_Strategies_Transaction_Data)), saving the results in a `performances_resampling_real_world_data.pkl` pickle file. The performances and execution times can be retrieved by loading the pickle file.
# 

# In[14]:


filehandler = open('images/performances_ensembles_real_world_data.pkl', 'rb') 
(performances_df_dictionary, execution_times) = pickle.load(filehandler)


# Performances were assessed for balanced bagging, balanced random forest, and weighted XGBoost.
# 
# ### Baseline
# 
# For the [baseline](Ensembling_Strategies_Transaction_Data_Baseline), the hyperparameters were chosen as follows:
# 
# * Bagging and random forest: 100 trees, with a maximum depth of 10. These were shown to provide the best performances for random forests in [Chapter 5, Model Selection - Random forest](Model_Selection_RWD_RF).
# * XGBoost: 100 trees, with a maximum depth of 6, and a learning rate of 0.1. These were shown to provide the best trade-off in terms of performances in [Chapter 5, Model Selection - XGBoost](Model_Selection_RWD_XGBoost). 
# 
# The baseline performances are reported in the table below. It is worth noting that the performances for random forest and XGBoost are the same as those reported in [Chapter 5, Model Selection - Random forest](Model_Selection_RWD_RF) and [Chapter 5, Model Selection - XGBoost](Model_Selection_RWD_XGBoost), respectively.
# 

# In[15]:


performances_df_baseline_bagging=performances_df_dictionary['Baseline Bagging']
summary_performances_baseline_bagging=get_summary_performances(performances_df_baseline_bagging, parameter_column_name="Parameters summary")

performances_df_baseline_rf=performances_df_dictionary['Baseline RF']
summary_performances_baseline_rf=get_summary_performances(performances_df_baseline_rf, parameter_column_name="Parameters summary")

performances_df_baseline_xgboost=performances_df_dictionary['Baseline XGBoost']
summary_performances_baseline_xgboost=get_summary_performances(performances_df_baseline_xgboost, parameter_column_name="Parameters summary")

summary_test_performances = pd.concat([summary_performances_baseline_bagging.iloc[2,:],
                                       summary_performances_baseline_rf.iloc[2,:],
                                       summary_performances_baseline_xgboost.iloc[2,:],
                                      ],axis=1)
summary_test_performances.columns=['Baseline Bagging', 'Baseline RF', 'Baseline XGBoost']


# In[16]:


summary_test_performances


# XGBoost was observed to provide better performances than random forest across all performance metrics, as was already reported in [](Model_Selection_RWD_Comparison). The performances of bagging were on par with random forest in terms of AUC ROC, but lower in terms of Average Precision and CP@100. 

# ### Balanced bagging
# 
# Similar to [](Ensembling_Strategies_Transaction_Data_Bagging) with simulated data, the imbalance ratio (`sampling_strategy` parameter) was parametrized to take values in the set $[0.01, 0.05, 0.1, 0.5, 1]$ for the model selection procedure. The number of trees and maximum tree depth were set to 100 and 10 (as with baseline bagging).

# In[17]:


performances_df_balanced_bagging=performances_df_dictionary['Balanced Bagging']
performances_df_balanced_bagging


# Let us summarize the performances to highlight the optimal imbalance ratio, and plot the performances as a function of the imbalance ratio for the three performance metrics.

# In[18]:


summary_performances_balanced_bagging=get_summary_performances(performances_df_balanced_bagging, parameter_column_name="Parameters summary")
summary_performances_balanced_bagging


# In[19]:


get_performances_plots(performances_df_balanced_bagging, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Imbalance ratio",
                       summary_performances=summary_performances_balanced_bagging)


# The results show that increasing the imbalance ratio leads to a decrease of both Average Precision and CP@100. The trend is different with AUC ROC, where increasing the imbalance ratio first leads to a slight improvement of the metric, before reaching a plateau. It is worth noting that the results are qualitatively similar to [](Ensembling_Strategies_Transaction_Data_Bagging) with simulated data for AUC ROC and Average Precision. 

# ### Balanced random forest
# 
# Similar to [](Ensembling_Strategies_Transaction_Data_RF) with simulated data, the imbalance ratio (`sampling_strategy` parameter) was parametrized to take values in the set $[0.01, 0.05, 0.1, 0.5, 1]$ for the model selection procedure. The number of trees and maximum tree depth were set to 100 and 10 (as with baseline random forest).

# In[20]:


performances_df_balanced_rf=performances_df_dictionary['Balanced RF']
performances_df_balanced_rf


# Let us summarize the performances to highlight the optimal imbalance ratio, and plot the performances as a function of the imbalance ratio for the three performance metrics.

# In[21]:


summary_performances_balanced_rf=get_summary_performances(performances_df_balanced_rf, parameter_column_name="Parameters summary")
summary_performances_balanced_rf


# In[22]:


get_performances_plots(performances_df_balanced_rf, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Imbalance ratio",
                       summary_performances=summary_performances_balanced_rf)


# The results follow the same trends as with balanced bagging: increasing the imbalance ratio is detrimental to Average Precision and CP@100, but can slightly increase AUC ROC. The results are qualitatively similar to [](Ensembling_Strategies_Transaction_Data_RF) with simulated data for AUC ROC and Average Precision.

# ### Weighted XGBoost
# 
# Finally, similar to [](Ensembling_Strategies_Transaction_Data_Weighted_XGBoost) with simulated data, we varied the `scale_pos_weight` parameter to take values in the set $[1,5,10,50,100]$. The same hyperparameters as the baseline XGBoost were otherwise kept (100 trees with a maximum depth of 6, and a learning rate of 0.1).
# 

# In[23]:


performances_df_weighted_xgboost=performances_df_dictionary['Weighted XGBoost']
performances_df_weighted_xgboost


# Let us summarize the performances to highlight the optimal imbalance ratio, and plot the performances as a function of the class weight for the three performance metrics.

# In[24]:


summary_performances_weighted_xgboost=get_summary_performances(performances_df_weighted_xgboost, parameter_column_name="Parameters summary")
summary_performances_weighted_xgboost


# In[25]:


get_performances_plots(performances_df_weighted_xgboost, 
                       performance_metrics_list=['AUC ROC', 'Average precision', 'Card Precision@100'], 
                       expe_type_list=['Test','Validation'], expe_type_color_list=['#008000','#FF0000'],
                       parameter_name="Class weight",
                       summary_performances=summary_performances_weighted_xgboost)


# Contrary to balanced bagging and balanced random forest, increasing the class weight of the minority class allows to slightly improve the performances in terms of Average Precision and CP@100. Improvements are only observed for a slight increase of the class weight (from 1 to 5). Higher values lead to slight decreases of performances. For AUC ROC, the optimal class weight is found to be 1 (equal cost for the minority and majority classes). 

# ### Summary
# 
# Let us finally summarize in a single table the results on the real-world dataset. Performance metrics are reported row-wise, while ensemble methods are reported column-wise. 

# In[26]:


summary_test_performances = pd.concat([summary_performances_baseline_bagging.iloc[2,:],
                                       summary_performances_balanced_bagging.iloc[2,:],
                                       summary_performances_baseline_rf.iloc[2,:],
                                       summary_performances_balanced_rf.iloc[2,:],
                                       summary_performances_baseline_xgboost.iloc[2,:],
                                       summary_performances_weighted_xgboost.iloc[2,:],
                                      ],axis=1)
summary_test_performances.columns=['Baseline Bagging', 'Balanced Bagging', 
                                   'Baseline RF', 'Balanced RF', 
                                   'Baseline XGBoost', 'Weighted XGBoost']


# In[27]:


summary_test_performances


# The best improvements were observed for balanced bagging and balanced random forest, for which better performances were obtained compared to baseline bagging and baseline random forest. However, as we noted for the [simulated data](Ensembling_Strategies_Transaction_Data), the benefits of resampling are most likely due to a higher diversity of the trees making up the ensembles, leading to a decrease of the overfitting phenomenon. In particular, the optimum imbalance ratio for Average Precision and CP@100 was found to be the lowest one (0.01), which shows that the best strategy for these metrics was to avoid rebalancing the training sets.
# 
# On the contrary, we observed that rebalancing the training sets could slightly improve the performances in terms of AUC ROC. The improvements were observed for imbalance ratios ranging from 0.1 to 0.5, leading to a slight increase of around 1% of the AUC ROC (from 0.91 to 0.92). Besides allowing a slight increase in performances, it is worth noting that rebalancing the dataset with undersampling techniques could speed up computation times by up to 20%.
# 
# As for the results obtained on [simulated data](Ensembling_Strategies_Transaction_Data_Summary), these experiments suggest that rebalancing can help improve performances in terms of AUC ROC or speed up the training time of an ensemble. It however appeared that keeping all of the training data was the best strategy if Average Precision and CP@100 are the performance metrics to optimize.
# 
# Overall, the best performances were obtained with XGBoost for the three metrics. As for the [simulated data](Ensembling_Strategies_Transaction_Data_Summary), modifying the class weight through weighted XGBoost did not allow to significantly improve performances, illustrating the robustness of XGBoost in class imbalance scenarios.
