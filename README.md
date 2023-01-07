# Home-Credit-Default-Risk
## Final Project for Fall 2022 course: CSCI-P 556 - Applied Machine Learning
## Project made in collaboration with anarkhed@iu.edu and shssingh@iu.edu

## Abstract
In this project, we are attempting to predict whether a specific client will repay the loan they have taken out. The main goal of this phase is to build a Multi-Layer Perception model in PyTorch for loan default classification. The initial phase was all about understanding the project requirements, understanding the HCDR dataset and building project plan. In phase 2, we performed data preprocessing (such as using imputer for missing values), Exploratory Data Analysis, some feature engineering (Polynomial Features) and created baseline models for prediction. Phase 3 involved more feature engineering (aggregated features based on existing numerical and categorical features) and Hyperparameter tuning using GridSearchCV. In this phase, we first performed Random Undersampling to remove the imbalance in the data. Feature importance was performed to derive the relative importance of features during prediction (using Random Forest Algorithm and feature_importances_ property). We have created separate pipelines for each model which have been fine tuned further for comparative analysis. Furthermore, we have trained an ANN model based on the features extracted in previous phases. We were able to extract the best performance with the developed neural network (F1 Score = 0.262235) after testing on the entire training data. Upon Kaggle Submission, the best performing model was again the developed neural network  with an AUC private score of 0.732 and public score of 0.7361.

## Project Description

The course project is based on the [Home Credit Default Risk (HCDR)  Kaggle Competition](https://www.kaggle.com/c/home-credit-default-risk/). The goal of this project is to predict whether or not a client will repay a loan. In order to make sure that people who struggle to get loans due to insufficient or non-existent credit histories have a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.


## Some of the challenges

1. Dataset size 
   * (688 meg compressed) with millions of rows of data
   * 2.71 Gig of data uncompressed
* Dealing with missing data
* Imbalanced datasets
* Summarizing transaction data

## Dataset and Project Description

The main training section of the data is compiled in a CSV labeled “application_train.csv” and the counterpart test data in “application_test.csv”. The training data set’s columns differ from the test set only in that it includes the target column for whether the client repaid the loan. There are 121 feature columns in the application dataset which include both categorical, such as gender, and numerical features, such as income. 

There are several other datasets that correlate to the application dataset, and they build off each other. The “bureau.csv” dataset contains a row for each previous credit the client had before the application date corresponding to the loan in the application dataset. The “bureau_balance.csv” dataset contains each month of history there is data for the previous credits mentioned in the bureau dataset. In this way, these subsidiary datasets build off the application dataset and each other. These subsidiary datasets contain categorical values and positive and negative numerical values. There are 6 subsidiary datasets in total, but the main dataset we will be looking at is the application set, as this is what the test set is based on. 

The following diagram shows the relation between various datasets provided in the problem:

![dataset.jpeg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/dataset.jpeg?token=GHSAT0AAAAAAAAASPN2WZEPWWV5E3JBZX5MY6AM4OQ)

**application_{train|test}.csv**
This is the main dataset consisting of test and training samples. It consists of personal details about the loan applicant and connects with other sets using SK_ID_CURR attribute.
**previous_application.csv**
This dataset consists of information about applicant’s previous loans in Home credit. It contains attributes such as type of loan, status of loan, down-payment etc.
**instalments_payments.csv**
This dataset contains information about repayment of previous loans.
**credit_card_balance.csv**
This dataset consists of transaction information related to Home Credit credit cards.
**POS_CASH_balance.csv**
This dataset consists of entries that define the status of previous credit of the individual at Home Credit which includes consumer credit and cash loans.
**bureau.csv**
This dataset consists of information of the individual’s previous credit history at other financial institutions that were reported to the credit bureau.

## Exploratory Data Analysis

- Proper EDA can be viewed at this [**link**](https://jovian.ai/hirenrupchandani/home-credit-default-risk#C35)

## Feature Engineering
- This phase involves preparing features from the application_train.csv and application_test.csv datasets by introducing 3rd-degree polynomial features. We took a subset of that new data to train and evaluate the baseline models. We also used other secondary tables that are provided to us in the HCDR dataset. We have primarily focused on previous_application.csv and bureau.csv tables in this phase.

- We will divide this phase of feature engineering into three parts:
	- Feature Engineering in **previous_application.csv** table.
	- Feature Engineering in **brueau.csv** table.
	- Merging the two secondary tables with **application_train.csv** and **application_test.csv** tables.
	
#### For previous_application.csv table:
- In this table, we have created three types of aggregated features (min, max, mean) for the following columns:

`'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
                'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
                'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'NAME_PAYMENT_TYPE',
                'CNT_PAYMENT', 'DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                'DAYS_LAST_DUE', 'DAYS_TERMINATION'`
                
- For `AMT_APPLICATION`, we also created a `range_AMT_APPLICATION` which takes the range from the maximum and minimum of the said feature.
- Then we performed a simple merge function of the newly prepared aggregated table with the `application_train.csv` and `application_test.csv` tables based on the `SK_ID_CURR` identifier.

#### For bureau.csv table:

- We see that 263,491 applicants have their data in this table.
- We first created a previous_loan_count feature for all these applicants based on the `SK_ID_BUREAU` feature.
- Then we created five types of aggregated features (count, mean, max, min, sum) for the rest of the columns by performing a groupby on the `SK_ID_CURR` feature:
- After appropriately renaming the columns, we performed a merge of the aggregated bureau table with the `application_train.csv` and `application_test.csv` tables based on the `SK_ID_CURR` identifier.

- At the end of this series of operations, we are left with 331 columns in the training dataset (merged application_train.csv) and 330 columns in the testing dataset (merged application_test.csv).

## Hyperparameter Tuning

- In order to efficiently perform a hyperparameter tuning grid search, we needed to use a roundabout method.
- Our goal for the hyperparameter tuning was to find which classifier worked the best, and with what hyperparameters.
- We tested logistic regression (LogisticRegression), random forests (RandomForestClassifier), decision trees (DecisionTreeClassifier), the GradientBoostingClassifier, CatBoostClassifier, LightGBMClassifier, AdaBoostClassifier, and lastly the XGBoostClassifier.
- We trained these models on the downsampled dataset
- Additionally, we tested many hyperparameters for each of these models. We used the Scikit-Learn library's GridsearchCV to run our grid search.

- **Logistic Regression**: we tested the penalty type and the penalty coefficient.
- **Random Forest**: we tested the number of trees, the max depth of each tree, and the minimum number of samples allowed for each leaf node.
- For decision trees, we tested the max depth and the minimum samples for leaf nodes. 
- **Gradient Boosting Classifier**: we tested the number of trees, the minimum samples for leaf nodes, the max depth, and the subsample parameter that controls the proportion of samples to be used to fit each tree.
- **Catboost classifier**: we tried a combination of n_estimators, learning rates, number of iterations, and max_depth.
- **Light gradient boost classifier**: tried different values of l1 and l2 regularizations, as well as the max_depth of the model.
- **XGBoost classifier**: we tested the number of trees, the max depth, the subsample, a 'colsample_bytree' parameter which controls the proportion of columns used to construct each tree, the alpha determining the amount of L1 regularization, the lambda for L2 regularization, and gamma, which determines how much loss reduction is needed to partition further down the tree.
 
- Some of these classifiers take a lot of time to train but eventually they yielded good results.

## Data Leakage

- Data Leakage has been a concern in this dataset since the beginning of this project.

- There are multiple ways to ensure that there is no leakage of data among the subsets of data.
  - After performing feature engineering, we dropped the SK_ID_CURR feature among the X_train, X_valid, X_test, and X_kaggle_test sets and performed an intersection among the datasets to check for any overlaps of data.
  - The goal is to check if there are any rows that are same in the either dataset. Fortunately, **there were none**, indicating that there is no such row that might affect the held-out datasets.
  
![data leakage](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/DataLeakage.png?token=GHSAT0AAAAAAAAASPN227IUGIK3IVOF3ADCY6AHIUA)
  
  - We already had a held-out dataset whose score is calculated on kaggle, but we still performed segregation on the engineered datasets by creating X_valid and X_test sets.
  - Among these two, X_valid was used as an evaluation set, that may cause some leakage between X_valid and X_train features, but ensured there is no leakage between X_test and X_train datasets, and subsequently no leakage between X_kaggle_test and X_train datasets.
  - There might be some information leakage from X_train set over to the other datasets due to imputation and PCA operations but that is as far as the data leakage goes.

- As for **cardinal sins of machine learning**, we are confident to check that we have not made such mistakes and assumptions on the data that might affect the EDA, interpretation, or the modelling and predictions of the plethora of models that we have implemented.

## Pipelines

- Just like last phase, creating separate pipelines will help us to fill in missing values and perform scaling on numerical features.  
- Our pipeline uses SimpleImputer to fill in the missing values with the median being used for the numerical pipeline and the most frequent value being used for the categorical pipeline.
- One Hot Encoding is used in the categorical pipeline to convert categorical values into integers for improving the performance of our models.
- Standard Scaler is used in the numerical pipeline for scaling purposes. - - The two pipelines are then combined into a single ‘`data_prep_pipeline`’ 

![Pipeline.jpeg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/Pipeline.jpeg?token=GHSAT0AAAAAAAAASPN32ZR6VZQ2ZPOIJLXKY6AMVVQ)

- Additionally, to avoid imbalance class issues, we have undersampled the data (using RandomUnderSampler class) in this phase to make it balanced.

![rr2.png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/rr2.png?token=GHSAT0AAAAAAAAASPN2K72RI2T2U5KWVINWY6AMW3Q)

- The target feature exhibits the following distribution:

![donut_distribution_target.jpg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/donut_distribution_target.jpg?token=GHSAT0AAAAAAAAASPN2AZZJP7MAQQN65XFUY6AMWJA)

- We have then performed Random Forest feature importance (assigning scores to input features) on the undersampled data.
- This has been done to determine the relative importance of features while predicting the target variable. 
- RandomForestClassifier class has been used to implement Random Forest algorithm for feature importance. After training the model, we have used the feature_importances_ property to retrieve the relative importance scores for each input feature.

![rr1.png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/rr1.png?token=GHSAT0AAAAAAAAASPN3PR3LR5PD5RIBMMQYY6AMXKQ)

- Some of the important features are as follows:

![FeaturePlot.png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/FeaturePlot.png?token=GHSAT0AAAAAAAAASPN3ZLRM7JKMZYKQH6QEY6AMYDQ)

- Before sending the data for modeling and experiments, we introduce Principal Component Analysis to the pipeline with an explained variance of 0.85.
- This is then followed by hyperparameter tuning of the models.

- Multilayer Perceptron Pipeline:

    - First data imputation is performed followed by converting  data into Tensors and then finally passed to the MLP model.
	- For our MLPs, we will use BinaryCrossEntropy Loss for optimization:
    
$$ BinaryCrossEntropy = 
H_p(q) = -\frac{1}{N}\sum_{i=1}^n y_i.log(p(y_i))+ (1-y_i).log(1-p(y_i)) $$

- We primarily focus on these two performance metrics and loss functions:

$$ F1 = \frac{2*Precision*Recall}{Precision+Recall} = \frac{2*TP}{2*TP+FP+FN} $$ 

$$  Sensitivity = Recall = \frac{TP}{TP+FN}  $$
$$  Specificity = \frac{TN}{FP+TN}  $$
AUC is calculated as the Area Under the $Sensitivity$(TPR)-$(1-Specificity)$(FPR) Curve.


## Experimental results

- After Downsampling the training dataset and appropriately tuning the models, we get the following performacne report:

![performance_other_models.jpeg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/performance_other_models.jpeg?token=GHSAT0AAAAAAAAASPN3ANT5434ZRVB2YSXSY6AMZYA)

- Moving a step further, we created various MLPs (more than 15) and these are the top 5 models based on best Test F1 score:

![Sigmoid_Relu.jpeg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/Sigmoid_Relu.jpeg?token=GHSAT0AAAAAAAAASPN2PKVAKGWIEIW6WCI4Y6AMZBA)

- We observe that some MLPs overfit after 10-15 epochs so we limit them to 10 epochs. This is how the training/validation loss curve looks like over the epochs:

![Epoch-plot.png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/Epoch-plot.png?token=GHSAT0AAAAAAAAASPN3LX7G3AREWXL6FSJMY6AMZ3Q)

- After selecting the best model, we see that our best performing model looks like this:

![nn (1).png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/nn%20(1).png?token=GHSAT0AAAAAAAAASPN2UETHAEAY7WAZNZ5UY6AMY4A)


- The best scores were obtained for **PyTorch FCNN**, **LightGBM**, and **LogisticRegression**.


## Discussion

- This phase of the project primarily focused on two things:
  - Reduce the complexity of the dataset and tune the classical and ensemble models
  - Prepare a Multi-Layer Perceptron using PyTorch library and select the best set of hyperparameters.

- **Focus #1** involved making an important decision - to reduce the number of rows of the training dataset and select the important features.

- RandomUnderSampler and RandomForestClassifier's feature_importances_ enables us to achieve these with ease.

- **LightGBM** and **LogisticRegression** performed very well over this data and gave a score of above **0.725** on the kaggle scoreboard.

- **Focus #2** pushed our skills to the limit. Even though we had a reference code from HW11, we were unable to improvise on the F1 score for a long time.

- A lot of tweaking enabled us to finally perfect the architecture and run the code on the entire dataset with ease.

- The model started overfitting after 10 epochs due to complexity of dataset so we grounded the networks to 10-15 epochs.

- Our **best FCNN** was then determined and after another set of tweaks with the model architecture, we were able to score a kaggle public score of **0.736**, best so far amongst all the models.

## Conclusion

- In this project, we are attempting to predict whether a specific client will repay the loan they have taken out. We had performed Exploratory Data Analysis, some feature engineering and created baseline models in the initial two phases. In the previous phase (i.e. phase 3), we merged the datasets, performed some more feature engineering (deriving polynomial and aggregated features) and hyperparameter tuning obtaining the best F1 score of 0.148820 for Decision Tree Classifier. 
- Upon Kaggle submission in **past** phases, Logistic Regression was the best performing model with an ROC-AUC score of approximately 72%. 
- In the **current phase**, we performed Random Undersampling to remove the imbalance in the dataset. We also performed Feature Importance to derive the relative importance scores of input features during prediction (using Random Forest Algorithm) and reduce the complexity of the dataset. 
- In this last phase, we have implemented a Multi-Layer Perception (MLP) model for loan default classification.  We obtained the best performance with the developed MLP model with an F1 Score of 0.262235.
- Upon Kaggle Submission, the best performing model was again the developed neural network  with an AUC private score of 73.21% and public score of 73.61%. We were able to obtain our best performance by implementing the MLP model in the current phase.
- For **Future** scope, we can try improving the accuracy by increasing the complexity of  the neural network and performing more hyperparameter tuning.


## Kaggle Submission

- These are the final submissions of all the tuned models:

![Kaggle_other_models.jpeg](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/Kaggle_other_models.jpeg?token=GHSAT0AAAAAAAAASPN2BKROK4DMRV3WURCUY6AM3FQ)

- Our Final best Predicting Model yields the best final score:

![nn_prediction.png](https://raw.github.iu.edu/hrupchan/Home-Credit-Default-Risk/main/Images/nn_prediction.png?token=GHSAT0AAAAAAAAASPN2LR2EJLHEHMJ4ZWXWY6AM3EQ)
