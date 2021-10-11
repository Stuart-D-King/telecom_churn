## Can You Hear Me Now?
## Telecom Churn Prediction Data Challenge

### Table of Contents
* Problem Statement & Objectives
* Getting the Data
* Data Cleaning
* Data Exploration
* Data Transformation
* Model Building
* Hyperparameter Tuning
* Run the Scripts
* Conclusion & Next Steps

### Problem Statement & Objective
A telecom provider would like to better understand its customer base and would like to know who are most likely to drop out from their service. The objective of this exercise is to understand what factors are driving customer churn and how the company can improve customer retention.

A supervised classification modeling approach will be used to try to identify the underlying patterns and signal, if any, present within the data. It is assumed no previous efforts have been made regarding estimating churn propensity. A baseline result/statistic of the most common outcome will be used to gauge performance of more complex models.

### Getting the Data
The data is provided in three separate `.csv` files and is stored in the `data/` directory.

    data/churn_data.csv
    data/customer_data.csv
    data/internet_data.csv

A separate data dictionary is provided that provides a description for each variable.

| variable_name    | definition                                                                       |
|:-----------------|:---------------------------------------------------------------------------------|
| CustomerID       | The unique ID of each customer                                                   |
| Gender           | The gender of a person                                                           |
| SeniorCitizen    | Whether a customer can be classified as a senior citizen.                        |
| Partner          | If a customer is married/ in a live-in relationship.                             |
| Dependents       | If a customer has dependents (children/ retired parents)                         |
| Tenure           | The time for which a customer has been using the service.                        |
| PhoneService     | Whether a customer has a landline phone service along with the internet service. |
| MultipleLines    | Whether a customer has multiple lines of internet connectivity.                  |
| InternetService  | The type of internet services chosen by the customer.                            |
| OnlineSecurity   | Specifies if a customer has online security.                                     |
| OnlineBackup     | Specifies if a customer has online backup.                                       |
| DeviceProtection | Specifies if a customer has opted for device protection.                         |
| TechSupport      | Whether a customer has opted for tech support of not.                            |
| StreamingTV      | Whether a customer has an option of TV streaming.                                |
| StreamingMovies  | Whether a customer has an option of Movie streaming.                             |
| Contract         | The type of contract a customer has chosen.                                      |
| PaperlessBilling | Whether a customer has opted for paperless billing.                              |
| PaymentMethod    | Specifies the method by which bills are paid.                                    |
| MonthlyCharges   | Specifies the money paid by a customer each month.                               |
| TotalCharges     | The total money paid by the customer to the company.                             |
| Churn            | This is the target variable which specifies if a customer has churned or not.    |

The three data file were read into memory using Pandas, each variable name was converted from camel case to snake case, and the three dataframes were combined using the `customer_id` field, which was subsequently dropped. The resultant dataframe contained 7,043 records.

### Data Cleaning
While attempting to convert the `total_charges` variable to a numeric datatype, it was discovered a handful of rows contained an empty space as data. This value was replaced with `0` which allowed for the variable to be fully converted to a `float` datatype.

As suggested in the data dictionary, and verified through visual inspection, many variables were `Yes/No` strings indicating membership in a particular group. These variables were converted to `1/0` representations. Variables with additional categories were one-hot-encoded using the Pandas `get_dummies` function and then joined back to the original dataframe, dropping the original dummified variables.

A correlation matrix was then compiled to analyze which features were correlated with the target variable. As a form of feature selection, variables with less than +/- 0.15 correlation were dropped from the dataset. In addition, the `total_charges` feature was removed due to the observed correlation with customer `tenure`.

### Data Exploration
Based on the data provided, the overall churn rate is **26.54%**. A naive estimator that predicts the most common outcome (not churned) would have accuracy equal to the inverse of the churn rate (100% - 26.54% = 73.46%). This value is used as a **baseline result** while building and testing models. The baseline result provides a meaningful reference point from which to compare estimators.

Pivoting the data helped arrive at some descriptive statistics and findings.
- Not surprisingly, customers with higher tenure churn less often than customers with lower tenure
- Average monthly charges are $13 more for churned customers than current customers
- Senior citizens are almost twice as likely to churn as younger customers and are charged $18 more per month on average
- Even though their monthly charges are higher, customers who opt for tech support are half as likely to churn
- Customers with month-to-month contracts are much more likely to churn than customers with one or two-year contracts
- Customers that pay by electronic check are much more likely to churn than customers using other payment methods

### Data Transformation
Sklearn's `ColumnTransformer` does the heavy lifting here, applying imputation and scaling based on the datatype of the feature. The dataset is split into stratified training and test sets using the `train_test_split` package.

### Model Testing
**Logistic Regression** and **Random Forest** are two classifiers fit out of the box and evaluated using stratified k-fold cross validation. The Logistic Regressor's mean accuracy for 5-fold cross validation was 79.91%, and the mean accuracy for the Random Forest Classifier for 5-fold cross validation was 78.17%. Both results represent incremental improvement from the baseline result. *Because Logistic Regression slightly outperformed the Random Forest Classifier, Logistic Regression was selected as the model of choice for script development.*

### Hyperparameter Tuning
Parameters for the two estimators were optimized by cross-validated grid-searches over parameter grids. Classification reports were generated for each tuned model using the holdout test set with Logistic Regression achieving 80% accuracy and Random Forest with 81% accuracy.

### Run the Scripts
The model can be trained by running the following command.
```
python train.py
```
The model can be evaluated by executing the following command. To test the script a subset of 100 records was pulled from each of the three data files. Evaluation datasets were saved to the `eval_data` directory.
```
python predict.py
```

### Conclusion & Next Steps
The stated purpose of this challenge is to demonstrate how to solve data science problems, not to build a perfect model. By clearly articulating the problem and business objective, meticulously examining the data, preparing the data for downstream modeling, and testing a variety of models the chances of arriving at a useful and reliable solution increase. In reality, more robust feature engineering and selection would be necessary, and additional models should be evaluated. To tune hyperparameters, the `Scikit-Optimize` library can be used to arrive at the best set of hyperparamters using Bayesian Optimization. Finally, deploying the fitted model as an API would make the model more resilient, scalable, and easy to integrate with other systems.
