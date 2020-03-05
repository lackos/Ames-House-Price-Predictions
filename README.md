# House Price Predictor
[https://www.kaggle.com/c/house-prices-advanced-regression-techniques]
Models used to predict house prices based on a feature rich training set of Ames house sales. Based on the Kaggle competition "House Prices: Advanced Regression Techniques". This repo contains several models with differing levels of effectiveness.
## Getting Started
Requirements:
Pandas 1.0.1
xgboost 0.90
numpy 1.18.0
matplotlib 3.1.3
sklearn 0.0
seaborn 0.10.0

### Exploration of the Data
For a detailed description of all the features in the training and test data set see the accompanying document Data/data_description.txt .

Executing 'exploration.py' produces the  overview file 'data_overview.md' of the training and test set of the data in ./Outputs.

From  this file we see:
* There are 1460 training observations.

* From this we see that there are a total of 80 feature columns (and one predictor column in the training data). Of these, 37 are numerical and the remainder are categorical and therefore we will need to use an encoder to use them in our model. 

* Only the 'Neighbourhood' and 'Exterior*' features has a large number of unique categories which could prove difficult to encode if we choose to use them. The rest have roughly 4-5 categories.

* 'LotFrontage', 'Alley', 'PoolQC', 'Fence' and 'MiscFeature' all have a large number of nan values. For ambiguous features like 'LotFrontage' and 'MiscFeature' these amount of values are prohibitive and we will not use those features. Hoever for the other feauture we can infer meaning from the nan values. For instance a nan value in 'PoolQC' would probably mean there is no pool which we would expect to be a big factor in the price of the house. We will fix this feature later. We will also ignore the 'Fence feature'. 

* The testing data has no obvious difference from the training data which is what we desire.

### Models
#### Decision Tree Models
This repository focuses on using decision tree models, specifically random forest and XGBoost model. These can be found in the 'tree_models.py' and 'XGBoost.py' files respectively. The random forest method is the cruder of the two and include only for personal reference, the model with optimized performance is the XGBoost model. 

The XGBoost model employs, if required, a Cross Validated Random Grid search method to optimize the hyper-parameters of the model.

Using little feature engineering other than those mentioned above, This model resulted in a score of 0.1418 on the Kaggle leaderboard. More sophisticated feature engineering such as group imputation may increase this level.

This XGBoost model was condensed into a pipeline in "pipelines.py" for greater ease of use and further optimisation (To come...).

### Model Evaluation
An important part of machine learning is model explanability. Therefore this repo includes codes to evaluate which features of the model have the largest affect. These are included in 'model_evaluation.py'.
