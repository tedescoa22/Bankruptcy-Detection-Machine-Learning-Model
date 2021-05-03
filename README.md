# Bankruptcy Detection Machine Learning Model



## Problem Statement:


There are over 40,000 corporate bankruptcy in the U.S. per year and can be even worse when the country is experiencing economic downturn or is facing a pandemic. As a result, many companies may find themselves in a position where they are in danger of going bankrupt, without even knowing it. For this project I will look at data from Taiwan (1999-2009) and Poland (2000-2012) to train a machine learning model, that aims to predict bankruptcies. My model seeks to create a model that can not only more accurately predict if a firm will go bankrupt but also have it optimized to minimize the number of firms that are predicted to not go bankrupt but actually did, in other terms the model optimizes for sensitivity.



## The Data:

The data sets for this model both come from kaggle.com, the data set from Taiwan has 95 columns and 6819 different companies while the data set from Poland has 65 columns and over 42,000 different companies.

https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction
https://www.kaggle.com/bhadaneeraj/bankruptcy-detection?select=Bankruptcy+detection_probelm+statement.docx


### Problems with the Data:

None of the columns in both data sets are correlated to bankruptcy. This is an obvious problem as there is nothing to key in on for the models. As a result, based on the finding of two papers, the most important features for determining if a company will go bankrupt or not are the financials of the company, qualitative features such as the quality of management and the economic environment that a firm finds itself in. Since my data only includes financial data, I decided to keep only those financial columns that I deemed the most important in order to better improve the model. 


## The Model:

This project uses several binary classification models such as Logistic Regression, Random Forest, Gradient Boost Classifier, KNN, Extra Trees Classifier and Neural Network,  to create a model that can accurately predict if a company will go bankrupt or not

The best model and final model was a gradient boost classifier that uses SMOTE in order to counter act the severally imbalanced classes in the data.


Taiwan Baseline: 

- 96.77% non bankruptcies
- 3.23% bankruptcies

Poland Baseline:
- 97.83% non bankruptcies
- 2.17% bankruptcies 

Model Accuracy:
- Taiwan: 97.57%
- Poland: 99.91%

Model Sensitivity:
- Taiwan: 99.45%
- POland: 99.9%

## Conclusion:

What I learned from this project was that using a machine learning model, we can accurately predict bankruptcies, while also reducing the risk of providing potential dangerous information and that a lot of what can affect the possibility of a company going bankrupt is dependent on harder to quantify metrics and events outside of a company’s control, such as changes in the market/industry, pandemics and lockdown protocols and difficulties caused by the business cycle such as recessions and depressions.



## Recommendations:

Since this project focuses on only corporate bankruptcies, my recommendations only focus on solutions for businesses. There are several ways to for a company to mitigate the risk of impending bankruptcy. These include paying the minimum amount possible to your creditors, cut any unnecessary or secondary spending, determine your minimum cost level that you can still operate at a profit from, re-negotiate what you can such as contracts with suppliers or company insurance, find ways to create short term cash flow to help fill immediate monetary needs, establish a long term business plan to get out of debt, utilize any tax loopholes/ reforms that you can to reduce spending, and seek the aid of a financial advisor, who can help you achieve many of these goals.


### Sources:


https://www.investopedia.com/articles/active-trading/081315/financial-ratios-spot-companies-headed-bankruptcy.asp
https://www.researchgate.net/publication/235643766_Bankruptcy_prediction_models_How_to_choose_the_most_relevant_variables
https://www.researchgate.net/publication/2340186_Choosing_the_Best_Set_of_Bankruptcy_Predictors
https://smallbusiness.chron.com/avoid-business-bankruptcy-248.html
https://www.fosterswift.com/communications-avoid-bankruptcy-build-strong-business.html
