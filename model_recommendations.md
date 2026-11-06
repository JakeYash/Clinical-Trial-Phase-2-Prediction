# Phase 2 Clinical Trial Prediction Model Recommendations

## Dataset Overview
The objective of this project is to predict whether a Phase 2 trial will be a success or a failure. The prediction target is the `label` column:
- `0` = failure
- `1` = success
I ran the pipeline on a test sample of 50 trials. After filtering we ended up with 46 trials and 95 columns. The sample contained 22 failures and 24 successes.

The dataset contains different types of information, including:
- Trial design and masking
- Planned enrollment and number of arms
- Drug and disease information
- Patient eligibility requirements
- Sponsor information
- Number of countries
- PubMed article counts
The current pipeline picks up the PubMed articles, and the article counts. Sentiment scores have not yet been incorporated.

## Data Preparation
Only information available prior to the trial commencing should be used for prediction. Any columns which would reveal information obtained during or after the trial must be excluded to prevent data leakage.

Examples of columns to exclude include:
- `overall_status`
- `why_stopped`
- `has_results`
- `primary_completion_date`
- `completion_date`

The following columns should also be handled separately:
- `label` is the prediction target.
- nct_id is merely a trial identifier.
- `top_split` should be used to determine whether a row is in the training set or the test set.

## Recommended Models

### 1. Logistic Regression
Logistic regression would give us a very simple, straightforward baseline model. It would take the various features we'd selected and estimate the probability of a trial being a success.

Categorical features would have to be converted into numeric indicator variables and missing values would have to be filled. The model coefficients could be used to determine whether sentiment has a positive or negative relationship with trial success.

### 2. Random Forest
Random Forest would give us an intermediate level of complexity. It's a collection of decision trees and can learn non-linear relationships and interactions between features.

For example, it may find that the effect of enrolment size is different depending on the sponsor type or study design. Feature importance could be used to quantify the impact of the sentiment features.

### 3. XGBoost
XGBoost would be the most sophisticated of the three models. It builds decision trees one after the other, each one attempting to correct the errors of the previous trees.

XGBoost tends to do well with structured data (numeric and categorical features). You could use feature importance or SHAP values to explain how strongly sentiment is impacting the predictions.

## Evaluating Sentiment

Each model should be trained twice:

1. Using trial information without sentiment
2. Using the same trial information with sentiment features

Possible sentiment features include:

- Average article sentiment
- Number of positive articles
- Number of negative articles
- Proportion of positive articles
- Proportion of negative articles

The same training and testing observations should be used for both versions, and their performance can be compared in terms of accuracy, F1 score and ROC-AUC.

If performance improves after sentiment is added, then that would suggest the pre-trial PubMed sentiment does contain some useful information for predicting Phase 2 trial success.

## Conclusion

The three models we're considering are logistic regression, random forest and XGBoost. These represent increasing levels of complexity. Logistic regression is the most interpretable and provides a good baseline. Random forest can capture non-linearity, and random forest is the model we're most likely to use in practice. XGBoost is the most sophisticated approach for structured data.
<img width="468" height="656" alt="image" src="https://github.com/user-attachments/assets/8dc75c10-245e-4259-948a-d3c454eb7fd2" />
