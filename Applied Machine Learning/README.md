# Machine Learning Analysis of Demographic Determinants of Depression

### Author: Logan Correa

## Abstract
Depression is a complex mental health condition influenced by a variety of demographic, psychological, and biological factors. This study investigates the potential of machine learning techniques to predict depression using demographic data from the 2022 Mental Health Client-Level Data (MH-CLD) provided by SAMHSA. The dataset included 228,891 entries and ten key variables, such as age, gender, education, and employment status.

Supervised machine learning models—Logistic Regression, Random Forest, and Gradient Boosting—were trained and evaluated for their predictive capabilities. SHAP analysis highlighted employment status and substance use as the most influential predictors, while variables such as race and ethnicity showed minimal impact. Despite addressing class imbalance using SMOTE and optimizing model hyperparameters via Optuna, all supervised models exhibited relatively low predictive performance, with Logistic Regression achieving the highest AUC of 0.55. Unsupervised models, including Hierarchical Clustering and K-Modes Clustering, also performed poorly in identifying meaningful patterns, with silhouette scores of 0.50 and 0.067, respectively.

While Multiple Correspondence Analysis (MCA) provided dimensionality reduction, it did not significantly enhance model performance. These findings underscore the limitations of demographic data alone in capturing the multifactorial nature of depression. Future research should focus on integrating additional data modalities, such as clinical or neuroimaging data, and exploring advanced machine learning techniques to improve predictive accuracy and uncover deeper insights into depression’s underlying factors.

---

## Introduction
Depression is a pervasive mental health condition, affecting individuals across diverse demographic groups and posing a significant global health challenge. Understanding the demographic determinants of depression is critical for designing effective, targeted interventions to address this multifaceted issue. While traditional statistical methods have been employed to study depression, the advent of machine learning offers an opportunity to explore complex patterns and interactions within large datasets.

This study aims to investigate demographic factors associated with depression by applying machine learning techniques to the 2022 SAMHSA Mental Health Client-Level Data (MH-CLD). This dataset provides a comprehensive collection of demographic and mental health information from clients across various treatment facilities in the United States. The objective is to identify meaningful patterns and evaluate the predictive capabilities of machine learning models in understanding the demographic aspects of depression.

---

## Background
Depression is a complex mental health disorder influenced by various demographic, socioeconomic, and biological factors. Extensive research has sought to identify the determinants of depression, with studies often focusing on individual variables such as age, gender, income, and education. For instance, it has been shown that certain demographic groups, such as women and individuals with lower socioeconomic status, are at higher risk of developing depression. Despite these findings, understanding how these factors interact and contribute to depression at a population level remains a challenge.

Several studies have explored the potential of machine learning in addressing these challenges:
- **Gao et al. (2018):** *Machine Learning in Major Depression: From Classification to Treatment Outcome Prediction.* This study utilized SVMs, Random Forests, and Bayesian networks, demonstrating moderate accuracy in predicting treatment outcomes for MDD.
- **Priya et al. (2020):** *Predicting Anxiety, Depression, and Stress in Modern Life Using Machine Learning Algorithms.* This research applied five machine learning models to classify psychological disorders, with Naïve Bayes achieving the highest accuracy.
- **Patel et al. (2016):** *Studying Depression Using Imaging and Machine Learning Methods.* Focused on neuroimaging, achieving accuracies exceeding 90% in specific datasets using SVMs and Gaussian Processes.

Building on these findings, this study applies both supervised and unsupervised machine learning methods to the 2022 SAMHSA MH-CLD dataset to analyze the demographic determinants of depression.

---

## Methodology and Results

### Dataset Preprocessing
- **Dataset:** 2022 Mental Health Client-Level Data (MH-CLD) from SAMHSA.
- **Selected Variables:** Age, education, ethnicity, race, gender, substance use flag, depression diagnosis flag (DEPRESSFLG), employment status, veteran status, and living arrangement.
- **Preprocessing:** 
  - Removed rows with missing/invalid data.
  - Balanced classes using SMOTE.
  - Cleaned dataset: 228,891 entries.

### Model Training and Evaluation
- **Supervised Models:** Logistic Regression, Random Forest, and Gradient Boosting.
- **Unsupervised Models:** Hierarchical Clustering, K-Modes Clustering, and MCA for dimensionality reduction.
- **Performance:**
  - Logistic Regression achieved the highest AUC (0.55).
  - SHAP analysis identified employment status and substance use as the most influential predictors.
  - Unsupervised models showed poor performance (e.g., K-Modes silhouette score = 0.067).

### Classification Report Summary

| Model                    | Precision (Depressed) | Recall (Depressed) | F1-Score (Depressed) | Overall Accuracy |
|--------------------------|-----------------------|--------------------|----------------------|------------------|
| Logistic Regression      | 0.43                 | 0.55              | 0.48                | 0.53             |
| Random Forest            | 0.43                 | 0.46              | 0.44                | 0.54             |
| Gradient Boosting        | 0.42                 | 0.47              | 0.44                | 0.54             |
| MCA Logistic Regression  | 0.42                 | 0.54              | 0.47                | 0.53             |

---

## Discussion
- Supervised models exhibited low predictive performance, highlighting the insufficiency of demographic data in capturing the complexity of depression.
- Unsupervised models struggled to identify meaningful patterns due to overlapping and imbalanced categorical data.
- Future research should incorporate multidimensional datasets (e.g., clinical, genetic, neuroimaging data) and advanced techniques like deep learning to improve model performance.

---

## Conclusion
This study underscores the limitations of demographic data in modeling depression using machine learning. The findings emphasize the need for richer datasets, integrating diverse data modalities to better understand depression's multifactorial nature. Future work should explore advanced modeling techniques, including deep learning and longitudinal analyses, to uncover deeper insights into depression.

---
