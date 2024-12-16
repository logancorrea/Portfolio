# Machine Learning Analysis of Demographic Determinants of Depression

### Author: Logan Correa

## Abstract
Depression is a complex mental health condition influenced by a variety of demographic, psychological, and biological factors. This study investigates the potential of machine learning techniques to predict depression using demographic data from the 2022 Mental Health Client-Level Data (MH-CLD) provided by SAMHSA. The dataset included 228,891 entries and ten key variables, such as age, gender, education, and employment status. Supervised machine learning models—Logistic Regression, Random Forest, and Gradient Boosting—were trained and evaluated for their predictive capabilities. SHAP analysis highlighted employment status and substance use as the most influential predictors, while variables such as race and ethnicity showed minimal impact.

Despite addressing class imbalance using SMOTE and optimizing model hyperparameters via Optuna, all supervised models exhibited relatively low predictive performance, with Logistic Regression achieving the highest AUC of 0.55. Unsupervised models, including Hierarchical Clustering and K-Modes Clustering, also performed poorly in identifying meaningful patterns, with silhouette scores of 0.50 and 0.067, respectively. While Multiple Correspondence Analysis (MCA) provided dimensionality reduction, it did not significantly enhance model performance. 

These findings underscore the limitations of demographic data alone in capturing the multifactorial nature of depression. Future research should focus on integrating additional data modalities, such as clinical or neuroimaging data, and exploring advanced machine learning techniques to improve predictive accuracy and uncover deeper insights into depression’s underlying factors.

## Introduction
Depression is a pervasive mental health condition, affecting individuals across diverse demographic groups and posing a significant global health challenge. Understanding the demographic determinants of depression is critical for designing effective, targeted interventions to address this multifaceted issue. While traditional statistical methods have been employed to study depression, the advent of machine learning offers an opportunity to explore complex patterns and interactions within large datasets.

This study aims to investigate demographic factors associated with depression by applying machine learning techniques to the 2022 SAMHSA Mental Health Client-Level Data (MH-CLD). This dataset provides a comprehensive collection of demographic and mental health information from clients across various treatment facilities in the United States. The objective is to identify meaningful patterns and evaluate the predictive capabilities of machine learning models in understanding the demographic aspects of depression.

## Background
Depression is a complex mental health disorder influenced by various demographic, socioeconomic, and biological factors. Extensive research has sought to identify the determinants of depression, with studies often focusing on individual variables such as age, gender, income, and education. For instance, it has been shown that certain demographic groups, such as women and individuals with lower socioeconomic status, are at higher risk of developing depression. Despite these findings, understanding how these factors interact and contribute to depression at a population level remains a challenge.

Several studies have explored the potential of machine learning in addressing these challenges:
- **Gao et al. (2018):** *Machine Learning in Major Depression: From Classification to Treatment Outcome Prediction.* This study utilized SVMs, Random Forests, and Bayesian networks, demonstrating moderate accuracy in predicting treatment outcomes for MDD.
- **Priya et al. (2020):** *Predicting Anxiety, Depression, and Stress in Modern Life Using Machine Learning Algorithms.* This research applied five machine learning models to classify psychological disorders, with Naïve Bayes achieving the highest accuracy.
- **Patel et al. (2016):** *Studying Depression Using Imaging and Machine Learning Methods.* Focused on neuroimaging, achieving accuracies exceeding 90% in specific datasets using SVMs and Gaussian Processes.

Building on these findings, this study applies both supervised and unsupervised machine learning methods to the 2022 SAMHSA MH-CLD dataset to analyze the demographic determinants of depression.

## Methodology and Results

### Dataset Preprocessing
The dataset used in this study was the 2022 Mental Health Client-Level Data (MH-CLD) provided by SAMHSA, containing demographic and mental health-related variables collected from treatment facility clients across the United States. Ten variables were selected for analysis, including age, education, ethnicity, race, gender, substance use flag, depression diagnosis flag (DEPRESSFLG), employment status, veteran status, and living arrangement. Preprocessing involved removing rows with missing or invalid data, resulting in a cleaned dataset of 228,891 entries. The target variable, DEPRESSFLG, showed a class imbalance, with 38% flagged as depressed and 62% not depressed, which was addressed using Synthetic Minority Oversampling Technique (SMOTE) to create a balanced dataset. Descriptive statistics indicated moderate variability in most features, while some variables, like ethnicity and race, displayed heavily skewed distributions.

Exploratory data analysis revealed generally weak correlations between features, with most coefficients near zero. However, age and employment status exhibited a mild positive correlation (0.23). Individual variable distributions showed patterns such as an approximately normal distribution for age, whereas ethnicity and race were imbalanced and highly skewed. Advanced statistical analyses of skewness and kurtosis confirmed these imbalances, particularly for ETHNIC and VETERAN, which deviated significantly from normality. These findings provided a comprehensive understanding of the dataset’s structure, which informed the application of machine learning models in subsequent analysis.


### Model Training and Evaluation
Three supervised machine learning models were implemented to predict depression status: Logistic Regression, Random Forest, and Gradient Boosting. Logistic Regression was determined to be the most suitable model due to its computational efficiency and comparable performance to the other models. Hyperparameter optimization was conducted using Optuna, and class imbalance in the target variable was addressed using the Synthetic Minority Oversampling Technique (SMOTE). Models were evaluated on an independent test set using standard classification metrics.

The ROC curves for all supervised models, shown in **Figure 1**, illustrate their ability to distinguish between depressed and non-depressed classes. Logistic Regression achieved the highest Area Under the Curve (AUC = 0.55), slightly outperforming Random Forest (AUC = 0.54) and Gradient Boosting (AUC = 0.53). These results suggest marginal improvements over random performance for all models.

![ROC Curves for Models](/Applied%20Machine%20Learning/Images/ROC.png)

<sub>**Figure 1.** ROC and PR curves compare Logistic Regression, Random Forest, and Gradient Boosting in distinguishing depressed and non-depressed classes. Logistic Regression achieved the highest AUC of 0.55.</sub>

SHAP analysis for Logistic Regression, shown in **Figure 2**, highlighted employment status (EMPLOY) and substance use flag (SUB) as the most influential predictors, followed by age (AGE) and gender (GENDER). These results demonstrate the interpretability of Logistic Regression and the importance of key demographic factors in predicting depression outcomes.

### Logistic Regression SHAP Bar Plot

![SHAP Analysis for Logistic Regression](/Applied%20Machine%20Learning/Images/SHAP.png)

<sub>**Figure 2.**  SHAP summary bar plot highlights employment status (EMPLOY) and substance use flag (SUB) as the most influential predictors of depression, followed by age (AGE) and gender (GENDER).</sub>

Unsupervised models, including Hierarchical Clustering, K-Modes Clustering, and Multiple Correspondence Analysis (MCA), were explored to identify patterns in the dataset. Hierarchical Clustering produced a silhouette score of 0.50, suggesting moderately distinct clusters; however, as shown in **Figure 3**, the silhouette plot revealed uneven cluster sizes and overlapping data points, indicating limited meaningful segmentation of the dataset. K-Modes Clustering performed poorly, yielding a low silhouette score of 0.067, which reflects the lack of strong distinctions among the categorical variables. These results demonstrate the challenges of clustering demographic data with limited variability and clear group separations.

![Silhouette Scores for Hierarchical Clustering](/Applied%20Machine%20Learning/Images/Silhouette.png)

<sub>**Figure 3:** Silhouette plot for Hierarchical Clustering shows an average score of 0.50, with uneven cluster sizes and overlapping data suggesting limited segmentation.</sub>

MCA proved effective in reducing the dataset’s dimensionality while retaining critical information. Logistic Regression applied to the MCA-transformed data achieved an accuracy of 52.86% on the test set after optimization. While this performance was comparable to the original Logistic Regression model, MCA provided additional insights into the dataset’s structure through its dimensionality reduction process.

Classification metrics for the supervised models are summarized in Table 1. Logistic Regression and the optimized MCA Logistic Regression exhibited similar performance, both achieving an accuracy of 53%. The computational efficiency and interpretability of Logistic Regression make it the preferred model for this dataset.

### Classification Report Summary

| Model                    | Precision (Depressed) | Recall (Depressed) | F1-Score (Depressed) | Overall Accuracy |
|--------------------------|-----------------------|--------------------|----------------------|------------------|
| Logistic Regression      | 0.43                 | 0.55              | 0.48                | 0.53             |
| Random Forest            | 0.43                 | 0.46              | 0.44                | 0.54             |
| Gradient Boosting        | 0.42                 | 0.47              | 0.44                | 0.54             |
| MCA Logistic Regression  | 0.42                 | 0.54              | 0.47                | 0.53             |

<sub>**Table 1.** Summary of classification metrics for supervised machine learning models. Precision, recall, F1-score, and overall accuracy are reported for the depressed class. Logistic Regression and Optimized MCA Logistic Regression achieved similar performance, with an overall accuracy of 53%, while Random Forest and Gradient Boosting slightly outperformed in accuracy (54%) but showed no significant advantage in F1-score or recall.</sub>

## Discussion
The results of this study demonstrate the challenges of using demographic data to predict depression with machine learning. The supervised models, including Logistic Regression, Random Forest, and Gradient Boosting, achieved relatively low performance across all evaluation metrics, indicating that demographic variables alone cannot fully capture the complexity of depression. For example, the best performing model Logistic Regression achieved an accuracy of 53%, which is only marginally above random prediction. SHAP analysis identified employment status and substance use as the most influential predictors, while variables such as race and ethnicity had minimal impact, suggesting that these features do not adequately represent the multifactorial nature of depression. These findings align with expectations, as depression is influenced by psychological, environmental, and biological factors not captured by the dataset.

Unsupervised models, such as Hierarchical Clustering and K-Modes Clustering, also faced limitations in identifying meaningful patterns. Hierarchical Clustering achieved a moderate silhouette score of 0.50 but failed to produce distinct clusters, while K-Modes Clustering, with a silhouette score of 0.067, highlighted the difficulty of grouping categorical data with overlapping characteristics. In contrast, MCA proved valuable for dimensionality reduction, offering insights into the dataset’s structure. However, its integration with Logistic Regression did not significantly enhance predictive performance. These results point to the need for richer datasets with more complex features to better address the challenges of modeling depression.

The implementation process also revealed areas for improvement. Addressing class imbalances with SMOTE was a necessary step, but the models still struggled to generalize, likely due to the inherent biases and skewness of the dataset. Hyperparameter optimization via Optuna helped optimize model performance, yet even with fine-tuning, the results remained suboptimal. Future efforts could explore incorporating additional data modalities and alternative feature engineering techniques to enhance performance.

## Conclusion
This study highlights the limitations of relying on demographic data to predict depression using machine learning. The low performance of both supervised and unsupervised models underscores the insufficiency of demographic variables in capturing the multifactorial nature of depression. Furthermore, dataset biases, such as imbalanced distributions of ethnicity and race, and the exclusion of critical factors like clinical and environmental data, further constrained the models’ effectiveness.
Despite these limitations, this study provides valuable insights into the demographic determinants of depression and emphasizes the need for multidimensional data and advanced modeling techniques. Future research should focus on integrating clinical, genetic, or neuroimaging data, exploring deep learning models, and incorporating longitudinal datasets to better understand the dynamic nature of depression. This work establishes a foundation for leveraging machine learning in mental health research and addressing its associated challenges.

## References

1. **Gao, S., Calhoun, V. D., & Sui, J.** (2018). *Machine learning in major depression: From classification to treatment outcome prediction.* CNS Neuroscience & Therapeutics, 24(12), 1037–1052. [https://doi.org/10.1111/cns.13048](https://doi.org/10.1111/cns.13048)

2. **Patel, M. J., Khalaf, A., & Aizenstein, H. J.** (2016). *Studying depression using imaging and machine learning methods.* NeuroImage: Clinical, 10, 115–123. [https://doi.org/10.1016/j.nicl.2015.11.003](https://doi.org/10.1016/j.nicl.2015.11.003)

3. **Priya, A., Garg, S., & Tigga, N. P.** (2020). *Predicting anxiety, depression and stress in modern life using machine learning algorithms.* Procedia Computer Science, 167, 1258–1267. [https://doi.org/10.1016/j.procs.2020.03.442](https://doi.org/10.1016/j.procs.2020.03.442)

4. **Substance Abuse and Mental Health Services Administration (SAMHSA).** (2022). *Mental Health Client-Level Data (MH-CLD) [Data files].* U.S. Department of Health and Human Services. Retrieved from [https://www.samhsa.gov/data/data-we-collect/mh-cld/datafiles](https://www.samhsa.gov/data/data-we-collect/mh-cld/datafiles)


