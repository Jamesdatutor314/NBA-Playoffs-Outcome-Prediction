# NBA-Playoffs-Outcome-Prediction
By James Weaver
Project: 

# Introduction
Performed exploratory data analysis on the NBA playoffs dataset utilizing summary statistics, histograms, scatter,
and box plots to identify variable distributions, significance, and correlations. Also developed a KNN model to predict the outcomes of the NBA playoffs.

# Files
For the "presentation1" series, a Jupyter Notebook was created to detail the data analysis, utilizing Python libraries for effective data manipulation and visualization. This was further translated into a cPDF report and a reusable standalone Python script for broader accessibility. Similarly, the "project1" series emphasized feature engineering, with a Jupyter Notebook illustrating the exploration and generation of new dataset features. This work was also documented in a PDF format and accompanied by a modular Python script, ensuring both the depth of analysis and adaptability for various audiences and systems. The "files" folder contains the datasets.

# Analysis
- Wrangled data through encoding, outlier detection, and imputation to ensure data quality.
- Applied borderline SMOTE to improve the class imbalance of target variables, enhancing model performance.
- Grid searched stratified K-Fold CV KNN model parameters to achieve an optimal F1 score of 0.8 while avoiding
model overfit to training data.
