# Ramen Ratings Classification

![This is an image](https://i.imgur.com/ZR4mrRB.jpeg)

# Repository Overview
This repository contains my analysis for the Ramen Ratings dataset. The goal is to predict whether a ramen product is rated as five stars or not and identify potential factors that increase or decrease a ramen product's rating using statistical analysis.

This repository was my first side project alongside my time in the Codeup data science program. I'm revisiting it now (a few months later) because I want to apply the new skills and knowledge I've gained since working on it to deliver a better product. The previous work remains in the form of old_analysis.ipynb and the README contents marked with (Old) below.

# Link to Data
https://www.kaggle.com/residentmario/ramen-ratings

# Work Done So Far
1. Wrangle: Prepared data for a two-class problem
2. Wrangle: Clean the data more thoroughly to enable improved statistical analysis
3. Explore: Used statistical analysis against our existing features for Brand, Country, and Packaging
    * Eliminated Ramen Brand and Packaging because they are independent from the target
    * Kept Ramen Country of Origin because it has a dependent relationship with the target
4. Explore: Used feature engineering to split out the Ramen Product Name column.
    * Split product names into keywords
    * Conducted domain research and translation to determine which categories a keyword belongs to
    * Grouped relevant keywords into features
    * Ran statistical tests to determine if the new features have a relationship with the target
    * Eliminated features that did not have a relationship with the target
    * Eliminated low-count or irrelevant keywords from important features
    * Split keywords back out from feature into low-, medium-, and high-proportion groups in terms of 5-Stars
    * Created new features for the new brackets, including an 'unknown' bracket
    * Checked final features against the target in terms of proportionality
5. Model: Used SMOTE+Tomek resampling to handle the class imbalance
6. Model: Built better predictive models
    * Did not need cross-validation to select better models
    * Calculated ROC AUC for model
    * Pushed model work to scripts
7. Final Notebook: Add Overview section containing summary of work and findings
8. Final Notebook: Add Wrangle section
9. Final Notebook: Add keyword categorization section to Explore
10. Final Notebook: Add feature creation and statistical testing section to Explore
11. Final Notebook: Add univariate analysis section to Explore

# Next Steps
1. Data Product - 'Five-Star Ramen Guesser'
    * Build a data product that takes a user's input and gives back the probability of it being five stars
2. Create Final Notebook

# (Old) Readme Contents
### (Old) What I Intend to Improve
1. Build better predictive models by focusing on a two-class problem and applying SMOTE+Tomek resampling for the model training split
2. Clean the data more thoroughly to enable improved statistical analysis
3. Apply better statistical analysis with trustworthy conclusions
4. Push work to scripts
5. Use cross-validation to select better models
6. Use ROC Curve AUC to select better models
7. Deliver specific findings in the README
8. Build a data product that takes a user's input and gives back the probability of it being five stars

### (Old) Repository Overview
My analysis of Kaggle dataset 'Ramen-Ratings'. Produce insights and build a classification model that predicts ratings of ramen based on brand, style, variety, and country of origin.

### (Old) Highlights, Takeaways
- Decided to use a classification approach to predict ramen ratings (star ratings from 0 to 5)
- Conducted a lot of statistical testing against existing and engineered features
- Created multiple engineered features
- Created many models with varying algorithms and hyperparameters
- Modified precision of target for modeling
- Beat baseline prediction accuracy with multiple models and feature combinations

### (Old) Specific Work That Was Done
0. Decide how to approach data
    * Rating is dependent variable
    * First try to predict each 0.25-step rating increment between 0 and 5
    * Then try to predict each 1.0-step rating increment between 0 and 5
    * Engineer columns having too many unique values into new columns
1. Ingest data
    * Download data from kaggle.com
    * Read data using pandas.read_csv()
2. Tidy up data for analysis/modeling
    * Fix dtypes
    * Eliminate unrated observations
    * Remove values with counts too low
    * Remove nulls
    * Check distributions
3. Explore data
    * Statistical testing of package styles
        * Bowls rank statistically higher than cups (95% confidence)
    * Statistical testing of country of origin
        * Japan statistically produces better-rated ramen than USA. (95% confidence)
        * Japan statistically produces better-rated ramen than the rest. (95% confidence)
    * Statistical testing of brands
        * Nissin does not statistically produce better-rated ramen than the rest.
    * Analysis of 5-star reviews by country
    * Analysis of 5-star reviews by brand
    * Engineer several features based on Variety
        * Artificial, Instant, Chicken, Beef, Shrimp, Seafood, Chow Mein, Spicy, Veggie
    * Statistical testing on all nine engineered features
        * Statistically higher rating than rest: None
        * Statistically lower rating than rest: Chicken, Beef, Shrimp, Veggie
        * Not statistically higher or lower: Seafood, Chow Mein, Spicy
4. Create machine learning models on 0.25-step rating increment
    * Calculate mean baseline
    * Build minimum viable product (MVP) model using Brand, Style, Country
        * Encode values for modeling
        * Split data for in-sample and out-of-sample testing with a set random_state
        * Isolate target (rating) from data
        * Create and fit models on in-sample data using varying hyperparameters
            * Random Forest, Decision Tree, K-Nearest Neighbors, Logistic Regression
        * Make predictions on in-sample and out-of-sample data
        * Compare accuracy of models against baseline for both data samples
            * Classification Report shows not all values had over-0% prediction accuracy
            * Performed best: DecisionTreeClassifier with max_depth=16, min_samples_leaf=4 (beats baseline)
    * Build models for data where the Brand has more than 10 reviews
        * Limit dataframe on Brand
        * Follow same steps as building MVP
        * Performed best: LogisticRegression with l2 penalty (beats baseline)
    * Build models using engineered Variety features
        * Create new columns for the features (see: Explore data section)
        * Follow same steps as building MVP
        * Performed best: LogisticRegression without l2 penalty (beats baseline)
    *  Use Recursive Feature Elimination to determine best features
        * RFE identifies one-hot-encoded Country column as better than engineered Variety and also one-hot-encoded Style
    * Create models using only one-hot-encoded Country as a feature
        * Performed best: LogisticRegression without l2 penalty (beats baseline)
5. Create machine learning models on 1.0-step rating increment
    * Decrease target precision from 0.25-step to 1.0-step
    * Calculate mean baseline
    * Build models using previous feature combinations (each approach replicated)
        * Only Country best-performed: KNearestClassifier with 60 neighbors
        * Brand >10 best-performed: DecisionTreeClassifier with max_depth=10, min_samples_leaf=1
        * Engineered features *and* Style, Country best-performed: DecisionTreeClassifier with max_depth=2, min_samples_leaf=1
        * Only-Country and Add-Brand approaches had better prediction accuracy than baseline, but All-Possible performed equally with baseline

### (Old) Recreate my work
1. Download the data from the link above
2. Initialize a Jupyter Notebook server in the same directory as the data
3. Run all cells in ramen_ratings_analysis.ipynb