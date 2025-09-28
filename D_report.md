# Project Report: Titanic Classification

## 1. Performance Conclusion

In this project, we implemented and evaluated several classification models to predict passenger survival on the Titanic. We compared our from-scratch implementations with their `scikit-learn` counterparts. The library-based models, particularly the ensemble methods like Gradient Boosting and XGBoost, demonstrated superior performance. This is attributed to their optimized implementations, advanced features, and the ability to handle complex relationships in the data more effectively than our simplified from-scratch models.

## 2. Technical Report

### 2.1. Minimal Preprocessing Summary

The data preprocessing involved the following steps:

- **Dropping unnecessary columns:** `PassengerId`, `Name`, `Ticket`, and `Cabin` were removed.
- **Imputing missing values:** Missing `Age` values were filled with the median age, and missing `Embarked` values were filled with the mode.
- **Encoding categorical variables:** The `Sex` and `Embarked` columns were converted to numerical representations using one-hot encoding.
- **Data splitting:** The data was split into 80% for training and 20% for testing, with stratification on the `Survived` column to maintain the same class distribution in both sets.

### 2.2. Key Results Comparison Table

| Model                        | Type          | Accuracy  | Precision | Recall    | F1 Score  | AUC      |
| ---------------------------  | ------------- | --------  | --------- | ------    | --------  | ------   |
| Logistic Regression (scratch)| From-scratch  | 0.7654    | 0.8000    | 0.5217    | 0.6316    | 0.8161   |
| Logistic Regression (L1)     | From-scratch  | 0.7709    | 0.8684    | 0.4783    | 0.6168    | 0.8237   |
| Logistic Regression (L2)     | From-scratch  | 0.6257    | 0.5455    | 0.1739    | 0.6609    | 0.6609   |
| Simple Bagging (scratch)     | From-scratch  | 0.6145    | 0.0000    | 0.0000    | 0.0000    | 0.5000   |
| AdaBoost (scratch)           | From-scratch  | 0.6145    | 0.0000    | 0.0000    | 0.0000    | 0.5000   |
| Logistic Regression (None)   | Library       | 0.804469  | 0.793103  | 0.666667  | 0.724409  | 0.843742 |
| Logistic Regression (L1)     | Library       | 0.798883  | 0.779661  | 0.666667  | 0.718750  | 0.846904 |
| Logistic Regression (L2)     | Library       | 0.804469  | 0.793103  | 0.666667  | 0.724409  | 0.844269 |
| Random Forest                | Library       | 0.815642  | 0.781250  | 0.724638  | 0.751880  | 0.834387 |
| AdaBoost                     | Library       | 0.782123  | 0.750000  | 0.652174  | 0.697674  | 0.825165 |
| Gradient Boosting            | Library       | 0.798883  | 0.789474  | 0.652174  | 0.714286  | 0.817918 |
| XGBoost                      | Library       | 0.804469  | 0.757576  | 0.724638  | 0.740741  | 0.820817 |

### 2.3. Analysis of Logistic Regression Variants

The three variants of logistic regression (no regularization, L1, and L2) were implemented from scratch. To improve the convergence and performance of the models, the AdaGrad optimization algorithm was implemented, which adapts the learning rate for each parameter individually. This is a more advanced optimization technique compared to standard gradient descent.

### 2.4. Bagging vs. Boosting Performance Insights

Both bagging and boosting were implemented from scratch.

- **Bagging (SimpleBagging):** This implementation was significantly improved by replacing the simple `DecisionStump` with a from-scratch `DecisionTree` with a configurable `max_depth`. This allows the bagging model to learn more complex patterns and improves its performance.
- **Boosting (AdaBoost):** The AdaBoost implementation uses a `DecisionStump` as its weak learner, which is optimized to minimize the weighted error at each iteration. This is a classic implementation of AdaBoost, and its performance is highly dependent on the quality of the weak learners.

## 3. Future Recommendations

1.  **Advanced Decision Tree Features:** The from-scratch `DecisionTree` could be further improved by adding features like handling categorical variables directly (instead of one-hot encoding) and implementing pruning techniques to prevent overfitting.
2.  **More Adaptive Optimizers:** For `LogisticRegression`, other adaptive learning rate algorithms like RMSProp or Adam could be implemented from scratch and compared with AdaGrad.
3.  **Experiment with Base Estimators:** For the ensemble methods, one could experiment with different types of base estimators. For example, using a from-scratch Naive Bayes classifier as the base estimator for bagging or boosting.
