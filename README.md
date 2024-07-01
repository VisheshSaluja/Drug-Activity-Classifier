# Implementation of Decision Tree and Naïve Bayes

This project involves the implementation of Decision Tree and Naïve Bayes algorithms for training a model on the training dataset and testing it on the test dataset using the F1 score as a validation metric.

## Dataset

We have two datasets:

- **Training Dataset**: This dataset consists of 800 records. It is represented as a sparse binary matrix, with patterns in lines and features in columns. The indices of the non-zero features are provided along with a class label (1 or 0) in the first column. The dataset has an imbalanced distribution, with 78 actives (+1) and 722 inactives (0). This dataset is used to train our model.

- **Test Dataset**: This dataset consists of 350 records. It is also represented as a sparse binary matrix, with patterns in lines and features in columns. The indices of non-zero features are provided.
