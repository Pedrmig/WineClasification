![cover.png](https://github.com/Pedrmig/WineClasification/blob/main/Image/vino.jpg)
# Gradient Boosting Classifier for Wine Classification </br> with Weights & Biases

This repository contains an experimentation project for the Data Science class at Upgrade Hub. We utilize Weights & Biases to systematically tune and evaluate the hyperparameters of a Gradient Boosting Classifier. The dataset we are working with is the Wine dataset.
## About Upgrade Hub

[Upgrade Hub](https://www.upgrade-hub.com/) is an educational institution dedicated to offering high-quality training programs in technology and data science.

## Results

The experimentation results can be viewed in detail on the Weights & Biases dashboard. This includes various performance metrics such as accuracy, along with the hyperparameters used for each experiment.

🔗 [View Experimentation Results](https://api.wandb.ai/links/pedro-ferraz-10/hbvyk0ji)

## Problem Description

The task at hand is a classic example of a multi-class classification problem. We aim to predict the category of wine based on several physicochemical attributes. The Wine dataset is a common benchmark dataset in the machine learning community.

## Dataset

The Wine dataset is a publicly available dataset that contains 178 samples of wines with 13 different attributes such as Alcohol content, Malic acid, Ash, etc. There are three classes, representing three different types of wines. The dataset is well-suited for classification experiments.

## Experimentation

For this project, we utilize the Gradient Boosting Classifier, a powerful ensemble machine learning algorithm that builds on decision trees. It is particularly known for its effectiveness in classification problems.

To find the best model, we explore various combinations of hyperparameters such as learning rate, maximum depth of the trees, the number of estimators, etc. Through systematic experimentation, we aim to understand the effect of these hyperparameters on the model's performance and find the combination that yields the best results.

We integrate Weights & Biases into our experimentation pipeline, which allows us to log the hyperparameters and the performance metrics for each experiment. Weights & Biases provides us with an interactive dashboard where we can visualize and analyze the results.

## Hyperparameter Tuning and Best Model

During the experimentation process, we performed an extensive search over the hyperparameter space. A total of **384 different combinations** of hyperparameters were tested to find the model that yields the best performance. The hyperparameters that we tuned include:

- Learning rate
- Maximum depth of the trees
- Number of estimators
- Loss function
- Subsample fraction
- Minimum number of samples required to split an internal node
- Minimum number of samples required to be at a leaf node

This extensive search allowed us to explore a wide range of models and identify the combination of hyperparameters that optimizes the performance for this specific dataset.

The model with the best score achieved an accuracy of **0.9815**. This high level of accuracy indicates that the model is highly effective in classifying the wine samples correctly. The hyperparameters of the best model are as follows:

- Learning rate: 0.25
- Loss function: deviance
- Max depth: 5
- min_samples_leaf: 2
- min_samples_split: 4
- n_estimators: 150
- subsample: 1.0

This combination of hyperparameters allowed the Gradient Boosting Classifier to capture the underlying patterns in the data efficiently and make highly accurate predictions.

## Running the Code

To run the code, first, ensure you have all the dependencies installed:

## License

This project is licensed under the MIT License.
