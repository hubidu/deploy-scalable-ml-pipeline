# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Author: Stefan Huber (Udacity Student)
Created: January 2023
Version: 1.0
Classifier Used: sklearn BaggingClassifier

The census model is generated from about 32 K records of the Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/census+income).

## Intended Use

The model can predict if a person earns a salary above or below 50K based on socio-demographic data such as

- age
- sex
- education
- race...

It can be used in an ecommerce application to predict high value customers (i. e. when the stakes are relatively low).
It should not be used to make any high value financial decisions like creditworthiness decisions or predicting credit defaults.

## Training Data

The model is trained on a preprocessed version of the Census Income Dataset (https://archive.ics.uci.edu/ml/datasets/census+income). A train/test split with a test size of 30% is performed.

## Evaluation Data

The model is evaluated on 30% test data which has been derived from the original data set.

## Metrics

Since the target variable is unbalanced, accuracy is not a good metric for the model and instead precision and recall are used.

- on the test set precision is 73% and recall is 61%

However metrics may vary considerably on slices of the dataset:

- for blue collar occupations like "Handlers-cleaners" or "occupation=Machine-op-inspct" both precision and recall can be significantly lower (e. g. 45% or 29% respectively)
- for races other than "White", recall can be significantly lower (e. g. recall dropping to 35% for "Amer-Indian-Eskimo")

## Ethical Considerations

## Caveats and Recommendations

- data is heavily skewed towards white and married males
- genders are binary
- some features (e. g. native-country) have little variation and should probably be removed from training data
- model seems to overfit on training data set, having 99% accuracy
