# Challenge_14_AlgoTrading
![14-4-challenge-image](https://user-images.githubusercontent.com/101449950/181592065-3303ef55-0eff-4bf7-acee-4e2de015e37f.png)

## 1. INTRO 
`AIM`: Enhance the existing trading signals with machine learning algorithms that can adapt to new data.

## 2. To - Do list

Update the `machine_learning_trading_bot.ipynb` file.

1. Establish a Baseline Performance
2. Tune the Baseline Trading Algorithm
3. Evaluate a New Machine Learning Classifier
4. Create an Evaluation Report

-----

#### Part 1: Establish a Baseline Performance

**(Restart the kernel, run the provided cells that correspond with the first three steps, and then proceed to step four).** 

1. Import the OHLCV dataset into a Pandas DataFrame.
2. Generate trading signals using short- and long-window SMA values. 
3. Split the data into training and testing datasets.
4. Use the `SVC` classifier model from SKLearn's support vector machine (SVM) to fit the training data and make predictions. 
5. Review the classification report associated with the `SVC` model predictions. 
6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.
7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. (Save a PNG image of this plot.)
8. Write your conclusions about the performance of the baseline trading algorithm in the `README.md` file that’s associated with your GitHub repository.

#### Part 2: Tune the Baseline Trading Algorithm

Adjust, the model’s input features to find the parameters that result in the best trading outcomes.

1. Tune the training algorithm by adjusting the size of the training dataset.
2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. 
3. Choose the set of parameters that best improved the trading algorithm returns.

#### Part 3: Evaluate a New Machine Learning Classifier

In this section, you’ll use the original parameters that the starter code provided. But, you’ll apply them to the performance of a second machine learning model.

1. Import a new classifier, such as `AdaBoost`, `DecisionTreeClassifier`, or `LogisticRegression`. 
2. Using the original training data as the baseline model, fit another model with the new classifier.
3. Backtest the new model to evaluate its performance.

#### Part 4: Create an Evaluation Report

In the previous sections, you updated your `README.md` file with your conclusions. To accomplish this section, you need to add a summary evaluation report at the end of the `README.md` file. For this report, express your final conclusions and analysis. Support your findings by using the PNG images that you created.

## 3. Prerequisites

### 3.1 Requirements Text File

I created a requirements file that has been included in this repo.
I used the following to create the requirements file

```bash
 pip freeze > requirements_14.txt

```

pip install for Requirements Text Files
```bash
python -m pip install -r requirements_14.txt
```

### 3.2 Libraries and Imports


```python
# Imports
import pandas as pd
import numpy as np
from pathlib import Path
import hvplot.pandas
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import classification_report
```

### Documentation Links (Streamlit, Hashlib, Dataclasses)
1. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
2. https://pandas.pydata.org/docs/reference/api/pandas.tseries.offsets.DateOffset.html
3. https://hvplot.holoviz.org/user_guide/Pandas_API.html

## 4 Code 

### Baseline Strategy

<img width="742" alt="Baseline_Strategy" src="https://user-images.githubusercontent.com/101449950/181600115-8f09adcb-8207-45b4-8ff9-7d0738ba8cd9.png">


#### Classifier: SVM Strategy
<img width="728" alt="SVM Strategy" src="https://user-images.githubusercontent.com/101449950/181599887-919f144e-6677-4d68-a181-8a8ac3df3a4a.png">


### Evaluate a new machine learning classifier

#### Logistic Regressions Strategy
>  LinearRegression fits a linear model with coefficients  to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation. The coefficient estimates for Ordinary Least Squares rely on the independence of the features. When features are correlated and the columns of the design matrix `X` have an approximately linear dependence, the design matrix becomes close to singular and as a result, the least-squares estimate becomes highly sensitive to random errors in the observed target, producing a large variance. This situation of multicollinearity can arise, for example, when data are collected without an experimental design.
<img width="720" alt="Logistic Regression Strategy" src="https://user-images.githubusercontent.com/101449950/181599338-b57766eb-5381-4256-8888-d921e2b9452c.png">

#### AdaBoost Strategy
>  The core principle of AdaBoost is to fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. The predictions from all of them are then combined through a weighted majority vote (or sum) to produce the final prediction. The data modifications at each so-called boosting iteration consist of applying weights , , …,  to each of the training samples. Initially, those weights are all set to , so that the first step simply trains a weak learner on the original data. For each successive iteration, the sample weights are individually modified and the learning algorithm is reapplied to the reweighted data. 


<img width="731" alt="AdaBoost Strategy" src="https://user-images.githubusercontent.com/101449950/181599423-796ce284-addd-4148-a2a3-04fbd8f3b080.png">


#### Decision Tree Strategy
>  Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.

<img width="738" alt="Decision Trees Strategy" src="https://user-images.githubusercontent.com/101449950/181599562-f6e5b066-ed66-479a-b21f-5199713c328c.png">

## **Conclusions / Questions**
We used the following classifiers in order to achieve optimal cumulative return from our trading bot.

1. support vector machine (SVM), 
2. **AdaBoost 
3. **DecisionTreeClassifier, 
4. **LogisticRegression 

Of the three highlighted classifiers (all of which were set-up using the SVM parameters) the AdaBoost Classifier performed the best.
