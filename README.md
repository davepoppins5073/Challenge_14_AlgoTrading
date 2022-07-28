# Challenge_14_AlgoTrading!
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
1. https://docs.streamlit.io/
2. https://docs.python.org/3/library/hashlib.html
3. https://docs.python.org/3/library/dataclasses.html

## 4 Code 
