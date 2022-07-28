# Challenge_14_AlgoTrading!
![14-4-challenge-image](https://user-images.githubusercontent.com/101449950/181592065-3303ef55-0eff-4bf7-acee-4e2de015e37f.png)

## 1. INTRO 
The Business ask for this particular task is to build a blockchain-based ledger system, complete with a user-friendly web interface. This ledger should allow partner banks to transfer money between senders and receivers and to verify the integrity of the data in the ledger.


## 2. To - Do list

Update the `pychain.py` file. It already has the functionality to create blocks, perform the proof of work consensus protocol, and validate blocks in the chain. The steps for this Challenge are divided into the following sections:

1. Create a Record Data Class
2. Modify the Existing Block Data Class to Store Record Data
3. Add Relevant User Inputs to the Streamlit Interface
4. Test the PyChain Ledger by Storing Records

## 3. Prerequisites

### 3.1 Requirements Text File

I created a requirements file that has been included in this repo.
I used the following to create the requirements file

```bash
 pip freeze > requirements_CH18.txt

```

pip install for Requirements Text Files
```bash
python -m pip install -r requirements_10.txt
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
