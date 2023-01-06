The goal of the project is to predict the failure of a water pump that experienced frequent failures during the spring and summer of 2018.

The dataset for this project was collected from the pump and consists of 52 sensors that measure various physical properties of the system.

## Import libraries


```python
# Loading dataset and numpy
import pandas as pd
import numpy as np

# Data visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import datetime

# Missing data imputation
from feature_engine.imputation import RandomSampleImputer

# Pipeline and training
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature selection
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures
from sklearn.feature_selection import VarianceThreshold

# Feature magnitude
from sklearn.preprocessing import MinMaxScaler

# Metrics
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score, 
    confusion_matrix,
    recall_score, 
    f1_score,
    precision_score,
    recall_score
)

# ML models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Libraries for Deep learning model

from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam


# Keras tuner
from kerastuner.tuners import RandomSearch, Hyperband

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings("ignore")
```


```python
data = pd.read_csv('data/sensor_TP.csv')
```


```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>timestamp</th>
      <th>sensor_00</th>
      <th>sensor_01</th>
      <th>sensor_02</th>
      <th>sensor_03</th>
      <th>sensor_04</th>
      <th>sensor_05</th>
      <th>sensor_06</th>
      <th>sensor_07</th>
      <th>sensor_08</th>
      <th>sensor_09</th>
      <th>sensor_10</th>
      <th>sensor_11</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_15</th>
      <th>sensor_16</th>
      <th>sensor_17</th>
      <th>sensor_18</th>
      <th>sensor_19</th>
      <th>sensor_20</th>
      <th>sensor_21</th>
      <th>sensor_22</th>
      <th>sensor_23</th>
      <th>sensor_24</th>
      <th>sensor_25</th>
      <th>sensor_26</th>
      <th>sensor_27</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_32</th>
      <th>sensor_33</th>
      <th>sensor_34</th>
      <th>sensor_35</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_38</th>
      <th>sensor_39</th>
      <th>sensor_40</th>
      <th>sensor_41</th>
      <th>sensor_42</th>
      <th>sensor_43</th>
      <th>sensor_44</th>
      <th>sensor_45</th>
      <th>sensor_46</th>
      <th>sensor_47</th>
      <th>sensor_48</th>
      <th>sensor_49</th>
      <th>sensor_50</th>
      <th>sensor_51</th>
      <th>machine_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-04-01 00:00</td>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>NaN</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
      <td>NORMAL</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2018-04-01 00:01</td>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>NaN</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
      <td>NORMAL</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2018-04-01 00:02</td>
      <td>2.444734</td>
      <td>47.35243</td>
      <td>53.2118</td>
      <td>46.397570</td>
      <td>638.8889</td>
      <td>73.54598</td>
      <td>13.32465</td>
      <td>16.03733</td>
      <td>15.61777</td>
      <td>15.01013</td>
      <td>37.86777</td>
      <td>48.17723</td>
      <td>32.08894</td>
      <td>1.708474</td>
      <td>420.8480</td>
      <td>NaN</td>
      <td>462.7798</td>
      <td>459.6364</td>
      <td>2.500062</td>
      <td>666.2234</td>
      <td>399.9418</td>
      <td>880.4237</td>
      <td>501.3617</td>
      <td>982.7342</td>
      <td>631.1326</td>
      <td>740.8031</td>
      <td>849.8997</td>
      <td>454.2390</td>
      <td>778.5734</td>
      <td>715.6266</td>
      <td>661.5740</td>
      <td>721.8750</td>
      <td>694.7721</td>
      <td>441.2635</td>
      <td>169.9820</td>
      <td>343.1955</td>
      <td>200.9694</td>
      <td>93.90508</td>
      <td>41.40625</td>
      <td>31.25000</td>
      <td>69.53125</td>
      <td>30.46875</td>
      <td>31.770830</td>
      <td>41.66666</td>
      <td>39.351852</td>
      <td>65.39352</td>
      <td>51.21528</td>
      <td>38.194443</td>
      <td>155.9606</td>
      <td>67.12963</td>
      <td>241.3194</td>
      <td>203.7037</td>
      <td>NORMAL</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2018-04-01 00:03</td>
      <td>2.460474</td>
      <td>47.09201</td>
      <td>53.1684</td>
      <td>46.397568</td>
      <td>628.1250</td>
      <td>76.98898</td>
      <td>13.31742</td>
      <td>16.24711</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>38.57977</td>
      <td>48.65607</td>
      <td>31.67221</td>
      <td>1.579427</td>
      <td>420.7494</td>
      <td>NaN</td>
      <td>462.8980</td>
      <td>460.8858</td>
      <td>2.509521</td>
      <td>666.0114</td>
      <td>399.1046</td>
      <td>878.8917</td>
      <td>499.0430</td>
      <td>977.7520</td>
      <td>625.4076</td>
      <td>739.2722</td>
      <td>847.7579</td>
      <td>474.8731</td>
      <td>779.5091</td>
      <td>690.4011</td>
      <td>686.1111</td>
      <td>754.6875</td>
      <td>683.3831</td>
      <td>446.2493</td>
      <td>166.4987</td>
      <td>343.9586</td>
      <td>193.1689</td>
      <td>101.04060</td>
      <td>41.92708</td>
      <td>31.51042</td>
      <td>72.13541</td>
      <td>30.46875</td>
      <td>31.510420</td>
      <td>40.88541</td>
      <td>39.062500</td>
      <td>64.81481</td>
      <td>51.21528</td>
      <td>38.194440</td>
      <td>155.9606</td>
      <td>66.84028</td>
      <td>240.4514</td>
      <td>203.1250</td>
      <td>NORMAL</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2018-04-01 00:04</td>
      <td>2.445718</td>
      <td>47.13541</td>
      <td>53.2118</td>
      <td>46.397568</td>
      <td>636.4583</td>
      <td>76.58897</td>
      <td>13.35359</td>
      <td>16.21094</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>39.48939</td>
      <td>49.06298</td>
      <td>31.95202</td>
      <td>1.683831</td>
      <td>419.8926</td>
      <td>NaN</td>
      <td>461.4906</td>
      <td>468.2206</td>
      <td>2.604785</td>
      <td>663.2111</td>
      <td>400.5426</td>
      <td>882.5874</td>
      <td>498.5383</td>
      <td>979.5755</td>
      <td>627.1830</td>
      <td>737.6033</td>
      <td>846.9182</td>
      <td>408.8159</td>
      <td>785.2307</td>
      <td>704.6937</td>
      <td>631.4814</td>
      <td>766.1458</td>
      <td>702.4431</td>
      <td>433.9081</td>
      <td>164.7498</td>
      <td>339.9630</td>
      <td>193.8770</td>
      <td>101.70380</td>
      <td>42.70833</td>
      <td>31.51042</td>
      <td>76.82291</td>
      <td>30.98958</td>
      <td>31.510420</td>
      <td>41.40625</td>
      <td>38.773150</td>
      <td>65.10416</td>
      <td>51.79398</td>
      <td>38.773150</td>
      <td>158.2755</td>
      <td>66.55093</td>
      <td>242.1875</td>
      <td>201.3889</td>
      <td>NORMAL</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (220320, 55)



## Step 2. Data analysis

### Functions


```python
# Function to show missing data
def missing_data_plot(df):
    data.isnull().mean().plot.bar(figsize = (12, 6))
    plt.ylabel("Missing values")
    plt.xlabel('Features')
    plt.title("Missing Data Analysis")
    plt.show()

# Function to show the distribution of variable
def value_counts_plot(df, variable):
    counts = df[variable].value_counts().rename_axis(variable).reset_index(name = 'count')

    ax = sns.barplot(x = variable, y = 'count', data = counts)
    ax.bar_label(ax.containers[0])
    plt.title('Value counts of {}'.format(variable))
    plt.ylabel('Number of Occurrences', fontsize = 12)
    plt.xlabel(variable, fontsize = 12)
    plt.show()

# Function to show the broken status compare to date
def plot_label_by_date(df, variable):
    df['datetime'] = pd.to_datetime(df['timestamp'])
    broken = df[df['machine_status']=='BROKEN']
    recovering = df[df['machine_status']=='RECOVERING']
    
    plt.figure(figsize=(18,3))
    plt.plot(broken['datetime'], broken[variable], linestyle = 'none', marker = 'X', color = 'red', markersize = 12, label = 'BROKEN')
    plt.plot(recovering['datetime'], recovering[variable], linestyle = 'none', marker='X', color='orange', markersize=6, label = 'RECOVERING')
    plt.plot(df['datetime'], df[variable], color = 'blue', label = 'WORKING')
    plt.title(variable)
    plt.legend()
    plt.show()

# Function to calculate recovering time
def get_recovering_times(df):
    recovering_times_hours = []

    BROKEN_rows = list(df[df['machine_status'] == "BROKEN"]['Unnamed: 0'].values)

    for i in BROKEN_rows:
        go_further = True
        j = i
        while go_further:
            j += 1
            machine_status_in_row_j = df.iloc[j]["machine_status"]        
            if machine_status_in_row_j != "RECOVERING":
                go_further = False
            
        recovering_hours = (j-i) / 60

        recovering_times_hours.append(recovering_hours)
    
    return recovering_times_hours, BROKEN_rows
   
def diagnostic_plots(df, variable):
    fig = plt.figure()
    plt.figure(figsize = (18, 5))
   
    plt.subplot(1, 3, 1)
    fig = df[variable].hist(bins = 30)
    fig.set_ylabel ('Count')
    fig.set_xlabel(variable)

    plt.subplot(1, 3, 2)
    stats.probplot(x = df[variable], dist = 'norm', plot = plt)
    
    plt.subplot(1, 3, 3)
    fig = df.boxplot(column = variable)
    fig.set_title('Box-plot')
    fig.set_ylabel(variable)

    plt.show()
```


```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 220320 entries, 0 to 220319
    Data columns (total 55 columns):
     #   Column          Non-Null Count   Dtype  
    ---  ------          --------------   -----  
     0   Unnamed: 0      220320 non-null  int64  
     1   timestamp       220320 non-null  object 
     2   sensor_00       210112 non-null  float64
     3   sensor_01       219951 non-null  float64
     4   sensor_02       220301 non-null  float64
     5   sensor_03       220301 non-null  float64
     6   sensor_04       220301 non-null  float64
     7   sensor_05       220301 non-null  float64
     8   sensor_06       215522 non-null  float64
     9   sensor_07       214869 non-null  float64
     10  sensor_08       215213 non-null  float64
     11  sensor_09       215725 non-null  float64
     12  sensor_10       220301 non-null  float64
     13  sensor_11       220301 non-null  float64
     14  sensor_12       220301 non-null  float64
     15  sensor_13       220301 non-null  float64
     16  sensor_14       220299 non-null  float64
     17  sensor_15       0 non-null       float64
     18  sensor_16       220289 non-null  float64
     19  sensor_17       220274 non-null  float64
     20  sensor_18       220274 non-null  float64
     21  sensor_19       220304 non-null  float64
     22  sensor_20       220304 non-null  float64
     23  sensor_21       220304 non-null  float64
     24  sensor_22       220279 non-null  float64
     25  sensor_23       220304 non-null  float64
     26  sensor_24       220304 non-null  float64
     27  sensor_25       220284 non-null  float64
     28  sensor_26       220300 non-null  float64
     29  sensor_27       220304 non-null  float64
     30  sensor_28       220304 non-null  float64
     31  sensor_29       220248 non-null  float64
     32  sensor_30       220059 non-null  float64
     33  sensor_31       220304 non-null  float64
     34  sensor_32       220252 non-null  float64
     35  sensor_33       220304 non-null  float64
     36  sensor_34       220304 non-null  float64
     37  sensor_35       220304 non-null  float64
     38  sensor_36       220304 non-null  float64
     39  sensor_37       220304 non-null  float64
     40  sensor_38       220293 non-null  float64
     41  sensor_39       220293 non-null  float64
     42  sensor_40       220293 non-null  float64
     43  sensor_41       220293 non-null  float64
     44  sensor_42       220293 non-null  float64
     45  sensor_43       220293 non-null  float64
     46  sensor_44       220293 non-null  float64
     47  sensor_45       220293 non-null  float64
     48  sensor_46       220293 non-null  float64
     49  sensor_47       220293 non-null  float64
     50  sensor_48       220293 non-null  float64
     51  sensor_49       220293 non-null  float64
     52  sensor_50       143303 non-null  float64
     53  sensor_51       204937 non-null  float64
     54  machine_status  220320 non-null  object 
    dtypes: float64(52), int64(1), object(2)
    memory usage: 92.5+ MB
    


```python
missing_data_plot(data)
```


    
![png](README_files/README_10_0.png)
    



```python
data.isnull().mean()
```




    Unnamed: 0        0.000000
    timestamp         0.000000
    sensor_00         0.046333
    sensor_01         0.001675
    sensor_02         0.000086
    sensor_03         0.000086
    sensor_04         0.000086
    sensor_05         0.000086
    sensor_06         0.021777
    sensor_07         0.024741
    sensor_08         0.023180
    sensor_09         0.020856
    sensor_10         0.000086
    sensor_11         0.000086
    sensor_12         0.000086
    sensor_13         0.000086
    sensor_14         0.000095
    sensor_15         1.000000
    sensor_16         0.000141
    sensor_17         0.000209
    sensor_18         0.000209
    sensor_19         0.000073
    sensor_20         0.000073
    sensor_21         0.000073
    sensor_22         0.000186
    sensor_23         0.000073
    sensor_24         0.000073
    sensor_25         0.000163
    sensor_26         0.000091
    sensor_27         0.000073
    sensor_28         0.000073
    sensor_29         0.000327
    sensor_30         0.001185
    sensor_31         0.000073
    sensor_32         0.000309
    sensor_33         0.000073
    sensor_34         0.000073
    sensor_35         0.000073
    sensor_36         0.000073
    sensor_37         0.000073
    sensor_38         0.000123
    sensor_39         0.000123
    sensor_40         0.000123
    sensor_41         0.000123
    sensor_42         0.000123
    sensor_43         0.000123
    sensor_44         0.000123
    sensor_45         0.000123
    sensor_46         0.000123
    sensor_47         0.000123
    sensor_48         0.000123
    sensor_49         0.000123
    sensor_50         0.349569
    sensor_51         0.069821
    machine_status    0.000000
    dtype: float64




```python
#correlation matrix

fig, ax = plt.subplots(figsize = (30, 30)) 
sns.heatmap(data.corr(), cmap = "YlGnBu", annot=True);
```


    
![png](README_files/README_12_0.png)
    



```python
data.describe().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Unnamed: 0</th>
      <td>220320.0</td>
      <td>110159.500000</td>
      <td>63601.049991</td>
      <td>0.000000</td>
      <td>55079.750000</td>
      <td>110159.500000</td>
      <td>165239.250000</td>
      <td>220319.000000</td>
    </tr>
    <tr>
      <th>sensor_00</th>
      <td>210112.0</td>
      <td>2.372221</td>
      <td>0.412227</td>
      <td>0.000000</td>
      <td>2.438831</td>
      <td>2.456539</td>
      <td>2.499826</td>
      <td>2.549016</td>
    </tr>
    <tr>
      <th>sensor_01</th>
      <td>219951.0</td>
      <td>47.591611</td>
      <td>3.296666</td>
      <td>0.000000</td>
      <td>46.310760</td>
      <td>48.133678</td>
      <td>49.479160</td>
      <td>56.727430</td>
    </tr>
    <tr>
      <th>sensor_02</th>
      <td>220301.0</td>
      <td>50.867392</td>
      <td>3.666820</td>
      <td>33.159720</td>
      <td>50.390620</td>
      <td>51.649300</td>
      <td>52.777770</td>
      <td>56.032990</td>
    </tr>
    <tr>
      <th>sensor_03</th>
      <td>220301.0</td>
      <td>43.752481</td>
      <td>2.418887</td>
      <td>31.640620</td>
      <td>42.838539</td>
      <td>44.227428</td>
      <td>45.312500</td>
      <td>48.220490</td>
    </tr>
    <tr>
      <th>sensor_04</th>
      <td>220301.0</td>
      <td>590.673936</td>
      <td>144.023912</td>
      <td>2.798032</td>
      <td>626.620400</td>
      <td>632.638916</td>
      <td>637.615723</td>
      <td>800.000000</td>
    </tr>
    <tr>
      <th>sensor_05</th>
      <td>220301.0</td>
      <td>73.396414</td>
      <td>17.298247</td>
      <td>0.000000</td>
      <td>69.976260</td>
      <td>75.576790</td>
      <td>80.912150</td>
      <td>99.999880</td>
    </tr>
    <tr>
      <th>sensor_06</th>
      <td>215522.0</td>
      <td>13.501537</td>
      <td>2.163736</td>
      <td>0.014468</td>
      <td>13.346350</td>
      <td>13.642940</td>
      <td>14.539930</td>
      <td>22.251160</td>
    </tr>
    <tr>
      <th>sensor_07</th>
      <td>214869.0</td>
      <td>15.843152</td>
      <td>2.201155</td>
      <td>0.000000</td>
      <td>15.907120</td>
      <td>16.167530</td>
      <td>16.427950</td>
      <td>23.596640</td>
    </tr>
    <tr>
      <th>sensor_08</th>
      <td>215213.0</td>
      <td>15.200721</td>
      <td>2.037390</td>
      <td>0.028935</td>
      <td>15.183740</td>
      <td>15.494790</td>
      <td>15.697340</td>
      <td>24.348960</td>
    </tr>
    <tr>
      <th>sensor_09</th>
      <td>215725.0</td>
      <td>14.799210</td>
      <td>2.091963</td>
      <td>0.000000</td>
      <td>15.053530</td>
      <td>15.082470</td>
      <td>15.118630</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>sensor_10</th>
      <td>220301.0</td>
      <td>41.470339</td>
      <td>12.093519</td>
      <td>0.000000</td>
      <td>40.705260</td>
      <td>44.291340</td>
      <td>47.463760</td>
      <td>76.106860</td>
    </tr>
    <tr>
      <th>sensor_11</th>
      <td>220301.0</td>
      <td>41.918319</td>
      <td>13.056425</td>
      <td>0.000000</td>
      <td>38.856420</td>
      <td>45.363140</td>
      <td>49.656540</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>sensor_12</th>
      <td>220301.0</td>
      <td>29.136975</td>
      <td>10.113935</td>
      <td>0.000000</td>
      <td>28.686810</td>
      <td>32.515830</td>
      <td>34.939730</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>sensor_13</th>
      <td>220301.0</td>
      <td>7.078858</td>
      <td>6.901755</td>
      <td>0.000000</td>
      <td>1.538516</td>
      <td>2.929809</td>
      <td>12.859520</td>
      <td>31.187550</td>
    </tr>
    <tr>
      <th>sensor_14</th>
      <td>220299.0</td>
      <td>376.860041</td>
      <td>113.206382</td>
      <td>32.409550</td>
      <td>418.103250</td>
      <td>420.106200</td>
      <td>420.997100</td>
      <td>500.000000</td>
    </tr>
    <tr>
      <th>sensor_15</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>sensor_16</th>
      <td>220289.0</td>
      <td>416.472892</td>
      <td>126.072642</td>
      <td>0.000000</td>
      <td>459.453400</td>
      <td>462.856100</td>
      <td>464.302700</td>
      <td>739.741500</td>
    </tr>
    <tr>
      <th>sensor_17</th>
      <td>220274.0</td>
      <td>421.127517</td>
      <td>129.156175</td>
      <td>0.000000</td>
      <td>454.138825</td>
      <td>462.020250</td>
      <td>466.857075</td>
      <td>599.999939</td>
    </tr>
    <tr>
      <th>sensor_18</th>
      <td>220274.0</td>
      <td>2.303785</td>
      <td>0.765883</td>
      <td>0.000000</td>
      <td>2.447542</td>
      <td>2.533704</td>
      <td>2.587682</td>
      <td>4.873250</td>
    </tr>
    <tr>
      <th>sensor_19</th>
      <td>220304.0</td>
      <td>590.829775</td>
      <td>199.345820</td>
      <td>0.000000</td>
      <td>662.768975</td>
      <td>665.672400</td>
      <td>667.146700</td>
      <td>878.917900</td>
    </tr>
    <tr>
      <th>sensor_20</th>
      <td>220304.0</td>
      <td>360.805165</td>
      <td>101.974118</td>
      <td>0.000000</td>
      <td>398.021500</td>
      <td>399.367000</td>
      <td>400.088400</td>
      <td>448.907900</td>
    </tr>
    <tr>
      <th>sensor_21</th>
      <td>220304.0</td>
      <td>796.225942</td>
      <td>226.679317</td>
      <td>95.527660</td>
      <td>875.464400</td>
      <td>879.697600</td>
      <td>882.129900</td>
      <td>1107.526000</td>
    </tr>
    <tr>
      <th>sensor_22</th>
      <td>220279.0</td>
      <td>459.792815</td>
      <td>154.528337</td>
      <td>0.000000</td>
      <td>478.962600</td>
      <td>531.855900</td>
      <td>534.254850</td>
      <td>594.061100</td>
    </tr>
    <tr>
      <th>sensor_23</th>
      <td>220304.0</td>
      <td>922.609264</td>
      <td>291.835280</td>
      <td>0.000000</td>
      <td>950.922400</td>
      <td>981.925000</td>
      <td>1090.808000</td>
      <td>1227.564000</td>
    </tr>
    <tr>
      <th>sensor_24</th>
      <td>220304.0</td>
      <td>556.235397</td>
      <td>182.297979</td>
      <td>0.000000</td>
      <td>601.151050</td>
      <td>625.873500</td>
      <td>628.607725</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>sensor_25</th>
      <td>220284.0</td>
      <td>649.144799</td>
      <td>220.865166</td>
      <td>0.000000</td>
      <td>693.957800</td>
      <td>740.203500</td>
      <td>750.357125</td>
      <td>839.575000</td>
    </tr>
    <tr>
      <th>sensor_26</th>
      <td>220300.0</td>
      <td>786.411781</td>
      <td>246.663608</td>
      <td>43.154790</td>
      <td>790.489575</td>
      <td>861.869600</td>
      <td>919.104775</td>
      <td>1214.420000</td>
    </tr>
    <tr>
      <th>sensor_27</th>
      <td>220304.0</td>
      <td>501.506589</td>
      <td>169.823173</td>
      <td>0.000000</td>
      <td>448.297950</td>
      <td>494.468450</td>
      <td>536.274550</td>
      <td>2000.000000</td>
    </tr>
    <tr>
      <th>sensor_28</th>
      <td>220304.0</td>
      <td>851.690339</td>
      <td>313.074032</td>
      <td>4.319347</td>
      <td>782.682625</td>
      <td>967.279850</td>
      <td>1043.976500</td>
      <td>1841.146000</td>
    </tr>
    <tr>
      <th>sensor_29</th>
      <td>220248.0</td>
      <td>576.195305</td>
      <td>225.764091</td>
      <td>0.636574</td>
      <td>518.947225</td>
      <td>564.872500</td>
      <td>744.021475</td>
      <td>1466.281000</td>
    </tr>
    <tr>
      <th>sensor_30</th>
      <td>220059.0</td>
      <td>614.596442</td>
      <td>195.726872</td>
      <td>0.000000</td>
      <td>627.777800</td>
      <td>668.981400</td>
      <td>697.222200</td>
      <td>1600.000000</td>
    </tr>
    <tr>
      <th>sensor_31</th>
      <td>220304.0</td>
      <td>863.323100</td>
      <td>283.544760</td>
      <td>23.958330</td>
      <td>839.062400</td>
      <td>917.708300</td>
      <td>981.249900</td>
      <td>1800.000000</td>
    </tr>
    <tr>
      <th>sensor_32</th>
      <td>220252.0</td>
      <td>804.283915</td>
      <td>260.602361</td>
      <td>0.240716</td>
      <td>760.607475</td>
      <td>878.850750</td>
      <td>943.877625</td>
      <td>1839.211000</td>
    </tr>
    <tr>
      <th>sensor_33</th>
      <td>220304.0</td>
      <td>486.405980</td>
      <td>150.751836</td>
      <td>6.460602</td>
      <td>489.761075</td>
      <td>512.271750</td>
      <td>555.163225</td>
      <td>1578.600000</td>
    </tr>
    <tr>
      <th>sensor_34</th>
      <td>220304.0</td>
      <td>234.971776</td>
      <td>88.376065</td>
      <td>54.882370</td>
      <td>172.486300</td>
      <td>226.356050</td>
      <td>316.844950</td>
      <td>425.549800</td>
    </tr>
    <tr>
      <th>sensor_35</th>
      <td>220304.0</td>
      <td>427.129817</td>
      <td>141.772519</td>
      <td>0.000000</td>
      <td>353.176625</td>
      <td>473.349350</td>
      <td>528.891025</td>
      <td>694.479126</td>
    </tr>
    <tr>
      <th>sensor_36</th>
      <td>220304.0</td>
      <td>593.033876</td>
      <td>289.385511</td>
      <td>2.260970</td>
      <td>288.547575</td>
      <td>709.668050</td>
      <td>837.333025</td>
      <td>984.060700</td>
    </tr>
    <tr>
      <th>sensor_37</th>
      <td>220304.0</td>
      <td>60.787360</td>
      <td>37.604883</td>
      <td>0.000000</td>
      <td>28.799220</td>
      <td>64.295485</td>
      <td>90.821928</td>
      <td>174.901200</td>
    </tr>
    <tr>
      <th>sensor_38</th>
      <td>220293.0</td>
      <td>49.655946</td>
      <td>10.540397</td>
      <td>24.479166</td>
      <td>45.572910</td>
      <td>49.479160</td>
      <td>53.645830</td>
      <td>417.708300</td>
    </tr>
    <tr>
      <th>sensor_39</th>
      <td>220293.0</td>
      <td>36.610444</td>
      <td>15.613723</td>
      <td>19.270830</td>
      <td>32.552080</td>
      <td>35.416660</td>
      <td>39.062500</td>
      <td>547.916600</td>
    </tr>
    <tr>
      <th>sensor_40</th>
      <td>220293.0</td>
      <td>68.844530</td>
      <td>21.371139</td>
      <td>23.437500</td>
      <td>57.812500</td>
      <td>66.406250</td>
      <td>77.864580</td>
      <td>512.760400</td>
    </tr>
    <tr>
      <th>sensor_41</th>
      <td>220293.0</td>
      <td>35.365126</td>
      <td>7.898665</td>
      <td>20.833330</td>
      <td>32.552080</td>
      <td>34.895832</td>
      <td>37.760410</td>
      <td>420.312500</td>
    </tr>
    <tr>
      <th>sensor_42</th>
      <td>220293.0</td>
      <td>35.453455</td>
      <td>10.259521</td>
      <td>22.135416</td>
      <td>32.812500</td>
      <td>35.156250</td>
      <td>36.979164</td>
      <td>374.218800</td>
    </tr>
    <tr>
      <th>sensor_43</th>
      <td>220293.0</td>
      <td>43.879591</td>
      <td>11.044404</td>
      <td>24.479166</td>
      <td>39.583330</td>
      <td>42.968750</td>
      <td>46.614580</td>
      <td>408.593700</td>
    </tr>
    <tr>
      <th>sensor_44</th>
      <td>220293.0</td>
      <td>42.656877</td>
      <td>11.576355</td>
      <td>25.752316</td>
      <td>36.747684</td>
      <td>40.509260</td>
      <td>45.138890</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>sensor_45</th>
      <td>220293.0</td>
      <td>43.094984</td>
      <td>12.837520</td>
      <td>26.331018</td>
      <td>36.747684</td>
      <td>40.219910</td>
      <td>44.849540</td>
      <td>320.312500</td>
    </tr>
    <tr>
      <th>sensor_46</th>
      <td>220293.0</td>
      <td>48.018585</td>
      <td>15.641284</td>
      <td>26.331018</td>
      <td>40.509258</td>
      <td>44.849540</td>
      <td>51.215280</td>
      <td>370.370400</td>
    </tr>
    <tr>
      <th>sensor_47</th>
      <td>220293.0</td>
      <td>44.340903</td>
      <td>10.442437</td>
      <td>27.199070</td>
      <td>39.062500</td>
      <td>42.534720</td>
      <td>46.585650</td>
      <td>303.530100</td>
    </tr>
    <tr>
      <th>sensor_48</th>
      <td>220293.0</td>
      <td>150.889044</td>
      <td>82.244957</td>
      <td>26.331018</td>
      <td>83.912030</td>
      <td>138.020800</td>
      <td>208.333300</td>
      <td>561.632000</td>
    </tr>
    <tr>
      <th>sensor_49</th>
      <td>220293.0</td>
      <td>57.119968</td>
      <td>19.143598</td>
      <td>26.620370</td>
      <td>47.743060</td>
      <td>52.662040</td>
      <td>60.763890</td>
      <td>464.409700</td>
    </tr>
    <tr>
      <th>sensor_50</th>
      <td>143303.0</td>
      <td>183.049260</td>
      <td>65.258650</td>
      <td>27.488426</td>
      <td>167.534700</td>
      <td>193.865700</td>
      <td>219.907400</td>
      <td>1000.000000</td>
    </tr>
    <tr>
      <th>sensor_51</th>
      <td>204937.0</td>
      <td>202.699667</td>
      <td>109.588607</td>
      <td>27.777779</td>
      <td>179.108800</td>
      <td>197.338000</td>
      <td>216.724500</td>
      <td>1000.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# distribution
fig_ = data.hist(figsize=(25, 24), bins=60, color="cyan",
                         edgecolor="gray", xlabelsize=10, ylabelsize=10)
```


    
![png](README_files/README_14_0.png)
    



```python
plot_label_by_date(data, 'sensor_02')
```


    
![png](README_files/README_15_0.png)
    



```python
# distribution of target variable

value_counts_plot(data, "machine_status")
```


    
![png](README_files/README_16_0.png)
    



```python
recovering_times_hours, broken_rows = get_recovering_times(data)

xpos = np.arange( len(recovering_times_hours) )

fig, ax = plt.subplots()
ax.bar(xpos,recovering_times_hours)
ax.set_xticks(xpos)
ax.set_xticklabels(xpos)
ax.set_title("Duration of RECOVERING for each pump failure")
plt.show()
```


    
![png](README_files/README_17_0.png)
    



```python
# show 
for var in data.columns:
    if var not in ['Unnamed: 0', 'timestamp', 'datetime', 'machine_status', 'sensor_15']:
        diagnostic_plots(data, var)
```


    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_1.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_3.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_5.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_7.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_9.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_11.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_13.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_15.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_17.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_19.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_21.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_23.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_25.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_27.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_29.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_31.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_33.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_35.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_37.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_39.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_41.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_43.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_45.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_47.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_49.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_51.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_53.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_55.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_57.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_59.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_61.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_63.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_65.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_67.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_69.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_71.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_73.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_75.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_77.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_79.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_81.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_83.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_85.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_87.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_89.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_91.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_93.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_95.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_97.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_99.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_18_101.png)
    



```python
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>timestamp</th>
      <th>sensor_00</th>
      <th>sensor_01</th>
      <th>sensor_02</th>
      <th>sensor_03</th>
      <th>sensor_04</th>
      <th>sensor_05</th>
      <th>sensor_06</th>
      <th>sensor_07</th>
      <th>sensor_08</th>
      <th>sensor_09</th>
      <th>sensor_10</th>
      <th>sensor_11</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_15</th>
      <th>sensor_16</th>
      <th>sensor_17</th>
      <th>sensor_18</th>
      <th>sensor_19</th>
      <th>sensor_20</th>
      <th>sensor_21</th>
      <th>sensor_22</th>
      <th>sensor_23</th>
      <th>sensor_24</th>
      <th>sensor_25</th>
      <th>sensor_26</th>
      <th>sensor_27</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_32</th>
      <th>sensor_33</th>
      <th>sensor_34</th>
      <th>sensor_35</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_38</th>
      <th>sensor_39</th>
      <th>sensor_40</th>
      <th>sensor_41</th>
      <th>sensor_42</th>
      <th>sensor_43</th>
      <th>sensor_44</th>
      <th>sensor_45</th>
      <th>sensor_46</th>
      <th>sensor_47</th>
      <th>sensor_48</th>
      <th>sensor_49</th>
      <th>sensor_50</th>
      <th>sensor_51</th>
      <th>machine_status</th>
      <th>datetime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2018-04-01 00:00</td>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>NaN</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
      <td>NORMAL</td>
      <td>2018-04-01 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2018-04-01 00:01</td>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>NaN</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
      <td>NORMAL</td>
      <td>2018-04-01 00:01:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2018-04-01 00:02</td>
      <td>2.444734</td>
      <td>47.35243</td>
      <td>53.2118</td>
      <td>46.397570</td>
      <td>638.8889</td>
      <td>73.54598</td>
      <td>13.32465</td>
      <td>16.03733</td>
      <td>15.61777</td>
      <td>15.01013</td>
      <td>37.86777</td>
      <td>48.17723</td>
      <td>32.08894</td>
      <td>1.708474</td>
      <td>420.8480</td>
      <td>NaN</td>
      <td>462.7798</td>
      <td>459.6364</td>
      <td>2.500062</td>
      <td>666.2234</td>
      <td>399.9418</td>
      <td>880.4237</td>
      <td>501.3617</td>
      <td>982.7342</td>
      <td>631.1326</td>
      <td>740.8031</td>
      <td>849.8997</td>
      <td>454.2390</td>
      <td>778.5734</td>
      <td>715.6266</td>
      <td>661.5740</td>
      <td>721.8750</td>
      <td>694.7721</td>
      <td>441.2635</td>
      <td>169.9820</td>
      <td>343.1955</td>
      <td>200.9694</td>
      <td>93.90508</td>
      <td>41.40625</td>
      <td>31.25000</td>
      <td>69.53125</td>
      <td>30.46875</td>
      <td>31.770830</td>
      <td>41.66666</td>
      <td>39.351852</td>
      <td>65.39352</td>
      <td>51.21528</td>
      <td>38.194443</td>
      <td>155.9606</td>
      <td>67.12963</td>
      <td>241.3194</td>
      <td>203.7037</td>
      <td>NORMAL</td>
      <td>2018-04-01 00:02:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2018-04-01 00:03</td>
      <td>2.460474</td>
      <td>47.09201</td>
      <td>53.1684</td>
      <td>46.397568</td>
      <td>628.1250</td>
      <td>76.98898</td>
      <td>13.31742</td>
      <td>16.24711</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>38.57977</td>
      <td>48.65607</td>
      <td>31.67221</td>
      <td>1.579427</td>
      <td>420.7494</td>
      <td>NaN</td>
      <td>462.8980</td>
      <td>460.8858</td>
      <td>2.509521</td>
      <td>666.0114</td>
      <td>399.1046</td>
      <td>878.8917</td>
      <td>499.0430</td>
      <td>977.7520</td>
      <td>625.4076</td>
      <td>739.2722</td>
      <td>847.7579</td>
      <td>474.8731</td>
      <td>779.5091</td>
      <td>690.4011</td>
      <td>686.1111</td>
      <td>754.6875</td>
      <td>683.3831</td>
      <td>446.2493</td>
      <td>166.4987</td>
      <td>343.9586</td>
      <td>193.1689</td>
      <td>101.04060</td>
      <td>41.92708</td>
      <td>31.51042</td>
      <td>72.13541</td>
      <td>30.46875</td>
      <td>31.510420</td>
      <td>40.88541</td>
      <td>39.062500</td>
      <td>64.81481</td>
      <td>51.21528</td>
      <td>38.194440</td>
      <td>155.9606</td>
      <td>66.84028</td>
      <td>240.4514</td>
      <td>203.1250</td>
      <td>NORMAL</td>
      <td>2018-04-01 00:03:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2018-04-01 00:04</td>
      <td>2.445718</td>
      <td>47.13541</td>
      <td>53.2118</td>
      <td>46.397568</td>
      <td>636.4583</td>
      <td>76.58897</td>
      <td>13.35359</td>
      <td>16.21094</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>39.48939</td>
      <td>49.06298</td>
      <td>31.95202</td>
      <td>1.683831</td>
      <td>419.8926</td>
      <td>NaN</td>
      <td>461.4906</td>
      <td>468.2206</td>
      <td>2.604785</td>
      <td>663.2111</td>
      <td>400.5426</td>
      <td>882.5874</td>
      <td>498.5383</td>
      <td>979.5755</td>
      <td>627.1830</td>
      <td>737.6033</td>
      <td>846.9182</td>
      <td>408.8159</td>
      <td>785.2307</td>
      <td>704.6937</td>
      <td>631.4814</td>
      <td>766.1458</td>
      <td>702.4431</td>
      <td>433.9081</td>
      <td>164.7498</td>
      <td>339.9630</td>
      <td>193.8770</td>
      <td>101.70380</td>
      <td>42.70833</td>
      <td>31.51042</td>
      <td>76.82291</td>
      <td>30.98958</td>
      <td>31.510420</td>
      <td>41.40625</td>
      <td>38.773150</td>
      <td>65.10416</td>
      <td>51.79398</td>
      <td>38.773150</td>
      <td>158.2755</td>
      <td>66.55093</td>
      <td>242.1875</td>
      <td>201.3889</td>
      <td>NORMAL</td>
      <td>2018-04-01 00:04:00</td>
    </tr>
  </tbody>
</table>
</div>



## Data Preprocessing


```python
sensors_df = data.drop(['sensor_15', 'Unnamed: 0', 'datetime', 'timestamp', 'machine_status'], axis = 1).copy()
```


```python
sensors_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sensor_00</th>
      <th>sensor_01</th>
      <th>sensor_02</th>
      <th>sensor_03</th>
      <th>sensor_04</th>
      <th>sensor_05</th>
      <th>sensor_06</th>
      <th>sensor_07</th>
      <th>sensor_08</th>
      <th>sensor_09</th>
      <th>sensor_10</th>
      <th>sensor_11</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_16</th>
      <th>sensor_17</th>
      <th>sensor_18</th>
      <th>sensor_19</th>
      <th>sensor_20</th>
      <th>sensor_21</th>
      <th>sensor_22</th>
      <th>sensor_23</th>
      <th>sensor_24</th>
      <th>sensor_25</th>
      <th>sensor_26</th>
      <th>sensor_27</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_32</th>
      <th>sensor_33</th>
      <th>sensor_34</th>
      <th>sensor_35</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_38</th>
      <th>sensor_39</th>
      <th>sensor_40</th>
      <th>sensor_41</th>
      <th>sensor_42</th>
      <th>sensor_43</th>
      <th>sensor_44</th>
      <th>sensor_45</th>
      <th>sensor_46</th>
      <th>sensor_47</th>
      <th>sensor_48</th>
      <th>sensor_49</th>
      <th>sensor_50</th>
      <th>sensor_51</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.465394</td>
      <td>47.09201</td>
      <td>53.2118</td>
      <td>46.310760</td>
      <td>634.3750</td>
      <td>76.45975</td>
      <td>13.41146</td>
      <td>16.13136</td>
      <td>15.56713</td>
      <td>15.05353</td>
      <td>37.22740</td>
      <td>47.52422</td>
      <td>31.11716</td>
      <td>1.681353</td>
      <td>419.5747</td>
      <td>461.8781</td>
      <td>466.3284</td>
      <td>2.565284</td>
      <td>665.3993</td>
      <td>398.9862</td>
      <td>880.0001</td>
      <td>498.8926</td>
      <td>975.9409</td>
      <td>627.6740</td>
      <td>741.7151</td>
      <td>848.0708</td>
      <td>429.0377</td>
      <td>785.1935</td>
      <td>684.9443</td>
      <td>594.4445</td>
      <td>682.8125</td>
      <td>680.4416</td>
      <td>433.7037</td>
      <td>171.9375</td>
      <td>341.9039</td>
      <td>195.0655</td>
      <td>90.32386</td>
      <td>40.36458</td>
      <td>31.51042</td>
      <td>70.57291</td>
      <td>30.98958</td>
      <td>31.770832</td>
      <td>41.92708</td>
      <td>39.641200</td>
      <td>65.68287</td>
      <td>50.92593</td>
      <td>38.194440</td>
      <td>157.9861</td>
      <td>67.70834</td>
      <td>243.0556</td>
      <td>201.3889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.444734</td>
      <td>47.35243</td>
      <td>53.2118</td>
      <td>46.397570</td>
      <td>638.8889</td>
      <td>73.54598</td>
      <td>13.32465</td>
      <td>16.03733</td>
      <td>15.61777</td>
      <td>15.01013</td>
      <td>37.86777</td>
      <td>48.17723</td>
      <td>32.08894</td>
      <td>1.708474</td>
      <td>420.8480</td>
      <td>462.7798</td>
      <td>459.6364</td>
      <td>2.500062</td>
      <td>666.2234</td>
      <td>399.9418</td>
      <td>880.4237</td>
      <td>501.3617</td>
      <td>982.7342</td>
      <td>631.1326</td>
      <td>740.8031</td>
      <td>849.8997</td>
      <td>454.2390</td>
      <td>778.5734</td>
      <td>715.6266</td>
      <td>661.5740</td>
      <td>721.8750</td>
      <td>694.7721</td>
      <td>441.2635</td>
      <td>169.9820</td>
      <td>343.1955</td>
      <td>200.9694</td>
      <td>93.90508</td>
      <td>41.40625</td>
      <td>31.25000</td>
      <td>69.53125</td>
      <td>30.46875</td>
      <td>31.770830</td>
      <td>41.66666</td>
      <td>39.351852</td>
      <td>65.39352</td>
      <td>51.21528</td>
      <td>38.194443</td>
      <td>155.9606</td>
      <td>67.12963</td>
      <td>241.3194</td>
      <td>203.7037</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.460474</td>
      <td>47.09201</td>
      <td>53.1684</td>
      <td>46.397568</td>
      <td>628.1250</td>
      <td>76.98898</td>
      <td>13.31742</td>
      <td>16.24711</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>38.57977</td>
      <td>48.65607</td>
      <td>31.67221</td>
      <td>1.579427</td>
      <td>420.7494</td>
      <td>462.8980</td>
      <td>460.8858</td>
      <td>2.509521</td>
      <td>666.0114</td>
      <td>399.1046</td>
      <td>878.8917</td>
      <td>499.0430</td>
      <td>977.7520</td>
      <td>625.4076</td>
      <td>739.2722</td>
      <td>847.7579</td>
      <td>474.8731</td>
      <td>779.5091</td>
      <td>690.4011</td>
      <td>686.1111</td>
      <td>754.6875</td>
      <td>683.3831</td>
      <td>446.2493</td>
      <td>166.4987</td>
      <td>343.9586</td>
      <td>193.1689</td>
      <td>101.04060</td>
      <td>41.92708</td>
      <td>31.51042</td>
      <td>72.13541</td>
      <td>30.46875</td>
      <td>31.510420</td>
      <td>40.88541</td>
      <td>39.062500</td>
      <td>64.81481</td>
      <td>51.21528</td>
      <td>38.194440</td>
      <td>155.9606</td>
      <td>66.84028</td>
      <td>240.4514</td>
      <td>203.1250</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.445718</td>
      <td>47.13541</td>
      <td>53.2118</td>
      <td>46.397568</td>
      <td>636.4583</td>
      <td>76.58897</td>
      <td>13.35359</td>
      <td>16.21094</td>
      <td>15.69734</td>
      <td>15.08247</td>
      <td>39.48939</td>
      <td>49.06298</td>
      <td>31.95202</td>
      <td>1.683831</td>
      <td>419.8926</td>
      <td>461.4906</td>
      <td>468.2206</td>
      <td>2.604785</td>
      <td>663.2111</td>
      <td>400.5426</td>
      <td>882.5874</td>
      <td>498.5383</td>
      <td>979.5755</td>
      <td>627.1830</td>
      <td>737.6033</td>
      <td>846.9182</td>
      <td>408.8159</td>
      <td>785.2307</td>
      <td>704.6937</td>
      <td>631.4814</td>
      <td>766.1458</td>
      <td>702.4431</td>
      <td>433.9081</td>
      <td>164.7498</td>
      <td>339.9630</td>
      <td>193.8770</td>
      <td>101.70380</td>
      <td>42.70833</td>
      <td>31.51042</td>
      <td>76.82291</td>
      <td>30.98958</td>
      <td>31.510420</td>
      <td>41.40625</td>
      <td>38.773150</td>
      <td>65.10416</td>
      <td>51.79398</td>
      <td>38.773150</td>
      <td>158.2755</td>
      <td>66.55093</td>
      <td>242.1875</td>
      <td>201.3889</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Data Imputation


```python
sensors_df.isnull().mean()
```




    sensor_00    0.046333
    sensor_01    0.001675
    sensor_02    0.000086
    sensor_03    0.000086
    sensor_04    0.000086
    sensor_05    0.000086
    sensor_06    0.021777
    sensor_07    0.024741
    sensor_08    0.023180
    sensor_09    0.020856
    sensor_10    0.000086
    sensor_11    0.000086
    sensor_12    0.000086
    sensor_13    0.000086
    sensor_14    0.000095
    sensor_16    0.000141
    sensor_17    0.000209
    sensor_18    0.000209
    sensor_19    0.000073
    sensor_20    0.000073
    sensor_21    0.000073
    sensor_22    0.000186
    sensor_23    0.000073
    sensor_24    0.000073
    sensor_25    0.000163
    sensor_26    0.000091
    sensor_27    0.000073
    sensor_28    0.000073
    sensor_29    0.000327
    sensor_30    0.001185
    sensor_31    0.000073
    sensor_32    0.000309
    sensor_33    0.000073
    sensor_34    0.000073
    sensor_35    0.000073
    sensor_36    0.000073
    sensor_37    0.000073
    sensor_38    0.000123
    sensor_39    0.000123
    sensor_40    0.000123
    sensor_41    0.000123
    sensor_42    0.000123
    sensor_43    0.000123
    sensor_44    0.000123
    sensor_45    0.000123
    sensor_46    0.000123
    sensor_47    0.000123
    sensor_48    0.000123
    sensor_49    0.000123
    sensor_50    0.349569
    sensor_51    0.069821
    dtype: float64




```python
imputer = RandomSampleImputer(
        seeding_method = 'add'
)

imputer.fit(sensors_df)
```




    RandomSampleImputer()




```python
sensors_df = imputer.transform(sensors_df)
```


```python
diagnostic_plots(sensors_df, "sensor_50")
```


    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_27_1.png)
    



```python
sensors_df.isnull().mean()
```




    sensor_00    0.0
    sensor_01    0.0
    sensor_02    0.0
    sensor_03    0.0
    sensor_04    0.0
    sensor_05    0.0
    sensor_06    0.0
    sensor_07    0.0
    sensor_08    0.0
    sensor_09    0.0
    sensor_10    0.0
    sensor_11    0.0
    sensor_12    0.0
    sensor_13    0.0
    sensor_14    0.0
    sensor_16    0.0
    sensor_17    0.0
    sensor_18    0.0
    sensor_19    0.0
    sensor_20    0.0
    sensor_21    0.0
    sensor_22    0.0
    sensor_23    0.0
    sensor_24    0.0
    sensor_25    0.0
    sensor_26    0.0
    sensor_27    0.0
    sensor_28    0.0
    sensor_29    0.0
    sensor_30    0.0
    sensor_31    0.0
    sensor_32    0.0
    sensor_33    0.0
    sensor_34    0.0
    sensor_35    0.0
    sensor_36    0.0
    sensor_37    0.0
    sensor_38    0.0
    sensor_39    0.0
    sensor_40    0.0
    sensor_41    0.0
    sensor_42    0.0
    sensor_43    0.0
    sensor_44    0.0
    sensor_45    0.0
    sensor_46    0.0
    sensor_47    0.0
    sensor_48    0.0
    sensor_49    0.0
    sensor_50    0.0
    sensor_51    0.0
    dtype: float64




```python
for var in sensors_df.columns:
    diagnostic_plots(sensors_df, var)
```


    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_1.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_3.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_5.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_7.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_9.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_11.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_13.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_15.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_17.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_19.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_21.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_23.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_25.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_27.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_29.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_31.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_33.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_35.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_37.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_39.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_41.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_43.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_45.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_47.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_49.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_51.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_53.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_55.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_57.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_59.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_61.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_63.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_65.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_67.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_69.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_71.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_73.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_75.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_77.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_79.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_81.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_83.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_85.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_87.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_89.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_91.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_93.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_95.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_97.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_99.png)
    



    <Figure size 432x288 with 0 Axes>



    
![png](README_files/README_29_101.png)
    


### Feature Scaling


```python
# Perform feature scaling using MinMaxScaler

scaler = MinMaxScaler()

sensors_df_scaled = scaler.fit_transform(sensors_df)

sensors_df_scaled = pd.DataFrame(sensors_df_scaled)

sensors_df_scaled.columns = list(sensors_df.columns)
```


```python
sensors_df_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sensor_00</th>
      <th>sensor_01</th>
      <th>sensor_02</th>
      <th>sensor_03</th>
      <th>sensor_04</th>
      <th>sensor_05</th>
      <th>sensor_06</th>
      <th>sensor_07</th>
      <th>sensor_08</th>
      <th>sensor_09</th>
      <th>sensor_10</th>
      <th>sensor_11</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_16</th>
      <th>sensor_17</th>
      <th>sensor_18</th>
      <th>sensor_19</th>
      <th>sensor_20</th>
      <th>sensor_21</th>
      <th>sensor_22</th>
      <th>sensor_23</th>
      <th>sensor_24</th>
      <th>sensor_25</th>
      <th>sensor_26</th>
      <th>sensor_27</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_32</th>
      <th>sensor_33</th>
      <th>sensor_34</th>
      <th>sensor_35</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_38</th>
      <th>sensor_39</th>
      <th>sensor_40</th>
      <th>sensor_41</th>
      <th>sensor_42</th>
      <th>sensor_43</th>
      <th>sensor_44</th>
      <th>sensor_45</th>
      <th>sensor_46</th>
      <th>sensor_47</th>
      <th>sensor_48</th>
      <th>sensor_49</th>
      <th>sensor_50</th>
      <th>sensor_51</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.967194</td>
      <td>0.830145</td>
      <td>0.876660</td>
      <td>0.884816</td>
      <td>0.792242</td>
      <td>0.764598</td>
      <td>0.602472</td>
      <td>0.683630</td>
      <td>0.638905</td>
      <td>0.602141</td>
      <td>0.489146</td>
      <td>0.792070</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.624378</td>
      <td>0.777214</td>
      <td>0.526401</td>
      <td>0.757067</td>
      <td>0.888793</td>
      <td>0.775172</td>
      <td>0.839800</td>
      <td>0.795022</td>
      <td>0.627674</td>
      <td>0.883441</td>
      <td>0.687219</td>
      <td>0.214519</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.369881</td>
      <td>0.271759</td>
      <td>0.315796</td>
      <td>0.492317</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.040397</td>
      <td>0.023153</td>
      <td>0.096328</td>
      <td>0.025424</td>
      <td>0.027367</td>
      <td>0.045424</td>
      <td>0.014256</td>
      <td>0.133858</td>
      <td>0.071489</td>
      <td>0.039791</td>
      <td>0.245946</td>
      <td>0.093853</td>
      <td>0.221660</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.967194</td>
      <td>0.830145</td>
      <td>0.876660</td>
      <td>0.884816</td>
      <td>0.792242</td>
      <td>0.764598</td>
      <td>0.602472</td>
      <td>0.683630</td>
      <td>0.638905</td>
      <td>0.602141</td>
      <td>0.489146</td>
      <td>0.792070</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.624378</td>
      <td>0.777214</td>
      <td>0.526401</td>
      <td>0.757067</td>
      <td>0.888793</td>
      <td>0.775172</td>
      <td>0.839800</td>
      <td>0.795022</td>
      <td>0.627674</td>
      <td>0.883441</td>
      <td>0.687219</td>
      <td>0.214519</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.369881</td>
      <td>0.271759</td>
      <td>0.315796</td>
      <td>0.492317</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.040397</td>
      <td>0.023153</td>
      <td>0.096328</td>
      <td>0.025424</td>
      <td>0.027367</td>
      <td>0.045424</td>
      <td>0.014256</td>
      <td>0.133858</td>
      <td>0.071489</td>
      <td>0.039791</td>
      <td>0.245946</td>
      <td>0.093853</td>
      <td>0.221660</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959089</td>
      <td>0.834736</td>
      <td>0.876660</td>
      <td>0.890052</td>
      <td>0.797904</td>
      <td>0.735461</td>
      <td>0.598568</td>
      <td>0.679645</td>
      <td>0.640988</td>
      <td>0.600405</td>
      <td>0.497561</td>
      <td>0.802954</td>
      <td>0.713088</td>
      <td>0.054781</td>
      <td>0.830724</td>
      <td>0.625597</td>
      <td>0.766061</td>
      <td>0.513017</td>
      <td>0.758004</td>
      <td>0.890922</td>
      <td>0.775590</td>
      <td>0.843956</td>
      <td>0.800556</td>
      <td>0.631133</td>
      <td>0.882355</td>
      <td>0.688781</td>
      <td>0.227120</td>
      <td>0.421517</td>
      <td>0.487833</td>
      <td>0.413484</td>
      <td>0.392962</td>
      <td>0.377674</td>
      <td>0.276568</td>
      <td>0.310520</td>
      <td>0.494177</td>
      <td>0.202392</td>
      <td>0.536904</td>
      <td>0.043046</td>
      <td>0.022660</td>
      <td>0.094199</td>
      <td>0.024120</td>
      <td>0.027367</td>
      <td>0.044746</td>
      <td>0.013959</td>
      <td>0.132874</td>
      <td>0.072330</td>
      <td>0.039791</td>
      <td>0.242162</td>
      <td>0.092531</td>
      <td>0.219875</td>
      <td>0.180952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.965264</td>
      <td>0.830145</td>
      <td>0.874763</td>
      <td>0.890052</td>
      <td>0.784402</td>
      <td>0.769891</td>
      <td>0.598243</td>
      <td>0.688535</td>
      <td>0.644259</td>
      <td>0.603299</td>
      <td>0.506916</td>
      <td>0.810935</td>
      <td>0.703827</td>
      <td>0.050643</td>
      <td>0.830513</td>
      <td>0.625756</td>
      <td>0.768143</td>
      <td>0.514958</td>
      <td>0.757763</td>
      <td>0.889057</td>
      <td>0.774076</td>
      <td>0.840053</td>
      <td>0.796498</td>
      <td>0.625408</td>
      <td>0.880531</td>
      <td>0.686952</td>
      <td>0.237437</td>
      <td>0.422027</td>
      <td>0.470622</td>
      <td>0.428819</td>
      <td>0.411437</td>
      <td>0.371481</td>
      <td>0.279739</td>
      <td>0.301123</td>
      <td>0.495276</td>
      <td>0.194447</td>
      <td>0.577701</td>
      <td>0.044371</td>
      <td>0.023153</td>
      <td>0.099521</td>
      <td>0.024120</td>
      <td>0.026627</td>
      <td>0.042712</td>
      <td>0.013662</td>
      <td>0.130905</td>
      <td>0.072330</td>
      <td>0.039791</td>
      <td>0.242162</td>
      <td>0.091870</td>
      <td>0.218982</td>
      <td>0.180357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.959475</td>
      <td>0.830910</td>
      <td>0.876660</td>
      <td>0.890052</td>
      <td>0.794855</td>
      <td>0.765891</td>
      <td>0.599870</td>
      <td>0.687002</td>
      <td>0.644259</td>
      <td>0.603299</td>
      <td>0.518868</td>
      <td>0.817716</td>
      <td>0.710045</td>
      <td>0.053990</td>
      <td>0.828680</td>
      <td>0.623854</td>
      <td>0.780368</td>
      <td>0.534507</td>
      <td>0.754577</td>
      <td>0.892260</td>
      <td>0.777728</td>
      <td>0.839204</td>
      <td>0.797983</td>
      <td>0.627183</td>
      <td>0.878544</td>
      <td>0.686235</td>
      <td>0.204408</td>
      <td>0.425142</td>
      <td>0.480374</td>
      <td>0.394676</td>
      <td>0.417889</td>
      <td>0.381845</td>
      <td>0.271889</td>
      <td>0.296404</td>
      <td>0.489522</td>
      <td>0.195168</td>
      <td>0.581493</td>
      <td>0.046358</td>
      <td>0.023153</td>
      <td>0.109101</td>
      <td>0.025424</td>
      <td>0.026627</td>
      <td>0.044068</td>
      <td>0.013365</td>
      <td>0.131890</td>
      <td>0.074012</td>
      <td>0.041885</td>
      <td>0.246487</td>
      <td>0.091210</td>
      <td>0.220768</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>220315</th>
      <td>0.944423</td>
      <td>0.840857</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.792678</td>
      <td>0.645910</td>
      <td>0.679245</td>
      <td>0.705702</td>
      <td>0.642474</td>
      <td>0.606482</td>
      <td>0.567240</td>
      <td>0.902675</td>
      <td>0.845650</td>
      <td>0.425340</td>
      <td>0.830620</td>
      <td>0.626208</td>
      <td>0.763936</td>
      <td>0.512823</td>
      <td>0.769885</td>
      <td>0.903900</td>
      <td>0.789591</td>
      <td>0.915024</td>
      <td>0.903823</td>
      <td>0.611174</td>
      <td>0.834456</td>
      <td>0.643272</td>
      <td>0.346057</td>
      <td>0.421862</td>
      <td>0.330503</td>
      <td>0.432292</td>
      <td>0.535484</td>
      <td>0.504289</td>
      <td>0.299500</td>
      <td>0.569651</td>
      <td>0.833030</td>
      <td>0.830424</td>
      <td>0.000000</td>
      <td>0.057616</td>
      <td>0.018719</td>
      <td>0.098457</td>
      <td>0.024120</td>
      <td>0.022929</td>
      <td>0.035932</td>
      <td>0.043659</td>
      <td>0.088583</td>
      <td>0.063919</td>
      <td>0.050262</td>
      <td>0.347568</td>
      <td>0.290152</td>
      <td>0.005653</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220316</th>
      <td>0.941721</td>
      <td>0.840857</td>
      <td>0.760911</td>
      <td>0.693717</td>
      <td>0.787887</td>
      <td>0.658337</td>
      <td>0.680872</td>
      <td>0.707848</td>
      <td>0.642474</td>
      <td>0.604745</td>
      <td>0.567759</td>
      <td>0.908767</td>
      <td>0.856330</td>
      <td>0.424601</td>
      <td>0.833522</td>
      <td>0.626155</td>
      <td>0.780731</td>
      <td>0.537316</td>
      <td>0.769872</td>
      <td>0.904991</td>
      <td>0.790547</td>
      <td>0.911861</td>
      <td>0.901274</td>
      <td>0.609492</td>
      <td>0.831958</td>
      <td>0.646340</td>
      <td>0.348900</td>
      <td>0.431852</td>
      <td>0.348184</td>
      <td>0.420139</td>
      <td>0.508504</td>
      <td>0.493594</td>
      <td>0.306212</td>
      <td>0.559369</td>
      <td>0.818028</td>
      <td>0.819672</td>
      <td>0.000000</td>
      <td>0.056954</td>
      <td>0.018227</td>
      <td>0.101650</td>
      <td>0.023468</td>
      <td>0.022189</td>
      <td>0.035932</td>
      <td>0.042174</td>
      <td>0.082677</td>
      <td>0.063078</td>
      <td>0.049215</td>
      <td>0.350270</td>
      <td>0.296100</td>
      <td>0.161857</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220317</th>
      <td>0.940178</td>
      <td>0.840857</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.781644</td>
      <td>0.672945</td>
      <td>0.677944</td>
      <td>0.707848</td>
      <td>0.644259</td>
      <td>0.604745</td>
      <td>0.566682</td>
      <td>0.918630</td>
      <td>0.856151</td>
      <td>0.422882</td>
      <td>0.829373</td>
      <td>0.625092</td>
      <td>0.781049</td>
      <td>0.537731</td>
      <td>0.770625</td>
      <td>0.906900</td>
      <td>0.787247</td>
      <td>0.913808</td>
      <td>0.901540</td>
      <td>0.610994</td>
      <td>0.837524</td>
      <td>0.646499</td>
      <td>0.352330</td>
      <td>0.432808</td>
      <td>0.335781</td>
      <td>0.430845</td>
      <td>0.507038</td>
      <td>0.503852</td>
      <td>0.310192</td>
      <td>0.555632</td>
      <td>0.797558</td>
      <td>0.818191</td>
      <td>0.000000</td>
      <td>0.054967</td>
      <td>0.017734</td>
      <td>0.109633</td>
      <td>0.022816</td>
      <td>0.022929</td>
      <td>0.037966</td>
      <td>0.040689</td>
      <td>0.076772</td>
      <td>0.063078</td>
      <td>0.049215</td>
      <td>0.356757</td>
      <td>0.294118</td>
      <td>0.250223</td>
      <td>0.210119</td>
    </tr>
    <tr>
      <th>220318</th>
      <td>0.944037</td>
      <td>0.840857</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.793839</td>
      <td>0.650918</td>
      <td>0.679245</td>
      <td>0.702023</td>
      <td>0.646044</td>
      <td>0.604745</td>
      <td>0.556552</td>
      <td>0.933220</td>
      <td>0.864258</td>
      <td>0.422395</td>
      <td>0.830129</td>
      <td>0.617832</td>
      <td>0.766324</td>
      <td>0.516000</td>
      <td>0.765278</td>
      <td>0.900692</td>
      <td>0.783074</td>
      <td>0.907925</td>
      <td>0.899305</td>
      <td>0.605718</td>
      <td>0.830624</td>
      <td>0.640805</td>
      <td>0.353485</td>
      <td>0.429405</td>
      <td>0.334038</td>
      <td>0.429398</td>
      <td>0.511144</td>
      <td>0.497667</td>
      <td>0.303825</td>
      <td>0.556931</td>
      <td>0.805559</td>
      <td>0.819739</td>
      <td>0.000000</td>
      <td>0.054305</td>
      <td>0.017241</td>
      <td>0.112826</td>
      <td>0.022816</td>
      <td>0.022929</td>
      <td>0.042034</td>
      <td>0.039501</td>
      <td>0.072835</td>
      <td>0.063919</td>
      <td>0.048168</td>
      <td>0.366486</td>
      <td>0.290813</td>
      <td>0.129723</td>
      <td>0.212202</td>
    </tr>
    <tr>
      <th>220319</th>
      <td>0.940178</td>
      <td>0.840857</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.799066</td>
      <td>0.654564</td>
      <td>0.679245</td>
      <td>0.705702</td>
      <td>0.642474</td>
      <td>0.600405</td>
      <td>0.560109</td>
      <td>0.941607</td>
      <td>0.875768</td>
      <td>0.420871</td>
      <td>0.831494</td>
      <td>0.633994</td>
      <td>0.760954</td>
      <td>0.510398</td>
      <td>0.769791</td>
      <td>0.903591</td>
      <td>0.792811</td>
      <td>0.912524</td>
      <td>0.903274</td>
      <td>0.608536</td>
      <td>0.831467</td>
      <td>0.646211</td>
      <td>0.351813</td>
      <td>0.433299</td>
      <td>0.338261</td>
      <td>0.428819</td>
      <td>0.503226</td>
      <td>0.503628</td>
      <td>0.306955</td>
      <td>0.549162</td>
      <td>0.803560</td>
      <td>0.823854</td>
      <td>0.000000</td>
      <td>0.052980</td>
      <td>0.016256</td>
      <td>0.111229</td>
      <td>0.022816</td>
      <td>0.022929</td>
      <td>0.044068</td>
      <td>0.038016</td>
      <td>0.067913</td>
      <td>0.065601</td>
      <td>0.047120</td>
      <td>0.375676</td>
      <td>0.282882</td>
      <td>0.163344</td>
      <td>0.212202</td>
    </tr>
  </tbody>
</table>
<p>220320 rows × 51 columns</p>
</div>



### Feature Selection

#### Remove constant features


```python
# Check for constant features in dataset
constant_features = [var for var in sensors_df_scaled.columns if sensors_df_scaled[var].std() == 0] 

constant_features
```




    []



#### Remove quasi-constant features


```python
# Remove quasi-constant features where 99% of the values are similar
remover = VarianceThreshold(threshold = 0.01)

# Find the values with low variance
remover.fit(sensors_df_scaled)

# Print remaining variables
print(remover.get_feature_names_out())
```

    ['sensor_00' 'sensor_02' 'sensor_03' 'sensor_04' 'sensor_05' 'sensor_10'
     'sensor_11' 'sensor_12' 'sensor_13' 'sensor_14' 'sensor_16' 'sensor_17'
     'sensor_18' 'sensor_19' 'sensor_20' 'sensor_21' 'sensor_22' 'sensor_23'
     'sensor_24' 'sensor_25' 'sensor_26' 'sensor_28' 'sensor_29' 'sensor_30'
     'sensor_31' 'sensor_32' 'sensor_34' 'sensor_35' 'sensor_36' 'sensor_37'
     'sensor_48' 'sensor_51']
    


```python
sensors_df_scaled = remover.transform(sensors_df_scaled)
```


```python
sensors_df_scaled = pd.DataFrame(sensors_df_scaled)

sensors_df_scaled.columns = remover.get_feature_names_out()
```


```python
sensors_df_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sensor_00</th>
      <th>sensor_02</th>
      <th>sensor_03</th>
      <th>sensor_04</th>
      <th>sensor_05</th>
      <th>sensor_10</th>
      <th>sensor_11</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_16</th>
      <th>sensor_17</th>
      <th>sensor_18</th>
      <th>sensor_19</th>
      <th>sensor_20</th>
      <th>sensor_21</th>
      <th>sensor_22</th>
      <th>sensor_23</th>
      <th>sensor_24</th>
      <th>sensor_25</th>
      <th>sensor_26</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_32</th>
      <th>sensor_34</th>
      <th>sensor_35</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_48</th>
      <th>sensor_51</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.884816</td>
      <td>0.792242</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.792070</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.624378</td>
      <td>0.777214</td>
      <td>0.526401</td>
      <td>0.757067</td>
      <td>0.888793</td>
      <td>0.775172</td>
      <td>0.839800</td>
      <td>0.795022</td>
      <td>0.627674</td>
      <td>0.883441</td>
      <td>0.687219</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.369881</td>
      <td>0.315796</td>
      <td>0.492317</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.884816</td>
      <td>0.792242</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.792070</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.624378</td>
      <td>0.777214</td>
      <td>0.526401</td>
      <td>0.757067</td>
      <td>0.888793</td>
      <td>0.775172</td>
      <td>0.839800</td>
      <td>0.795022</td>
      <td>0.627674</td>
      <td>0.883441</td>
      <td>0.687219</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.369881</td>
      <td>0.315796</td>
      <td>0.492317</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959089</td>
      <td>0.876660</td>
      <td>0.890052</td>
      <td>0.797904</td>
      <td>0.735461</td>
      <td>0.497561</td>
      <td>0.802954</td>
      <td>0.713088</td>
      <td>0.054781</td>
      <td>0.830724</td>
      <td>0.625597</td>
      <td>0.766061</td>
      <td>0.513017</td>
      <td>0.758004</td>
      <td>0.890922</td>
      <td>0.775590</td>
      <td>0.843956</td>
      <td>0.800556</td>
      <td>0.631133</td>
      <td>0.882355</td>
      <td>0.688781</td>
      <td>0.421517</td>
      <td>0.487833</td>
      <td>0.413484</td>
      <td>0.392962</td>
      <td>0.377674</td>
      <td>0.310520</td>
      <td>0.494177</td>
      <td>0.202392</td>
      <td>0.536904</td>
      <td>0.242162</td>
      <td>0.180952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.965264</td>
      <td>0.874763</td>
      <td>0.890052</td>
      <td>0.784402</td>
      <td>0.769891</td>
      <td>0.506916</td>
      <td>0.810935</td>
      <td>0.703827</td>
      <td>0.050643</td>
      <td>0.830513</td>
      <td>0.625756</td>
      <td>0.768143</td>
      <td>0.514958</td>
      <td>0.757763</td>
      <td>0.889057</td>
      <td>0.774076</td>
      <td>0.840053</td>
      <td>0.796498</td>
      <td>0.625408</td>
      <td>0.880531</td>
      <td>0.686952</td>
      <td>0.422027</td>
      <td>0.470622</td>
      <td>0.428819</td>
      <td>0.411437</td>
      <td>0.371481</td>
      <td>0.301123</td>
      <td>0.495276</td>
      <td>0.194447</td>
      <td>0.577701</td>
      <td>0.242162</td>
      <td>0.180357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.959475</td>
      <td>0.876660</td>
      <td>0.890052</td>
      <td>0.794855</td>
      <td>0.765891</td>
      <td>0.518868</td>
      <td>0.817716</td>
      <td>0.710045</td>
      <td>0.053990</td>
      <td>0.828680</td>
      <td>0.623854</td>
      <td>0.780368</td>
      <td>0.534507</td>
      <td>0.754577</td>
      <td>0.892260</td>
      <td>0.777728</td>
      <td>0.839204</td>
      <td>0.797983</td>
      <td>0.627183</td>
      <td>0.878544</td>
      <td>0.686235</td>
      <td>0.425142</td>
      <td>0.480374</td>
      <td>0.394676</td>
      <td>0.417889</td>
      <td>0.381845</td>
      <td>0.296404</td>
      <td>0.489522</td>
      <td>0.195168</td>
      <td>0.581493</td>
      <td>0.246487</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>220315</th>
      <td>0.944423</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.792678</td>
      <td>0.645910</td>
      <td>0.567240</td>
      <td>0.902675</td>
      <td>0.845650</td>
      <td>0.425340</td>
      <td>0.830620</td>
      <td>0.626208</td>
      <td>0.763936</td>
      <td>0.512823</td>
      <td>0.769885</td>
      <td>0.903900</td>
      <td>0.789591</td>
      <td>0.915024</td>
      <td>0.903823</td>
      <td>0.611174</td>
      <td>0.834456</td>
      <td>0.643272</td>
      <td>0.421862</td>
      <td>0.330503</td>
      <td>0.432292</td>
      <td>0.535484</td>
      <td>0.504289</td>
      <td>0.569651</td>
      <td>0.833030</td>
      <td>0.830424</td>
      <td>0.000000</td>
      <td>0.347568</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220316</th>
      <td>0.941721</td>
      <td>0.760911</td>
      <td>0.693717</td>
      <td>0.787887</td>
      <td>0.658337</td>
      <td>0.567759</td>
      <td>0.908767</td>
      <td>0.856330</td>
      <td>0.424601</td>
      <td>0.833522</td>
      <td>0.626155</td>
      <td>0.780731</td>
      <td>0.537316</td>
      <td>0.769872</td>
      <td>0.904991</td>
      <td>0.790547</td>
      <td>0.911861</td>
      <td>0.901274</td>
      <td>0.609492</td>
      <td>0.831958</td>
      <td>0.646340</td>
      <td>0.431852</td>
      <td>0.348184</td>
      <td>0.420139</td>
      <td>0.508504</td>
      <td>0.493594</td>
      <td>0.559369</td>
      <td>0.818028</td>
      <td>0.819672</td>
      <td>0.000000</td>
      <td>0.350270</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220317</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.781644</td>
      <td>0.672945</td>
      <td>0.566682</td>
      <td>0.918630</td>
      <td>0.856151</td>
      <td>0.422882</td>
      <td>0.829373</td>
      <td>0.625092</td>
      <td>0.781049</td>
      <td>0.537731</td>
      <td>0.770625</td>
      <td>0.906900</td>
      <td>0.787247</td>
      <td>0.913808</td>
      <td>0.901540</td>
      <td>0.610994</td>
      <td>0.837524</td>
      <td>0.646499</td>
      <td>0.432808</td>
      <td>0.335781</td>
      <td>0.430845</td>
      <td>0.507038</td>
      <td>0.503852</td>
      <td>0.555632</td>
      <td>0.797558</td>
      <td>0.818191</td>
      <td>0.000000</td>
      <td>0.356757</td>
      <td>0.210119</td>
    </tr>
    <tr>
      <th>220318</th>
      <td>0.944037</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.793839</td>
      <td>0.650918</td>
      <td>0.556552</td>
      <td>0.933220</td>
      <td>0.864258</td>
      <td>0.422395</td>
      <td>0.830129</td>
      <td>0.617832</td>
      <td>0.766324</td>
      <td>0.516000</td>
      <td>0.765278</td>
      <td>0.900692</td>
      <td>0.783074</td>
      <td>0.907925</td>
      <td>0.899305</td>
      <td>0.605718</td>
      <td>0.830624</td>
      <td>0.640805</td>
      <td>0.429405</td>
      <td>0.334038</td>
      <td>0.429398</td>
      <td>0.511144</td>
      <td>0.497667</td>
      <td>0.556931</td>
      <td>0.805559</td>
      <td>0.819739</td>
      <td>0.000000</td>
      <td>0.366486</td>
      <td>0.212202</td>
    </tr>
    <tr>
      <th>220319</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.693717</td>
      <td>0.799066</td>
      <td>0.654564</td>
      <td>0.560109</td>
      <td>0.941607</td>
      <td>0.875768</td>
      <td>0.420871</td>
      <td>0.831494</td>
      <td>0.633994</td>
      <td>0.760954</td>
      <td>0.510398</td>
      <td>0.769791</td>
      <td>0.903591</td>
      <td>0.792811</td>
      <td>0.912524</td>
      <td>0.903274</td>
      <td>0.608536</td>
      <td>0.831467</td>
      <td>0.646211</td>
      <td>0.433299</td>
      <td>0.338261</td>
      <td>0.428819</td>
      <td>0.503226</td>
      <td>0.503628</td>
      <td>0.549162</td>
      <td>0.803560</td>
      <td>0.823854</td>
      <td>0.000000</td>
      <td>0.375676</td>
      <td>0.212202</td>
    </tr>
  </tbody>
</table>
<p>220320 rows × 32 columns</p>
</div>



#### Remove correlated features


```python
# function first calculates the correlations between the columns of the dataset 

correlated = DropCorrelatedFeatures(method='pearson', threshold=0.8)
```


```python
correlated.fit(sensors_df_scaled)
```




    DropCorrelatedFeatures()




```python
correlated.correlated_feature_sets_
```




    [{'sensor_02', 'sensor_03', 'sensor_04'},
     {'sensor_10', 'sensor_11'},
     {'sensor_14',
      'sensor_16',
      'sensor_17',
      'sensor_18',
      'sensor_19',
      'sensor_20',
      'sensor_21',
      'sensor_22',
      'sensor_23',
      'sensor_24',
      'sensor_25',
      'sensor_26'},
     {'sensor_30', 'sensor_32'},
     {'sensor_34', 'sensor_35'}]




```python
sensors_df_scaled = correlated.transform(sensors_df_scaled)
```


```python
# Removed 16 correlated features
sensors_df_scaled
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sensor_00</th>
      <th>sensor_02</th>
      <th>sensor_05</th>
      <th>sensor_10</th>
      <th>sensor_12</th>
      <th>sensor_13</th>
      <th>sensor_14</th>
      <th>sensor_28</th>
      <th>sensor_29</th>
      <th>sensor_30</th>
      <th>sensor_31</th>
      <th>sensor_34</th>
      <th>sensor_36</th>
      <th>sensor_37</th>
      <th>sensor_48</th>
      <th>sensor_51</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959089</td>
      <td>0.876660</td>
      <td>0.735461</td>
      <td>0.497561</td>
      <td>0.713088</td>
      <td>0.054781</td>
      <td>0.830724</td>
      <td>0.421517</td>
      <td>0.487833</td>
      <td>0.413484</td>
      <td>0.392962</td>
      <td>0.310520</td>
      <td>0.202392</td>
      <td>0.536904</td>
      <td>0.242162</td>
      <td>0.180952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.965264</td>
      <td>0.874763</td>
      <td>0.769891</td>
      <td>0.506916</td>
      <td>0.703827</td>
      <td>0.050643</td>
      <td>0.830513</td>
      <td>0.422027</td>
      <td>0.470622</td>
      <td>0.428819</td>
      <td>0.411437</td>
      <td>0.301123</td>
      <td>0.194447</td>
      <td>0.577701</td>
      <td>0.242162</td>
      <td>0.180357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.959475</td>
      <td>0.876660</td>
      <td>0.765891</td>
      <td>0.518868</td>
      <td>0.710045</td>
      <td>0.053990</td>
      <td>0.828680</td>
      <td>0.425142</td>
      <td>0.480374</td>
      <td>0.394676</td>
      <td>0.417889</td>
      <td>0.296404</td>
      <td>0.195168</td>
      <td>0.581493</td>
      <td>0.246487</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>220315</th>
      <td>0.944423</td>
      <td>0.759013</td>
      <td>0.645910</td>
      <td>0.567240</td>
      <td>0.845650</td>
      <td>0.425340</td>
      <td>0.830620</td>
      <td>0.421862</td>
      <td>0.330503</td>
      <td>0.432292</td>
      <td>0.535484</td>
      <td>0.569651</td>
      <td>0.830424</td>
      <td>0.000000</td>
      <td>0.347568</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220316</th>
      <td>0.941721</td>
      <td>0.760911</td>
      <td>0.658337</td>
      <td>0.567759</td>
      <td>0.856330</td>
      <td>0.424601</td>
      <td>0.833522</td>
      <td>0.431852</td>
      <td>0.348184</td>
      <td>0.420139</td>
      <td>0.508504</td>
      <td>0.559369</td>
      <td>0.819672</td>
      <td>0.000000</td>
      <td>0.350270</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220317</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.672945</td>
      <td>0.566682</td>
      <td>0.856151</td>
      <td>0.422882</td>
      <td>0.829373</td>
      <td>0.432808</td>
      <td>0.335781</td>
      <td>0.430845</td>
      <td>0.507038</td>
      <td>0.555632</td>
      <td>0.818191</td>
      <td>0.000000</td>
      <td>0.356757</td>
      <td>0.210119</td>
    </tr>
    <tr>
      <th>220318</th>
      <td>0.944037</td>
      <td>0.759013</td>
      <td>0.650918</td>
      <td>0.556552</td>
      <td>0.864258</td>
      <td>0.422395</td>
      <td>0.830129</td>
      <td>0.429405</td>
      <td>0.334038</td>
      <td>0.429398</td>
      <td>0.511144</td>
      <td>0.556931</td>
      <td>0.819739</td>
      <td>0.000000</td>
      <td>0.366486</td>
      <td>0.212202</td>
    </tr>
    <tr>
      <th>220319</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.654564</td>
      <td>0.560109</td>
      <td>0.875768</td>
      <td>0.420871</td>
      <td>0.831494</td>
      <td>0.433299</td>
      <td>0.338261</td>
      <td>0.428819</td>
      <td>0.503226</td>
      <td>0.549162</td>
      <td>0.823854</td>
      <td>0.000000</td>
      <td>0.375676</td>
      <td>0.212202</td>
    </tr>
  </tbody>
</table>
<p>220320 rows × 16 columns</p>
</div>




```python
pipe2 = Pipeline([
    ('imputer', RandomSampleImputer()),      
    ('scaler', MinMaxScaler()),
    ('constant_features', VarianceThreshold(threshold = 0.01)),
    ('duplicate_features', DropDuplicateFeatures()),
    ('correlated_features', DropCorrelatedFeatures(threshold = 0.8, method = 'pearson'))
])
```


```python
df_main2 = pipe2.fit_transform(sensors_df)
```


```python
df_main2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>4</th>
      <th>5</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>26</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959089</td>
      <td>0.876660</td>
      <td>0.735461</td>
      <td>0.497561</td>
      <td>0.713088</td>
      <td>0.054781</td>
      <td>0.830724</td>
      <td>0.421517</td>
      <td>0.487833</td>
      <td>0.413484</td>
      <td>0.392962</td>
      <td>0.310520</td>
      <td>0.202392</td>
      <td>0.536904</td>
      <td>0.242162</td>
      <td>0.180952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.965264</td>
      <td>0.874763</td>
      <td>0.769891</td>
      <td>0.506916</td>
      <td>0.703827</td>
      <td>0.050643</td>
      <td>0.830513</td>
      <td>0.422027</td>
      <td>0.470622</td>
      <td>0.428819</td>
      <td>0.411437</td>
      <td>0.301123</td>
      <td>0.194447</td>
      <td>0.577701</td>
      <td>0.242162</td>
      <td>0.180357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.959475</td>
      <td>0.876660</td>
      <td>0.765891</td>
      <td>0.518868</td>
      <td>0.710045</td>
      <td>0.053990</td>
      <td>0.828680</td>
      <td>0.425142</td>
      <td>0.480374</td>
      <td>0.394676</td>
      <td>0.417889</td>
      <td>0.296404</td>
      <td>0.195168</td>
      <td>0.581493</td>
      <td>0.246487</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>220315</th>
      <td>0.944423</td>
      <td>0.759013</td>
      <td>0.645910</td>
      <td>0.567240</td>
      <td>0.845650</td>
      <td>0.425340</td>
      <td>0.830620</td>
      <td>0.421862</td>
      <td>0.330503</td>
      <td>0.432292</td>
      <td>0.535484</td>
      <td>0.569651</td>
      <td>0.830424</td>
      <td>0.000000</td>
      <td>0.347568</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220316</th>
      <td>0.941721</td>
      <td>0.760911</td>
      <td>0.658337</td>
      <td>0.567759</td>
      <td>0.856330</td>
      <td>0.424601</td>
      <td>0.833522</td>
      <td>0.431852</td>
      <td>0.348184</td>
      <td>0.420139</td>
      <td>0.508504</td>
      <td>0.559369</td>
      <td>0.819672</td>
      <td>0.000000</td>
      <td>0.350270</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220317</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.672945</td>
      <td>0.566682</td>
      <td>0.856151</td>
      <td>0.422882</td>
      <td>0.829373</td>
      <td>0.432808</td>
      <td>0.335781</td>
      <td>0.430845</td>
      <td>0.507038</td>
      <td>0.555632</td>
      <td>0.818191</td>
      <td>0.000000</td>
      <td>0.356757</td>
      <td>0.210119</td>
    </tr>
    <tr>
      <th>220318</th>
      <td>0.944037</td>
      <td>0.759013</td>
      <td>0.650918</td>
      <td>0.556552</td>
      <td>0.864258</td>
      <td>0.422395</td>
      <td>0.830129</td>
      <td>0.429405</td>
      <td>0.334038</td>
      <td>0.429398</td>
      <td>0.511144</td>
      <td>0.556931</td>
      <td>0.819739</td>
      <td>0.000000</td>
      <td>0.366486</td>
      <td>0.212202</td>
    </tr>
    <tr>
      <th>220319</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.654564</td>
      <td>0.560109</td>
      <td>0.875768</td>
      <td>0.420871</td>
      <td>0.831494</td>
      <td>0.433299</td>
      <td>0.338261</td>
      <td>0.428819</td>
      <td>0.503226</td>
      <td>0.549162</td>
      <td>0.823854</td>
      <td>0.000000</td>
      <td>0.375676</td>
      <td>0.212202</td>
    </tr>
  </tbody>
</table>
<p>220320 rows × 16 columns</p>
</div>



### Preprocessing Pipeline


```python
# Main preprocessor Pipeline
preprocessor_pipe = Pipeline([
    ('imputer', RandomSampleImputer()),      
    ('scaler', MinMaxScaler()),
    ('constant_features', VarianceThreshold(threshold = 0.01)),
    ('duplicate_features', DropDuplicateFeatures()),
    ('correlated_features', DropCorrelatedFeatures(threshold = 0.8, method = 'pearson'))
])
```


```python
df_final = preprocessor_pipe.fit_transform(sensors_df)
```


```python
df_final
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>4</th>
      <th>5</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>26</th>
      <th>28</th>
      <th>29</th>
      <th>30</th>
      <th>31</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.967194</td>
      <td>0.876660</td>
      <td>0.764598</td>
      <td>0.489146</td>
      <td>0.691492</td>
      <td>0.053911</td>
      <td>0.828001</td>
      <td>0.425121</td>
      <td>0.466899</td>
      <td>0.371528</td>
      <td>0.370968</td>
      <td>0.315796</td>
      <td>0.196379</td>
      <td>0.516428</td>
      <td>0.245946</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959089</td>
      <td>0.876660</td>
      <td>0.735461</td>
      <td>0.497561</td>
      <td>0.713088</td>
      <td>0.054781</td>
      <td>0.830724</td>
      <td>0.421517</td>
      <td>0.487833</td>
      <td>0.413484</td>
      <td>0.392962</td>
      <td>0.310520</td>
      <td>0.202392</td>
      <td>0.536904</td>
      <td>0.242162</td>
      <td>0.180952</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.965264</td>
      <td>0.874763</td>
      <td>0.769891</td>
      <td>0.506916</td>
      <td>0.703827</td>
      <td>0.050643</td>
      <td>0.830513</td>
      <td>0.422027</td>
      <td>0.470622</td>
      <td>0.428819</td>
      <td>0.411437</td>
      <td>0.301123</td>
      <td>0.194447</td>
      <td>0.577701</td>
      <td>0.242162</td>
      <td>0.180357</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.959475</td>
      <td>0.876660</td>
      <td>0.765891</td>
      <td>0.518868</td>
      <td>0.710045</td>
      <td>0.053990</td>
      <td>0.828680</td>
      <td>0.425142</td>
      <td>0.480374</td>
      <td>0.394676</td>
      <td>0.417889</td>
      <td>0.296404</td>
      <td>0.195168</td>
      <td>0.581493</td>
      <td>0.246487</td>
      <td>0.178571</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>220315</th>
      <td>0.944423</td>
      <td>0.759013</td>
      <td>0.645910</td>
      <td>0.567240</td>
      <td>0.845650</td>
      <td>0.425340</td>
      <td>0.830620</td>
      <td>0.421862</td>
      <td>0.330503</td>
      <td>0.432292</td>
      <td>0.535484</td>
      <td>0.569651</td>
      <td>0.830424</td>
      <td>0.000000</td>
      <td>0.347568</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220316</th>
      <td>0.941721</td>
      <td>0.760911</td>
      <td>0.658337</td>
      <td>0.567759</td>
      <td>0.856330</td>
      <td>0.424601</td>
      <td>0.833522</td>
      <td>0.431852</td>
      <td>0.348184</td>
      <td>0.420139</td>
      <td>0.508504</td>
      <td>0.559369</td>
      <td>0.819672</td>
      <td>0.000000</td>
      <td>0.350270</td>
      <td>0.209226</td>
    </tr>
    <tr>
      <th>220317</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.672945</td>
      <td>0.566682</td>
      <td>0.856151</td>
      <td>0.422882</td>
      <td>0.829373</td>
      <td>0.432808</td>
      <td>0.335781</td>
      <td>0.430845</td>
      <td>0.507038</td>
      <td>0.555632</td>
      <td>0.818191</td>
      <td>0.000000</td>
      <td>0.356757</td>
      <td>0.210119</td>
    </tr>
    <tr>
      <th>220318</th>
      <td>0.944037</td>
      <td>0.759013</td>
      <td>0.650918</td>
      <td>0.556552</td>
      <td>0.864258</td>
      <td>0.422395</td>
      <td>0.830129</td>
      <td>0.429405</td>
      <td>0.334038</td>
      <td>0.429398</td>
      <td>0.511144</td>
      <td>0.556931</td>
      <td>0.819739</td>
      <td>0.000000</td>
      <td>0.366486</td>
      <td>0.212202</td>
    </tr>
    <tr>
      <th>220319</th>
      <td>0.940178</td>
      <td>0.759013</td>
      <td>0.654564</td>
      <td>0.560109</td>
      <td>0.875768</td>
      <td>0.420871</td>
      <td>0.831494</td>
      <td>0.433299</td>
      <td>0.338261</td>
      <td>0.428819</td>
      <td>0.503226</td>
      <td>0.549162</td>
      <td>0.823854</td>
      <td>0.000000</td>
      <td>0.375676</td>
      <td>0.212202</td>
    </tr>
  </tbody>
</table>
<p>220320 rows × 16 columns</p>
</div>




```python
# Transform pandas DataFrame to numpy array
df_final = df_final.to_numpy()
```


```python
df_final
```




    array([[0.9671944 , 0.87665996, 0.76459842, ..., 0.5164279 , 0.2459459 ,
            0.17857144],
           [0.9671944 , 0.87665996, 0.76459842, ..., 0.5164279 , 0.2459459 ,
            0.17857144],
           [0.95908931, 0.87665996, 0.73546068, ..., 0.53690358, 0.24216205,
            0.18095238],
           ...,
           [0.9401777 , 0.75901303, 0.67294531, ..., 0.        , 0.35675664,
            0.21011906],
           [0.94403723, 0.75901312, 0.65091828, ..., 0.        , 0.3664865 ,
            0.21220233],
           [0.9401777 , 0.75901312, 0.65456419, ..., 0.        , 0.37567572,
            0.21220233]])



### Make Timeseries training pairs


```python
# Function checks whether the machine experienced a pump failure during the specified time period
def check_for_pump_failures(start, stop):
    for minute in range(start, stop):
        machine_state_in_this_minute = data["machine_status"].iloc[minute]

    if machine_state_in_this_minute in ["BROKEN", "RECOVERING"]:
        return 1
    
    return 0
```


```python
# Function to create list of training pairs
def make_training_pairs(df, nr_examples_to_prepare):
    input_minutes_window = 60
    
    output_minutes_window = 60 * 24

    nr_rows_total = df.shape[0]

    max_row_nr = nr_rows_total - input_minutes_window - output_minutes_window
    
    training_pairs = []
    
    for example_nr in range(0, nr_examples_to_prepare):
        
        # Flag to track whether we've found a suitable example
        found_example_where_pump_worked_in_input_window = False
        
        while not found_example_where_pump_worked_in_input_window:
            # Choose a random minute within the dataset
            rnd_minute = np.random.randint(0, max_row_nr)
            
            # Define the start and stop times for the input window
            start = rnd_minute
            stop = start + input_minutes_window
            
            # Check if there were any pump failures within the input window
            if check_for_pump_failures(start, stop) == 0:
                found_example_where_pump_worked_in_input_window = True
        
        # Extract the input window from the sensor data
        input_window = df[rnd_minute: rnd_minute + input_minutes_window]
        
        # Flatten the input window into a single vector
        input_vector = input_window.flatten()
        
        # Define the start and stop times for the output window
        start = rnd_minute + input_minutes_window
        stop = rnd_minute + input_minutes_window + output_minutes_window
        
        output_value = check_for_pump_failures(start, stop)
        
        training_pairs.append((input_vector, output_value))
    
    return training_pairs

# Function to split training pairs into a training set and a test set
def split_training_pairs(training_pairs, nr_examples_to_prepare):
    input_vec_len = training_pairs[0][0].shape[0]
    output_vec_len = 1
    
    D = np.zeros((nr_examples_to_prepare, input_vec_len + output_vec_len))
    
    for nr in range(0, nr_examples_to_prepare):
        (x, y) = training_pairs[nr]
        
        D[nr, 0:input_vec_len] = x
        D[nr, input_vec_len] = y
    
    nr_train_samples = int(nr_examples_to_prepare / 2)

    x_train = D[0:nr_train_samples, 0:input_vec_len]
    y_train = D[0:nr_train_samples, input_vec_len]

    x_test = D[nr_train_samples: , 0:input_vec_len]
    y_test = D[nr_train_samples: , input_vec_len]
    
    return x_train, x_test, y_train, y_test
```


```python
examples_to_prepare = 2000

training_pairs = make_training_pairs(df_final, examples_to_prepare)
```


```python
len(training_pairs)
```




    2000




```python
X_train, X_test, y_train, y_test = split_training_pairs(training_pairs, examples_to_prepare)
```


```python
X_train.shape, X_test.shape
```




    ((1000, 960), (1000, 960))




```python
y_train.shape, y_test.shape
```




    ((1000,), (1000,))



## Model Training using Machine Learning Algorithms


```python
# Function performs a grid search using the specified machine learning model
def model_training(model, param_grid, cv, X_train, y_train, X_test, y_test):
    grid_search = GridSearchCV(model, param_grid, cv = cv, n_jobs = -1, scoring = 'roc_auc')
    grid_search = grid_search.fit(X_train, y_train)
    
    print("{0} evaluation results:".format(type(model).__name__))
    
    models_ml_list.append(type(model).__name__)
    
    score_func_ml(grid_search, X_train, y_train, X_test, y_test)

# Function prints the evaluation scores for the specified grid search model and stores results into lists
def score_func_ml(grid_search_model, X_train, y_train, X_test, y_test):
    pred = grid_search_model.predict(X_test)
    train_score = grid_search_model.score(X_train, y_train)
    test_score = grid_search_model.score(X_test, y_test)
    
    print("Train roc-auc score: {0:.3g}".format(train_score))
    print("Test roc-auc score: {0:.3g}\n".format(test_score))
    print(confusion_matrix(y_test, pred), '\n')
    print("Precision_score: {0:.3g}\n".format(precision_score(y_test, pred)))
    print("Recall: {0:.3g}\n".format(recall_score(y_test, pred)))
    print("Accuracy_score: {0:.3g}\n".format(accuracy_score(y_test, pred)))
    print("F1_score: {0:.3g}\n".format(f1_score(y_test, pred)))
    print(grid_search_model.best_params_)
    print("\n")
    
    roc_auc_score_ml_list.append(round(test_score, 3))
    accuracy_score_ml_list.append(round(accuracy_score(y_test, pred), 3))
    precision_score_ml_list.append(round(precision_score(y_test, pred), 3))
    recall_score_ml_list.append(round(recall_score(y_test, pred), 3))
    f1_score_ml_list.append(round(f1_score(y_test, pred), 3))
    cm_ml_list.append(confusion_matrix(y_test, pred))
```


```python
# Create empty lists to save ML models evaluation results.

models_ml_list = []

roc_auc_score_ml_list = []

accuracy_score_ml_list = []

precision_score_ml_list = []

recall_score_ml_list = []

f1_score_ml_list = []

cm_ml_list = []
```

### Logistic Regression


```python
%%timeit -r 1 -n 1

lr_model = LogisticRegression()

param_grid_lr = {
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'penalty': ['l1', 'l2', 'none', 'elasticnet'],
    'dual': [True, False],
    'max_iter': range(100, 300, 50),
    'C': [0.7, 0.8, 0.9, 1.0],
    'fit_intercept': [True, False]
}

model_training(lr_model, param_grid_lr, 10, X_train, y_train, X_test, y_test)
```

    LogisticRegression evaluation results:
    Train roc-auc score: 0.861
    Test roc-auc score: 0.795
    
    [[948   0]
     [ 49   3]] 
    
    Precision_score: 1
    
    Recall: 0.0577
    
    Accuracy_score: 0.951
    
    F1_score: 0.109
    
    {'C': 0.7, 'dual': False, 'fit_intercept': True, 'max_iter': 200, 'penalty': 'l2', 'solver': 'sag'}
    
    
    3min 7s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    

### KNN Classifier


```python
%%timeit -r 1 -n 1

knn_model = KNeighborsClassifier()

param_grid_knn = {
    'n_neighbors': range(3, 13, 2),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree'],
    'p': [1, 2]
}

model_training(knn_model, param_grid_knn, 10, X_train, y_train, X_test, y_test)
```

    KNeighborsClassifier evaluation results:
    Train roc-auc score: 1
    Test roc-auc score: 0.958
    
    [[947   1]
     [ 48   4]] 
    
    Precision_score: 0.8
    
    Recall: 0.0769
    
    Accuracy_score: 0.951
    
    F1_score: 0.14
    
    {'algorithm': 'ball_tree', 'n_neighbors': 11, 'p': 1, 'weights': 'distance'}
    
    
    6.38 s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    

### Random Forest Classifier


```python
%%timeit -r 1 -n 1

rf_model = RandomForestClassifier()

param_grid_rf = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 3],
        'n_estimators': range(100, 300, 50)
}

model_training(rf_model, param_grid_rf, 10, X_train, y_train, X_test, y_test)
```

    RandomForestClassifier evaluation results:
    Train roc-auc score: 1
    Test roc-auc score: 0.952
    
    [[948   0]
     [ 45   7]] 
    
    Precision_score: 1
    
    Recall: 0.135
    
    Accuracy_score: 0.955
    
    F1_score: 0.237
    
    {'criterion': 'entropy', 'min_samples_leaf': 3, 'min_samples_split': 2, 'n_estimators': 200}
    
    
    1min 14s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    

### Gradient Boosting Classifier


```python
%%timeit -r 1 -n 1

gbc_model = GradientBoostingClassifier()

param_grid_gbc = {
    'loss': ['log_loss', 'deviance', 'exponential'],
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [150, 200, 250],
    'criterion': ['friedman_mse', 'squared_error', 'mse']
}

model_training(gbc_model, param_grid_gbc, 10, X_train, y_train, X_test, y_test)
```

    GradientBoostingClassifier evaluation results:
    Train roc-auc score: 1
    Test roc-auc score: 0.89
    
    [[940   8]
     [ 46   6]] 
    
    Precision_score: 0.429
    
    Recall: 0.115
    
    Accuracy_score: 0.946
    
    F1_score: 0.182
    
    {'criterion': 'squared_error', 'learning_rate': 0.3, 'loss': 'deviance', 'n_estimators': 250}
    
    
    14min 19s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    

### Machine Learning Models Results


```python
# Function to show the top 10 models sorted by the specified metric
def results_plot(metric, df):
    plt.figure(figsize=(12, 5))
    
    df = df.loc[(df[metric] != 0)]
    
    sns.barplot(data = df.sort_values(by = metric, ascending = False).head(10), 
            x = metric, y = 'Model')
    plt.title(metric, fontsize = 12)
    plt.xlabel(metric)
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.show()

# Function to show a heatmap of the confusion matrix
def plot_cm(model, cm):
    f, ax = plt.subplots(figsize=(5,5))
    sns.heatmap(cm, annot = True, linewidths=0.5, linecolor="red", fmt = ".0f", ax = ax)
    plt.title(model)
    plt.xlabel("y_pred")
    plt.ylabel("y_test")
    plt.show()
```


```python
# Create dataframe with model evaluation results

model_ml_results_df = pd.DataFrame({
    'Model': models_ml_list,
    'roc_auc_score': roc_auc_score_ml_list, 
    'accuracy_score': accuracy_score_ml_list, 
    'precision_score': precision_score_ml_list,
    'recall_score': recall_score_ml_list,
    'f1_score': f1_score_ml_list
})
```


```python
model_ml_results_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>roc_auc_score</th>
      <th>accuracy_score</th>
      <th>precision_score</th>
      <th>recall_score</th>
      <th>f1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.795</td>
      <td>0.951</td>
      <td>1.000</td>
      <td>0.058</td>
      <td>0.109</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KNeighborsClassifier</td>
      <td>0.958</td>
      <td>0.951</td>
      <td>0.800</td>
      <td>0.077</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>2</th>
      <td>RandomForestClassifier</td>
      <td>0.952</td>
      <td>0.955</td>
      <td>1.000</td>
      <td>0.135</td>
      <td>0.237</td>
    </tr>
    <tr>
      <th>3</th>
      <td>GradientBoostingClassifier</td>
      <td>0.890</td>
      <td>0.946</td>
      <td>0.429</td>
      <td>0.115</td>
      <td>0.182</td>
    </tr>
  </tbody>
</table>
</div>




```python
results_plot("roc_auc_score", model_ml_results_df)
```


    
![png](README_files/README_79_0.png)
    



```python
results_plot('recall_score', model_ml_results_df)
```


    
![png](README_files/README_80_0.png)
    



```python
# Show confusion matrix for each model

for model, cm in zip(models_ml_list, cm_ml_list):
    plot_cm(model, cm)
```


    
![png](README_files/README_81_0.png)
    



    
![png](README_files/README_81_1.png)
    



    
![png](README_files/README_81_2.png)
    



    
![png](README_files/README_81_3.png)
    


## Model Training using Deep Learning (ANN)


```python
# Function plots the training and validation loss and accuracy for each epoch of training and saves the plot as an image file
def show_final_history(epochs_history, model_name):
    fig, ax = plt.subplots(1, 2, figsize = (18, 5))
    
    ax[0].set_title('Loss')
    ax[0].plot(epochs_history.epoch, epochs_history.history['loss'], label = 'Train loss')
    ax[0].plot(epochs_history.epoch, epochs_history.history['val_loss'], label = 'Validation loss')

    ax[1].set_title('Accuracy')
    ax[1].plot(epochs_history.epoch, epochs_history.history['accuracy'], label = 'Train accuracy')
    ax[1].plot(epochs_history.epoch, epochs_history.history['val_accuracy'], label = 'Validation accuracy')
    ax[0].legend()
    ax[1].legend()
    plt.savefig('epochs_plot_histories/{}_plot_history.png'.format(model_name))
```


```python
# Function to get model evaluation results and store it to list

def evaluate_model(model, model_name, X_test, y_test):
    pred = model.predict(X_test, verbose = 0)
    pred = np.where(pred < 0.5, 0, 1)
    
    models_ann_list.append(model_name)
    
    model.save('keras_models/{}.h5'.format(model_name))
    
    roc_auc_score_ann_list.append(round(roc_auc_score(y_test, pred), 3))
    accuracy_score_ann_list.append(round(accuracy_score(y_test, pred), 3))
    precision_score_ann_list.append(round(precision_score(y_test, pred), 3))
    recall_score_ann_list.append(round(recall_score(y_test, pred), 3))
    f1_score_ann_list.append(round(f1_score(y_test, pred), 3))
    cm_ann_list.append(confusion_matrix(y_test, pred))
```


```python
models_ann_list = []

roc_auc_score_ann_list = []

accuracy_score_ann_list = []

precision_score_ann_list = []

recall_score_ann_list = []

f1_score_ann_list = []

cm_ann_list = []
```

### Hyperband


```python
# Model builder functions for keras tuner

def build_model_adam(hp):
    input_vec = 960
    hp_units_1 = hp.Int("units_1", min_value = 100, max_value = 300, step = 20)
    hp_units_2 = hp.Int("units_2", min_value = 20, max_value = 60, step = 10)
    opt = "adam"
    
    model = Sequential()
    model.add(
        Dense(
            units = hp_units_1,
            activation = "relu",
            input_shape = (input_vec, )
        )
    )
    if hp.Boolean("dropout1"):
        model.add(Dropout(rate = 0.2))
        
    model.add(Dense(
        units = hp_units_2,
        activation = "relu"))
    
    if hp.Boolean("dropout2"):
        model.add(Dropout(rate = 0.1))
    
    model.add(Dense(1))
    model.compile(
        optimizer = opt,
        loss = "mean_squared_error",
        metrics = ["accuracy"],
    )
    return model

def build_model_sgd(hp):
    input_vec = 960
    hp_units_1 = hp.Int("units_1", min_value = 100, max_value = 300, step = 20)
    hp_units_2 = hp.Int("units_2", min_value = 20, max_value = 60, step = 10)
    opt = "sgd"
    
    model = Sequential()
    model.add(
        Dense(
            units = hp_units_1,
            activation = "relu",
            input_shape = (input_vec, )
        )
    )
    if hp.Boolean("dropout1"):
        model.add(Dropout(rate = 0.2))
        
    model.add(Dense(
        units = hp_units_2,
        activation = "relu"))
    
    if hp.Boolean("dropout2"):
        model.add(Dropout(rate = 0.1))
    
    model.add(Dense(1))
    model.compile(
        optimizer = opt,
        loss = "mean_squared_error",
        metrics = ["accuracy"],
    )
    return model

def build_model_rmsprop(hp):
    input_vec = 960
    hp_units_1 = hp.Int("units_1", min_value = 100, max_value = 300, step = 20)
    hp_units_2 = hp.Int("units_2", min_value = 20, max_value = 60, step = 10)
    opt = "rmsprop"
    
    model = Sequential()
    model.add(
        Dense(
            units = hp_units_1,
            activation = "relu",
            input_shape = (input_vec, )
        )
    )
    if hp.Boolean("dropout1"):
        model.add(Dropout(rate = 0.2))
        
    model.add(Dense(
        units = hp_units_2,
        activation = "relu"))
    
    if hp.Boolean("dropout2"):
        model.add(Dropout(rate = 0.1))
    
    model.add(Dense(1))
    model.compile(
        optimizer = opt,
        loss = "mean_squared_error",
        metrics = ["accuracy"],
    )
    return model

def build_model_adadelta(hp):
    input_vec = 960
    hp_units_1 = hp.Int("units_1", min_value = 100, max_value = 300, step = 20)
    hp_units_2 = hp.Int("units_2", min_value = 20, max_value = 60, step = 10)
    opt = "adadelta"
    
    model = Sequential()
    model.add(
        Dense(
            units = hp_units_1,
            activation = "relu",
            input_shape = (input_vec, )
        )
    )
    if hp.Boolean("dropout1"):
        model.add(Dropout(rate = 0.2))
        
    model.add(Dense(
        units = hp_units_2,
        activation = "relu"))
    
    if hp.Boolean("dropout2"):
        model.add(Dropout(rate = 0.1))
    
    model.add(Dense(1))
    model.compile(
        optimizer = opt,
        loss = "mean_squared_error",
        metrics = ["accuracy"],
    )
    return model
```


```python
# Create list with model builders

model_builders_list = [ build_model_adam, build_model_sgd, build_model_rmsprop, build_model_adadelta]
```


```python
%%timeit -r 1 -n 1

for model_builder in model_builders_list:
    tuner = Hyperband(
        hypermodel = model_builder,
        objective = 'val_loss',
        max_epochs = 50,
        directory = 'hyperband_{}'.format(model_builders_list.index(model_builder)),
        project_name = 'water_pump_failure_prediction'
    )
    
    tuner.search(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))
    
    batch_size = range(8, 49, 8)

    for i in batch_size:
        best_model = tuner.get_best_models(num_models = 3)[0]

        best_model.build(input_shape = (training_pairs[0][0].shape[0], ))
        
        epochs_history = best_model.fit(
            X_train,
            y_train,
            validation_data = (X_test, y_test),
            batch_size = i,
            epochs = 100,
            verbose = 0
        )

        show_final_history(epochs_history, "model_hyperband_{0}_batch_size_{1}".format(model_builders_list.index(model_builder), i))

        evaluate_model(best_model, "model_hyperband_{0}_batch_size_{1}".format(model_builders_list.index(model_builder), i), X_test, y_test)
    
```

    Trial 90 Complete [00h 00m 13s]
    val_loss: 0.05583332106471062
    
    Best val_loss So Far: 0.04960395395755768
    Total elapsed time: 00h 05m 50s
    INFO:tensorflow:Oracle triggered exit
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    32min 44s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    


    
![png](README_files/README_89_1.png)
    



    
![png](README_files/README_89_2.png)
    



    
![png](README_files/README_89_3.png)
    



    
![png](README_files/README_89_4.png)
    



    
![png](README_files/README_89_5.png)
    



    
![png](README_files/README_89_6.png)
    



    
![png](README_files/README_89_7.png)
    



    
![png](README_files/README_89_8.png)
    



    
![png](README_files/README_89_9.png)
    



    
![png](README_files/README_89_10.png)
    



    
![png](README_files/README_89_11.png)
    



    
![png](README_files/README_89_12.png)
    



    
![png](README_files/README_89_13.png)
    



    
![png](README_files/README_89_14.png)
    



    
![png](README_files/README_89_15.png)
    



    
![png](README_files/README_89_16.png)
    



    
![png](README_files/README_89_17.png)
    



    
![png](README_files/README_89_18.png)
    



    
![png](README_files/README_89_19.png)
    



    
![png](README_files/README_89_20.png)
    



    
![png](README_files/README_89_21.png)
    



    
![png](README_files/README_89_22.png)
    



    
![png](README_files/README_89_23.png)
    



    
![png](README_files/README_89_24.png)
    



```python
model_results_df_ann = pd.DataFrame({
    'Model': models_ann_list,
    'roc_auc_score': roc_auc_score_ann_list, 
    'accuracy_score': accuracy_score_ann_list, 
    'precision_score': precision_score_ann_list,
    'recall_score': recall_score_ann_list,
    'f1_score': f1_score_ann_list
})

model_results_df_ann
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>roc_auc_score</th>
      <th>accuracy_score</th>
      <th>precision_score</th>
      <th>recall_score</th>
      <th>f1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>model_hyperband_0_batch_size_8</td>
      <td>0.596</td>
      <td>0.958</td>
      <td>1.000</td>
      <td>0.192</td>
      <td>0.323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>model_hyperband_0_batch_size_16</td>
      <td>0.528</td>
      <td>0.950</td>
      <td>0.750</td>
      <td>0.058</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>model_hyperband_0_batch_size_24</td>
      <td>0.717</td>
      <td>0.963</td>
      <td>0.742</td>
      <td>0.442</td>
      <td>0.554</td>
    </tr>
    <tr>
      <th>3</th>
      <td>model_hyperband_0_batch_size_32</td>
      <td>0.538</td>
      <td>0.951</td>
      <td>0.800</td>
      <td>0.077</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>4</th>
      <td>model_hyperband_0_batch_size_40</td>
      <td>0.556</td>
      <td>0.951</td>
      <td>0.667</td>
      <td>0.115</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>5</th>
      <td>model_hyperband_0_batch_size_48</td>
      <td>0.576</td>
      <td>0.955</td>
      <td>0.889</td>
      <td>0.154</td>
      <td>0.262</td>
    </tr>
    <tr>
      <th>6</th>
      <td>model_hyperband_1_batch_size_8</td>
      <td>0.537</td>
      <td>0.950</td>
      <td>0.667</td>
      <td>0.077</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>7</th>
      <td>model_hyperband_1_batch_size_16</td>
      <td>0.519</td>
      <td>0.950</td>
      <td>1.000</td>
      <td>0.038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>8</th>
      <td>model_hyperband_1_batch_size_24</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>9</th>
      <td>model_hyperband_1_batch_size_32</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>10</th>
      <td>model_hyperband_1_batch_size_40</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>11</th>
      <td>model_hyperband_1_batch_size_48</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>12</th>
      <td>model_hyperband_2_batch_size_8</td>
      <td>0.710</td>
      <td>0.967</td>
      <td>0.880</td>
      <td>0.423</td>
      <td>0.571</td>
    </tr>
    <tr>
      <th>13</th>
      <td>model_hyperband_2_batch_size_16</td>
      <td>0.812</td>
      <td>0.971</td>
      <td>0.767</td>
      <td>0.635</td>
      <td>0.695</td>
    </tr>
    <tr>
      <th>14</th>
      <td>model_hyperband_2_batch_size_24</td>
      <td>0.698</td>
      <td>0.962</td>
      <td>0.750</td>
      <td>0.404</td>
      <td>0.525</td>
    </tr>
    <tr>
      <th>15</th>
      <td>model_hyperband_2_batch_size_32</td>
      <td>0.625</td>
      <td>0.961</td>
      <td>1.000</td>
      <td>0.250</td>
      <td>0.400</td>
    </tr>
    <tr>
      <th>16</th>
      <td>model_hyperband_2_batch_size_40</td>
      <td>0.673</td>
      <td>0.966</td>
      <td>1.000</td>
      <td>0.346</td>
      <td>0.514</td>
    </tr>
    <tr>
      <th>17</th>
      <td>model_hyperband_2_batch_size_48</td>
      <td>0.606</td>
      <td>0.959</td>
      <td>1.000</td>
      <td>0.212</td>
      <td>0.349</td>
    </tr>
    <tr>
      <th>18</th>
      <td>model_hyperband_3_batch_size_8</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>model_hyperband_3_batch_size_16</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>model_hyperband_3_batch_size_24</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>model_hyperband_3_batch_size_32</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>model_hyperband_3_batch_size_40</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>model_hyperband_3_batch_size_48</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
for model, cm in zip(models_ann_list, cm_ann_list):
    plot_cm(model, cm)
```


    
![png](README_files/README_91_0.png)
    



    
![png](README_files/README_91_1.png)
    



    
![png](README_files/README_91_2.png)
    



    
![png](README_files/README_91_3.png)
    



    
![png](README_files/README_91_4.png)
    



    
![png](README_files/README_91_5.png)
    



    
![png](README_files/README_91_6.png)
    



    
![png](README_files/README_91_7.png)
    



    
![png](README_files/README_91_8.png)
    



    
![png](README_files/README_91_9.png)
    



    
![png](README_files/README_91_10.png)
    



    
![png](README_files/README_91_11.png)
    



    
![png](README_files/README_91_12.png)
    



    
![png](README_files/README_91_13.png)
    



    
![png](README_files/README_91_14.png)
    



    
![png](README_files/README_91_15.png)
    



    
![png](README_files/README_91_16.png)
    



    
![png](README_files/README_91_17.png)
    



    
![png](README_files/README_91_18.png)
    



    
![png](README_files/README_91_19.png)
    



    
![png](README_files/README_91_20.png)
    



    
![png](README_files/README_91_21.png)
    



    
![png](README_files/README_91_22.png)
    



    
![png](README_files/README_91_23.png)
    



```python
results_plot("recall_score", model_results_df_ann)
```


    
![png](README_files/README_92_0.png)
    


### Random Search


```python
%%timeit -r 1 -n 1

for model_builder in model_builders_list:
    tuner = RandomSearch(
        hypermodel = model_builder,
        objective = "val_loss",
        directory = "random_search_{}".format(model_builders_list.index(model_builder)),
        project_name = "water_pump_failure_prediction"
    )

    tuner.search(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))
    
    batch_size = range(8, 49, 8)

    for i in batch_size:
        best_model = tuner.get_best_models(num_models = 3)[0]

        best_model.build(input_shape = (training_pairs[0][0].shape[0], ))
        
        epochs_history = best_model.fit(
            X_train,
            y_train,
            validation_data = (X_test, y_test),
            batch_size = i,
            epochs = 100,
            verbose = 0
        )

        show_final_history(epochs_history, "model_random_search_{0}_batch_size_{1}".format(model_builders_list.index(model_builder), i))

        evaluate_model(best_model, "model_random_search_{0}_batch_size_{1}".format(model_builders_list.index(model_builder), i), X_test, y_test)
    
```

    Trial 10 Complete [00h 00m 26s]
    val_loss: 0.04688766598701477
    
    Best val_loss So Far: 0.04688766598701477
    Total elapsed time: 00h 04m 05s
    INFO:tensorflow:Oracle triggered exit
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    WARNING:tensorflow:Detecting that an object or model or tf.train.Checkpoint is being deleted with unrestored values. See the following logs for the specific values in question. To silence these warnings, use `status.expect_partial()`. See https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#restorefor details about the status object returned by the restore function.
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.iter
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.decay
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.learning_rate
    WARNING:tensorflow:Value in checkpoint could not be found in the restored object: (root).optimizer.rho
    26min 28s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
    


    
![png](README_files/README_94_1.png)
    



    
![png](README_files/README_94_2.png)
    



    
![png](README_files/README_94_3.png)
    



    
![png](README_files/README_94_4.png)
    



    
![png](README_files/README_94_5.png)
    



    
![png](README_files/README_94_6.png)
    



    
![png](README_files/README_94_7.png)
    



    
![png](README_files/README_94_8.png)
    



    
![png](README_files/README_94_9.png)
    



    
![png](README_files/README_94_10.png)
    



    
![png](README_files/README_94_11.png)
    



    
![png](README_files/README_94_12.png)
    



    
![png](README_files/README_94_13.png)
    



    
![png](README_files/README_94_14.png)
    



    
![png](README_files/README_94_15.png)
    



    
![png](README_files/README_94_16.png)
    



    
![png](README_files/README_94_17.png)
    



    
![png](README_files/README_94_18.png)
    



    
![png](README_files/README_94_19.png)
    



    
![png](README_files/README_94_20.png)
    



    
![png](README_files/README_94_21.png)
    



    
![png](README_files/README_94_22.png)
    



    
![png](README_files/README_94_23.png)
    



    
![png](README_files/README_94_24.png)
    



```python
model_results_df_ann = pd.DataFrame({
    'Model': models_ann_list,
    'roc_auc_score': roc_auc_score_ann_list, 
    'accuracy_score': accuracy_score_ann_list, 
    'precision_score': precision_score_ann_list,
    'recall_score': recall_score_ann_list,
    'f1_score': f1_score_ann_list
})

model_results_df_ann
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>roc_auc_score</th>
      <th>accuracy_score</th>
      <th>precision_score</th>
      <th>recall_score</th>
      <th>f1_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>model_hyperband_0_batch_size_8</td>
      <td>0.596</td>
      <td>0.958</td>
      <td>1.000</td>
      <td>0.192</td>
      <td>0.323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>model_hyperband_0_batch_size_16</td>
      <td>0.528</td>
      <td>0.950</td>
      <td>0.750</td>
      <td>0.058</td>
      <td>0.107</td>
    </tr>
    <tr>
      <th>2</th>
      <td>model_hyperband_0_batch_size_24</td>
      <td>0.717</td>
      <td>0.963</td>
      <td>0.742</td>
      <td>0.442</td>
      <td>0.554</td>
    </tr>
    <tr>
      <th>3</th>
      <td>model_hyperband_0_batch_size_32</td>
      <td>0.538</td>
      <td>0.951</td>
      <td>0.800</td>
      <td>0.077</td>
      <td>0.140</td>
    </tr>
    <tr>
      <th>4</th>
      <td>model_hyperband_0_batch_size_40</td>
      <td>0.556</td>
      <td>0.951</td>
      <td>0.667</td>
      <td>0.115</td>
      <td>0.197</td>
    </tr>
    <tr>
      <th>5</th>
      <td>model_hyperband_0_batch_size_48</td>
      <td>0.576</td>
      <td>0.955</td>
      <td>0.889</td>
      <td>0.154</td>
      <td>0.262</td>
    </tr>
    <tr>
      <th>6</th>
      <td>model_hyperband_1_batch_size_8</td>
      <td>0.537</td>
      <td>0.950</td>
      <td>0.667</td>
      <td>0.077</td>
      <td>0.138</td>
    </tr>
    <tr>
      <th>7</th>
      <td>model_hyperband_1_batch_size_16</td>
      <td>0.519</td>
      <td>0.950</td>
      <td>1.000</td>
      <td>0.038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>8</th>
      <td>model_hyperband_1_batch_size_24</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>9</th>
      <td>model_hyperband_1_batch_size_32</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>10</th>
      <td>model_hyperband_1_batch_size_40</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>11</th>
      <td>model_hyperband_1_batch_size_48</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>12</th>
      <td>model_hyperband_2_batch_size_8</td>
      <td>0.710</td>
      <td>0.967</td>
      <td>0.880</td>
      <td>0.423</td>
      <td>0.571</td>
    </tr>
    <tr>
      <th>13</th>
      <td>model_hyperband_2_batch_size_16</td>
      <td>0.812</td>
      <td>0.971</td>
      <td>0.767</td>
      <td>0.635</td>
      <td>0.695</td>
    </tr>
    <tr>
      <th>14</th>
      <td>model_hyperband_2_batch_size_24</td>
      <td>0.698</td>
      <td>0.962</td>
      <td>0.750</td>
      <td>0.404</td>
      <td>0.525</td>
    </tr>
    <tr>
      <th>15</th>
      <td>model_hyperband_2_batch_size_32</td>
      <td>0.625</td>
      <td>0.961</td>
      <td>1.000</td>
      <td>0.250</td>
      <td>0.400</td>
    </tr>
    <tr>
      <th>16</th>
      <td>model_hyperband_2_batch_size_40</td>
      <td>0.673</td>
      <td>0.966</td>
      <td>1.000</td>
      <td>0.346</td>
      <td>0.514</td>
    </tr>
    <tr>
      <th>17</th>
      <td>model_hyperband_2_batch_size_48</td>
      <td>0.606</td>
      <td>0.959</td>
      <td>1.000</td>
      <td>0.212</td>
      <td>0.349</td>
    </tr>
    <tr>
      <th>18</th>
      <td>model_hyperband_3_batch_size_8</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>model_hyperband_3_batch_size_16</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>model_hyperband_3_batch_size_24</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>model_hyperband_3_batch_size_32</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>model_hyperband_3_batch_size_40</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>model_hyperband_3_batch_size_48</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>model_random_search_0_batch_size_8</td>
      <td>0.662</td>
      <td>0.962</td>
      <td>0.850</td>
      <td>0.327</td>
      <td>0.472</td>
    </tr>
    <tr>
      <th>25</th>
      <td>model_random_search_0_batch_size_16</td>
      <td>0.566</td>
      <td>0.953</td>
      <td>0.778</td>
      <td>0.135</td>
      <td>0.230</td>
    </tr>
    <tr>
      <th>26</th>
      <td>model_random_search_0_batch_size_24</td>
      <td>0.718</td>
      <td>0.965</td>
      <td>0.793</td>
      <td>0.442</td>
      <td>0.568</td>
    </tr>
    <tr>
      <th>27</th>
      <td>model_random_search_0_batch_size_32</td>
      <td>0.652</td>
      <td>0.960</td>
      <td>0.800</td>
      <td>0.308</td>
      <td>0.444</td>
    </tr>
    <tr>
      <th>28</th>
      <td>model_random_search_0_batch_size_40</td>
      <td>0.717</td>
      <td>0.963</td>
      <td>0.742</td>
      <td>0.442</td>
      <td>0.554</td>
    </tr>
    <tr>
      <th>29</th>
      <td>model_random_search_0_batch_size_48</td>
      <td>0.708</td>
      <td>0.963</td>
      <td>0.759</td>
      <td>0.423</td>
      <td>0.543</td>
    </tr>
    <tr>
      <th>30</th>
      <td>model_random_search_1_batch_size_8</td>
      <td>0.548</td>
      <td>0.952</td>
      <td>0.833</td>
      <td>0.096</td>
      <td>0.172</td>
    </tr>
    <tr>
      <th>31</th>
      <td>model_random_search_1_batch_size_16</td>
      <td>0.519</td>
      <td>0.950</td>
      <td>1.000</td>
      <td>0.038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>32</th>
      <td>model_random_search_1_batch_size_24</td>
      <td>0.519</td>
      <td>0.950</td>
      <td>1.000</td>
      <td>0.038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>33</th>
      <td>model_random_search_1_batch_size_32</td>
      <td>0.519</td>
      <td>0.950</td>
      <td>1.000</td>
      <td>0.038</td>
      <td>0.074</td>
    </tr>
    <tr>
      <th>34</th>
      <td>model_random_search_1_batch_size_40</td>
      <td>0.510</td>
      <td>0.949</td>
      <td>1.000</td>
      <td>0.019</td>
      <td>0.038</td>
    </tr>
    <tr>
      <th>35</th>
      <td>model_random_search_1_batch_size_48</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>model_random_search_2_batch_size_8</td>
      <td>0.576</td>
      <td>0.955</td>
      <td>0.889</td>
      <td>0.154</td>
      <td>0.262</td>
    </tr>
    <tr>
      <th>37</th>
      <td>model_random_search_2_batch_size_16</td>
      <td>0.577</td>
      <td>0.956</td>
      <td>1.000</td>
      <td>0.154</td>
      <td>0.267</td>
    </tr>
    <tr>
      <th>38</th>
      <td>model_random_search_2_batch_size_24</td>
      <td>0.710</td>
      <td>0.967</td>
      <td>0.880</td>
      <td>0.423</td>
      <td>0.571</td>
    </tr>
    <tr>
      <th>39</th>
      <td>model_random_search_2_batch_size_32</td>
      <td>0.702</td>
      <td>0.969</td>
      <td>1.000</td>
      <td>0.404</td>
      <td>0.575</td>
    </tr>
    <tr>
      <th>40</th>
      <td>model_random_search_2_batch_size_40</td>
      <td>0.596</td>
      <td>0.957</td>
      <td>0.909</td>
      <td>0.192</td>
      <td>0.317</td>
    </tr>
    <tr>
      <th>41</th>
      <td>model_random_search_2_batch_size_48</td>
      <td>0.729</td>
      <td>0.969</td>
      <td>0.889</td>
      <td>0.462</td>
      <td>0.608</td>
    </tr>
    <tr>
      <th>42</th>
      <td>model_random_search_3_batch_size_8</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>43</th>
      <td>model_random_search_3_batch_size_16</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>model_random_search_3_batch_size_24</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>45</th>
      <td>model_random_search_3_batch_size_32</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>46</th>
      <td>model_random_search_3_batch_size_40</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>47</th>
      <td>model_random_search_3_batch_size_48</td>
      <td>0.500</td>
      <td>0.948</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We use recall score as a main metric, because data is imbalanced. We are not able 
results_plot("recall_score", model_results_df_ann)
```


    
![png](README_files/README_96_0.png)
    


## Results


```python
for model, cm in zip(models_ann_list, cm_ann_list):
    if (model == "model_hyperband_2_batch_size_16"):
        plot_cm(model, cm)
```


    
![png](README_files/README_98_0.png)
    


We trained 48 keras models with different optimizer and different batch size.
As you can see, the best model was trained with hyperband, rmsprop optimizer and 16 batch size.
All trained keras models are saved in a folder.


```python

```
