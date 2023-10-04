import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
data = pd.read_csv('migration_nz.csv')
data.head(10)
data['Measure'].unique()

data['Measure'].replace("Arrivals",0,inplace=True)
data['Measure'].replace("Departures",1,inplace=True)
data['Measure'].replace("Net",2,inplace=True)

data['Measure'].unique()

data['Country'].unique()

data['CountryID'] = pd.factorize(data.Country)[0]
data['CitID'] = pd.factorize(data.Citizenship)[0]

data['CountryID'].unique()

data.isnull().sum()

data["Value"].fillna(data["Value"].median(),inplace=True)

data.isnull().sum()

data.drop('Country', axis=1, inplace=True)
data.drop('Citizenship', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X= data[['CountryID','Measure','Year','CitID']].values
Y= data['Value'].values
X_train, X_test, y_train, y_test = train_test_split(
  X, Y, test_size=0.3, random_state=9)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=70,max_features = 3,max_depth=5,n_jobs=-1)
rf.fit(X_train ,y_train)
rf.score(X_test, y_test)

X = data[['CountryID', 'Measure', 'Year', 'CitID']]
Y = data['Value']
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, random_state=9)
grouped = data.groupby(['Year']).aggregate({'Value': 'sum'})
grouped.plot(kind='line')
plt.axhline(0, color='g')
plt.show()  # Display the line plot

grouped.plot(kind='bar')
plt.axhline(0, color='g')
plt.show()  # Display the bar plot

corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()  # Display the heatmap

# Filter data for different values (e.g., Arrivals, Departures, Net)
arrivals_data = data[data['Measure'] == 0]  # Assuming 0 represents Arrivals
departures_data = data[data['Measure'] == 1]  # Assuming 1 represents Departures
net_data = data[data['Measure'] == 2]  # Assuming 2 represents Net

# Group data by year and sum the values for each measure
arrivals_grouped = arrivals_data.groupby('Year')['Value'].sum()
departures_grouped = departures_data.groupby('Year')['Value'].sum()
net_grouped = net_data.groupby('Year')['Value'].sum()

# Create a stacked bar chart
plt.figure(figsize=(10, 6))

plt.bar(arrivals_grouped.index, arrivals_grouped, label='Arrivals', color='blue')
plt.bar(departures_grouped.index, departures_grouped, label='Departures', color='green', bottom=arrivals_grouped)
plt.bar(net_grouped.index, net_grouped, label='Net', color='red', bottom=arrivals_grouped + departures_grouped)

plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Migration Trends Over Years')
plt.legend()
plt.show()
