import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
# Create a K-Means clustering model


# Fit the model to your data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv('migration_nz.csv')

# Handle missing values by replacing with the median
data["Value"].fillna(data["Value"].median(), inplace=True)

# Encoding categorical variables for 'Measure' column
data['Measure'] = data['Measure'].map({"Arrivals": 0, "Departures": 1, "Net": 2})

# Factorize the 'Country' and 'Citizenship' columns for numerical representation
data['CountryID'] = pd.factorize(data.Country)[0]
data['CitID'] = pd.factorize(data.Citizenship)[0]

# Drop unnecessary columns
data.drop(['Country', 'Citizenship'], axis=1, inplace=True)

# Split the data into features and target
X = data[['CountryID', 'Measure', 'Year', 'CitID']]
Y = data['Value']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=9)

# Train a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=70, max_features=3, max_depth=5, n_jobs=-1)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

# Data visualization
grouped = data.groupby('Year').agg({'Value': 'sum'})

# Plotting line chart for aggregated values over the years
plt.figure(figsize=(10, 6))
plt.subplot(131)
grouped['Value'].plot(kind='line')
plt.axhline(0, color='g')
plt.title('Line Plot')

# Plotting bar chart for aggregated values over the years
plt.subplot(132)
grouped['Value'].plot(kind='bar')
plt.axhline(0, color='g')
plt.title('Bar Plot')

# Calculating the correlation matrix and displaying a heatmap
plt.subplot(133)
corr = data.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# Filter data for different values (e.g., Arrivals, Departures, Net)
arrivals_data = data[data['Measure'] == 0]  # Assuming 0 represents Arrivals
departures_data = data[data['Measure'] == 1]  # Assuming 1 represents Departures
net_data = data[data['Measure'] == 2]  # Assuming 2 represents Net

# Group data by year and sum the values for each measure
arrivals_grouped = arrivals_data.groupby('Year')['Value'].sum()
departures_grouped = departures_data.groupby('Year')['Value'].sum()
net_grouped = net_data.groupby('Year')['Value'].sum()

# Create a stacked bar chart to visualize migration trends over the years
plt.figure(figsize=(10, 6))
plt.bar(arrivals_grouped.index, arrivals_grouped, label='Arrivals', color='blue')
plt.bar(departures_grouped.index, departures_grouped, label='Departures', color='green', bottom=arrivals_grouped)
plt.bar(net_grouped.index, net_grouped, label='Net', color='red', bottom=arrivals_grouped + departures_grouped)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Migration Trends Over Years')
plt.legend()
plt.show()




# Train a Support Vector Machine (SVM) Regressor
svm = SVR()
#svm.fit(X_train, y_train)
#svm_score = svm.score(X_test, y_test)

# Train a k-Nearest Neighbors (KNN) Regressor
knn = KNeighborsRegressor(n_neighbors=5)  # You can adjust the number of neighbors as needed
knn.fit(X_train, y_train)
knn_score = knn.score(X_test, y_test)

# Print the KNN Regressor score
print(f"KNN Regressor Score: {knn_score}")

# Calculate MAE for each model
mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
#mae_svm = mean_absolute_error(y_test, svm.predict(X_test))
mae_knn = mean_absolute_error(y_test, knn.predict(X_test))

# Print the MAE for each model
print(f"Random Forest MAE: {mae_rf}")
#print(f"SVM MAE: {mae_svm}")
print(f"KNN MAE: {mae_knn}")

kmeans = KMeans(n_clusters=3, random_state=9)  # You can adjust the number of clusters as needed

# Standardize the features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=3, random_state=9)  # You can adjust the number of clusters as needed

# Fit the model to your standardized data
data['Cluster'] = kmeans.fit_predict(X_std)

plt.scatter(X_std[:, 0], X_std[:, 1], c=data['Cluster'], cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
