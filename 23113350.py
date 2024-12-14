import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
# Read the dataset
df = pd.read_csv('student-por.csv')

# Check for any missing values
# print(df.isnull().sum())

#drop rows with missing values:
# df = df.dropna()
# print(df.head())

# Bar chart for gender distribution
df['gender'].value_counts().plot(kind='bar')
plt.title('gender Distribution')
plt.xlabel('gender')
plt.ylabel('gender')
plt.show()


numeric_df = df.select_dtypes(include=['number'])
# Create the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# numerical columns for clustering
numerical_cols = ['age', 'traveltime', 'studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
X = df[numerical_cols]

# Scaling the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method for optimal number of clusters
inertia = []
sil_scores = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    if k > 1:
        sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Create subplots: one for the elbow plot and one for the silhouette scores
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Elbow plot
ax[0].plot(range(1, 11), inertia, marker='o', color='blue')
ax[0].set_title('Elbow Method')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Inertia')

# Silhouette scores plot
ax[1].plot(range(2, 11), sil_scores, marker='o', color='green')
ax[1].set_title('Silhouette Scores')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Silhouette Score')

#avoid overlapping labels
plt.tight_layout()
plt.show()
# Use these columns for clustering and regression
numerical_cols = ['G1', 'G3']  
X = df[numerical_cols]
# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Fit K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)
# Line fitting  between G1 and G3
X = df[['G1']]
y = df['G3']

model = LinearRegression()
model.fit(X, y)

# Predicting G3 based on G1
y_pred = model.predict(X)

# Plotting both K-means clusters and linear regression line
plt.figure(figsize=(10, 6))

# Scatter plot for K-means clusters
plt.scatter(df['G1'], df['G3'], c=df['cluster'], cmap='viridis', label='Clusters', alpha=0.6)

# Plot the regression line
plt.plot(df['G1'], y_pred, color='red', label='Regression Line')

# Labels and title
plt.title('K-means Clustering with Line Fit (G1 vs. G3)')
plt.xlabel('G1 (First Period Grade)')
plt.ylabel('G3 (Final Grade)')
plt.legend()
plt.show()