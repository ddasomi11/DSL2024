import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster

def get_data_frames(filename, countries, indicator):
    """
This function returns a data frame with countries as columns and years as rows, 
and a data frame with years as columns and countries as rows
    """
    
    # Use Pandas to read data into a dataframe
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # Filter data by country
    df = df.loc[df['Country Name'].isin(countries)]
    # Filter data by indicator code
    df = df.loc[df['Indicator Code'] == indicator]


    # Reshape the dataframe by converting all year columns into a single column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name', 'Indicator Code'], var_name='Years')
   
    # Remove the country code column as it's not required for further analysis
    del df2['Country Code']
    
    # Convert countries from rows to separate columns using pivot_table function
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code'], 'Country Name').reset_index()
    
    df_countries = df
    df_years = df2

    # Clean data by dropping NaN values
    df_countries.dropna(inplace=True)
    df_years.dropna(inplace=True)

    return df_countries, df_years


# Setting filename
filename = "Climate.csv"
indicators = ['SP.POP.GROW', 'SP.URB.GROW']
df_y, df_i = get_data_frames1(filename, indicators)

df_i = df_i.loc[df_i['Years'].eq('2019')]
# XKX' and 'MAF' countries are excluded due to missing values
df_i = df_i.loc[~df_i['Country Code'].isin(['XKX', 'MAF'])]

# prepare columns for fitting
df_fit = df_i[["SP.POP.GROW", "SP.URB.GROW"]].copy()
# normalise dataframe and check result
df_fit = norm_df(df_fit)
print(df_fit.describe())

for ic in range(2, 7):
    # set up kmeans and fit the model
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhouette score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit, labels))

# Set up k-means with three clusters and fit the model
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(8, 6))
for cluster_label in range(3):
    cluster_points = df_fit[labels == cluster_label]
    plt.scatter(cluster_points["SP.POP.GROW"], cluster_points["SP.URB.GROW"], alpha=0.6, label=f'Cluster {cluster_label}')

    # mark cluster centres
    xc, yc = cen[cluster_label]
    plt.text(xc, yc, f'Cluster {cluster_label}', fontsize=12, color='black', ha='center', va='center')
    plt.plot(xc, yc, "o", markersize=10, label=f'Cluster {cluster_label}', markeredgecolor='k')

# Display the plot
plt.xlabel("Population Growth")
plt.ylabel("Urban Population Growth")
plt.title("Clusters For All Countries")
plt.legend()
plt.grid(True)
plt.show()

from scipy.stats import t

# Use the curve_fit function to fit the model to the data.
popt, pcov = curve_fit(exponential_growth, x_data, y_data)

# Calculate predicted values based on the fitted model.
x_predict = np.linspace(min(x_data), max(x_data), 100)
y_predict = exponential_growth(x_predict, *popt)

# Visualize the results.
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_predict, y_predict, color='red', label='Fitted curve')
plt.xlabel('Population Growth Rate')
plt.ylabel('Urbanization Growth Rate')
plt.title('Exponential Growth Model Fitting')
plt.legend()
plt.grid(True)
plt.show()

