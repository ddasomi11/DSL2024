import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def load_climate_data():
    """
    Load climate.csv file.
    """
    # Create dataframes
    file_path = "Climate.csv"
    df = pd.read_csv(file_path, skiprows=4)
    df = df.drop(columns=['Country Code', 'Indicator Code', "Unnamed: 68"])
    df = df.dropna(how='all')
    df.columns = ['Country', 'Indicator'] + [str(col) for col in df.columns[2:]]
    df.set_index(['Country', 'Indicator'], inplace=True)
    df.fillna(0, inplace=True)
    
    # Select data for the time periods 1960-1979 and 2003-2022.
    early_years = ['196{}'.format(i) for i in range(0, 10)] + ['197{}'.format(i) for i in range(0, 10)]
    early_df = df[early_years]

    recent_years = ['200{}'.format(i) for i in range(3, 10)] + ['201{}'.format(i) for i in range(0, 3)] + ['202{}'.format(i) for i in range(0, 3)]  
    recent_df = df[recent_years]

    return early_df, recent_df

def analyze_data(df, countries_name, indicators_name):
    """
    Analyze climate data over time and visualize trends.
    """
    # Define countries and indicators of interest
    countries = ['Afghanistan', 'China', 'United Kingdom']
    indicators = ['Urban population (% of total population)', 
                  'CO2 emissions from liquid fuel consumption (% of total)']

    # Visualize trends for each indicator over the specified time periods
    # Plotting trends for each indicator over the specified time periods
    for period, df in [('1960~1979', early_df), ('2003~2022', recent_df)]:
        print(f"Trends for {period}:")
        for indicator_name in indicators:
            print(f"{indicator_name} over Time")
            plt.figure(figsize=(8,6))
            for country in countries:
                plt.plot(df.loc[(country, indicator_name)].index, df.loc[(country, indicator_name)], label=country)
            plt.xlabel('Year')
            plt.ylabel(indicator_name)
            plt.xticks(rotation=45)
            plt.title(f'{indicator_name} over Time ({period})')
            plt.legend()
            plt.show()

    # Group by statistics
    group_by = df.groupby(['Country', 'Indicator'])['2010'].sum()
    group_by_data = group_by.reset_index()

    # Filter data
    data = group_by_data[group_by_data['Country'].isin(countries_name) 
                         & group_by_data['Indicator'].isin(indicators_name)]
    
    # Create a pivot table
    data_df = data.pivot(index='Country', columns='Indicator', values='2010')

    # Calculate correlation 
    correlation = data_df.corr()

    # Visualize the correlation with heatmap
    plot_heatmap(correlation)

def plot_heatmap(correlation_matrix):
    """
    Plot a heatmap of the correlation matrix.
    """
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                xticklabels=correlation_matrix.columns.str.wrap(20),
                yticklabels=correlation_matrix.columns.str.wrap(20))
    plt.xticks(rotation='horizontal')
    plt.title('Correlation Matrix (Year 2010)')
    plt.show()

def main():
    """
    Main function to load the data and analyze
    """
    # Load data
    early_df, recent_df = load_climate_data()

    # Define countries and indicators to analyze
    countries_name = ['Afghanistan', 'China', 'United Kingdom']
    indicators_name = ['Urban population (% of total population)', 
                       'CO2 emissions from liquid fuel consumption (% of total)']

    # Analyze data
    analyze_data(recent_df, countries_name, indicators_name)

if __name__ == '__main__':
    main()
