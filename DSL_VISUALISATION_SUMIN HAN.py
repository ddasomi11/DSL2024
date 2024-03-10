# Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv("Sales.csv")
print( df.head () )


# 1. Drawing a line plot representing the total sales volume by hour (time slot) for both genders. 

#  add 'Hour' column and  Round the hour to the nearest hour
df['Time'] = pd.to_datetime(df['Time'])

df['Hour'] = df['Time'].dt.hour

def round_hour(hour):
    """This code line rounds the hour to the nearest whole number"""
    return int(hour)

df['Hour'] = df['Hour'].apply(round_hour)

def plot_sales_hour_gender(df):
    """"This code draws a line plot representing the total sales by hour for both genders"""


# Group the data by hour and calculate the total consumption for each gender
grouped = df.groupby(['Hour', 'Gender'])['Total'].mean().unstack()

# Draw line plots
plt.figure(figsize=(15, 8))

# Draw line plot for male data
plt.plot(grouped.index, grouped['Male'], marker='o', label='blue')

# Draw line plot for female data
plt.plot(grouped.index, grouped['Female'], marker='s', label='Female', color="magenta")

# Set title and axis labels
plt.title('Total Consumption Variation by Hour')
plt.xlabel('Hour')
plt.ylabel('Total Sales')

# Display legend
plt.legend()

# Show the plot 
plt.grid(True)
plt.xticks(grouped.index)
plt.tight_layout()
plt.show()


# 2. Creating a bar plot to illustrate the product line consumption by gender

def plot_product_line_gender(df):
    
    """Creates a bar plot to illustrate the product line consumption by gender"""
    
# Calculate the total sum of each Product line for each Gender
grouped = df.groupby(['Gender', 'Product line'])['Total'].mean().unstack()

# Set colors
colors = ['#FFD1DC', '#FFB6C1', '#FFA07A', '#FFDAB9', '#87CEFA', '#98FB98']

# Create a bar plot
grouped.plot(kind='bar', figsize=(10, 6),color=colors)

# Set title, xlabel, ylabel
plt.title('Consumption of Product Lines by Gender')
plt.xlabel('Gender')
plt.ylabel('Total Sales')
plt.legend(title='Product Line')
plt.xticks(rotation=0) 
plt.tight_layout()  # Adjust the layout of the plot

# Show the plot
plt.show() 


# 3. Creating a pie chart to represent customer types by branch

# Grouping data by Branch and Customer type
def plot_customer_type_by_branch(df):
    """Creates a pie chart to represent customer types by branch"""
    
grouped = df.groupby(['Branch', 'Customer type']).size().unstack()
"""
This function groups the data in the DataFrame 'df' by the columns 'Branch' and 'Customer type'. 
Then, It counts the occurrences of each combination of branch and customer type. 
Finally, it reshapes the result into a two-dimensional table, with branches as rows and customer types as columns, 
providing a clear overview of the distribution of customer types across different branches.
"""

# Create pie charts
plt.figure(figsize=(12, 8))
colors = ['#ADD8E6', '#FFB6C1']

# Draw pie charts, set titles
for i in range(len(grouped.index)): 
    plt.pie(grouped.iloc[i], labels=grouped.columns, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title('Customer Type Distribution in branch {}'.format(grouped.index[i]))
    plt.axis('equal') 
    plt.show()
    
#Summary

plot_sales_hour_gender(df)
plot_product_line_gender(df)
plot_customer_type_by_branch(df)
    

   