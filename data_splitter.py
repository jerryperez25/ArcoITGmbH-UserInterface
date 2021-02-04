import pandas as pd

full_data = pd.read_csv("sample_data/sample_data.csv", header =0)
#drops all NaN
full_data = full_data.dropna()
full_data = full_data.drop(columns = ['Packets','Bytes','Packets A -> B','Bytes A -> B','Packets B -> A','Bytes B -> A'
    ,'Rel Start','Duration','Bits/s A -> B','Bits/s B -> A'])
#drops that row from being included in data
full_data = full_data.drop(index = 0)
#this populates the red data (will be the unique) - first half of the data
red_half_data = full_data.head(937)
#this drops all duplicated from the red data to make it unique
red_half_data = red_half_data.drop_duplicates()
#this populates the blue data (will not be touched) - tail end of the data
blue_half_data = full_data.tail(937)
intersecting_data = pd.merge(red_half_data, blue_half_data, on = ['Address A', 'Port A', 'Address B', 'Port B'], how='inner')
#this gets an empty dataframe, which either means that all data is unique, or that its not correct
print(intersecting_data)



