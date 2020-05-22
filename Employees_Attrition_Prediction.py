#import libraries
import pandas as pd
import numpy as np

AttritionData = pd.ExcelFile("Company_X_Employees_Data.xlsx")

#now parse the sheets as dataframes into a variable
Employees_Left = AttritionData.parse("Employees who have left")
Employees_Existing = AttritionData.parse("Existing employees")

#merge both dataframes/sheets but before that insert a new column (Attrition) in each dataframe

#Get the length of the rows in each dataframe
len1 = Employees_Left.shape[0]
len2 = Employees_Existing.shape[0]

#create a dummy random array equal in length to each dataframe
dummyArray1 = np.arange(len1)
dummyArray2 = np.arange(len2)

#Re-assign the values of the dummy array respectively, such that, say Attrition in Employees who left dataframe is 1
# while in the other dataframe is 0
for x in dummyArray1:
    dummyArray1[x] = 1

for n in dummyArray2:
    dummyArray2[n] = 0

#insert the dummy arrays as new columns in the dataframes
Employees_Left["Attrition"] = dummyArray1
Employees_Existing["Attrition"] = dummyArray2

#merge existing employees to those who left using concat, use ignore_index=True so as to reassign the index after merge
employees = pd.concat([Employees_Left, Employees_Existing], ignore_index=True)

#enable all columns to show during print, instead of being truncated
pd.set_option('max_columns', None)

#Convert Categorical values to Numeric Values
employees = pd.get_dummies(employees)

#Separating the dependent and independent values
X = employees.drop(['Attrition'], axis=1)
Y = employees['Attrition']