#import libraries
import pandas as pd
import numpy as np

AttritionData = pd.ExcelFile("Company_X_Employees_Data.xlsx")

#now parse the sheets as dataframes into a variable
Employees_Left = AttritionData.parse("Employees who have left")
Employees_Existing = AttritionData.parse("Existing employees")