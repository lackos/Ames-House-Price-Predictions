# Data Overview 
 ## Training Data 
Total num of rows: 1460
 
Total num of columns: 80
Num numerical of columns: 37
Num object of columns: 43
### Numerical Data 
Numerical columns: Index(['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
       'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
       'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
       'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
       'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
       'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal',
       'MoSold', 'YrSold', 'SalePrice'],
      dtype='object')
        MSSubClass  LotFrontage        LotArea  OverallQual  OverallCond  ...     PoolArea       MiscVal       MoSold       YrSold      SalePrice
count  1460.000000  1201.000000    1460.000000  1460.000000  1460.000000  ...  1460.000000   1460.000000  1460.000000  1460.000000    1460.000000
mean     56.897260    70.049958   10516.828082     6.099315     5.575342  ...     2.758904     43.489041     6.321918  2007.815753  180921.195890
std      42.300571    24.284752    9981.264932     1.382997     1.112799  ...    40.177307    496.123024     2.703626     1.328095   79442.502883
min      20.000000    21.000000    1300.000000     1.000000     1.000000  ...     0.000000      0.000000     1.000000  2006.000000   34900.000000
25%      20.000000    59.000000    7553.500000     5.000000     5.000000  ...     0.000000      0.000000     5.000000  2007.000000  129975.000000
50%      50.000000    69.000000    9478.500000     6.000000     5.000000  ...     0.000000      0.000000     6.000000  2008.000000  163000.000000
75%      70.000000    80.000000   11601.500000     7.000000     6.000000  ...     0.000000      0.000000     8.000000  2009.000000  214000.000000
max     190.000000   313.000000  215245.000000    10.000000     9.000000  ...   738.000000  15500.000000    12.000000  2010.000000  755000.000000

[8 rows x 37 columns]
 
### Catagorical Data 
Number of unique objects 
{'MSZoning': 5, 'Street': 2, 'Alley': 2, 'LotShape': 4, 'LandContour': 4, 'Utilities': 2, 'LotConfig': 5, 'LandSlope': 3, 'Neighborhood': 25, 'Condition1': 9, 'Condition2': 8, 'BldgType': 5, 'HouseStyle': 8, 'RoofStyle': 6, 'RoofMatl': 8, 'Exterior1st': 15, 'Exterior2nd': 16, 'MasVnrType': 4, 'ExterQual': 4, 'ExterCond': 5, 'Foundation': 6, 'BsmtQual': 4, 'BsmtCond': 4, 'BsmtExposure': 4, 'BsmtFinType1': 6, 'BsmtFinType2': 6, 'Heating': 6, 'HeatingQC': 5, 'CentralAir': 2, 'Electrical': 5, 'KitchenQual': 4, 'Functional': 7, 'FireplaceQu': 5, 'GarageType': 6, 'GarageFinish': 3, 'GarageQual': 5, 'GarageCond': 5, 'PavedDrive': 3, 'PoolQC': 3, 'Fence': 4, 'MiscFeature': 4, 'SaleType': 9, 'SaleCondition': 6}
 
### Columns with nan values 
{'LotFrontage': 259, 'Alley': 1369, 'MasVnrType': 8, 'MasVnrArea': 8, 'BsmtQual': 37, 'BsmtCond': 37, 'BsmtExposure': 38, 'BsmtFinType1': 37, 'BsmtFinType2': 38, 'Electrical': 1, 'FireplaceQu': 690, 'GarageType': 81, 'GarageYrBlt': 81, 'GarageFinish': 81, 'GarageQual': 81, 'GarageCond': 81, 'PoolQC': 1453, 'Fence': 1179, 'MiscFeature': 1406}
 
## Test data 
### Numerical Data 
        MSSubClass  LotFrontage       LotArea  OverallQual  OverallCond  ...  ScreenPorch     PoolArea       MiscVal       MoSold       YrSold
count  1459.000000  1232.000000   1459.000000  1459.000000  1459.000000  ...  1459.000000  1459.000000   1459.000000  1459.000000  1459.000000
mean     57.378341    68.580357   9819.161069     6.078821     5.553804  ...    17.064428     1.744345     58.167923     6.104181  2007.769705
std      42.746880    22.376841   4955.517327     1.436812     1.113740  ...    56.609763    30.491646    630.806978     2.722432     1.301740
min      20.000000    21.000000   1470.000000     1.000000     1.000000  ...     0.000000     0.000000      0.000000     1.000000  2006.000000
25%      20.000000    58.000000   7391.000000     5.000000     5.000000  ...     0.000000     0.000000      0.000000     4.000000  2007.000000
50%      50.000000    67.000000   9399.000000     6.000000     5.000000  ...     0.000000     0.000000      0.000000     6.000000  2008.000000
75%      70.000000    80.000000  11517.500000     7.000000     6.000000  ...     0.000000     0.000000      0.000000     8.000000  2009.000000
max     190.000000   200.000000  56600.000000    10.000000     9.000000  ...   576.000000   800.000000  17000.000000    12.000000  2010.000000

[8 rows x 36 columns]
 
### Catagorical Data 
Number of unique objects 
{'MSZoning': 5, 'Street': 2, 'Alley': 2, 'LotShape': 4, 'LandContour': 4, 'Utilities': 1, 'LotConfig': 5, 'LandSlope': 3, 'Neighborhood': 25, 'Condition1': 9, 'Condition2': 5, 'BldgType': 5, 'HouseStyle': 7, 'RoofStyle': 6, 'RoofMatl': 4, 'Exterior1st': 13, 'Exterior2nd': 15, 'MasVnrType': 4, 'ExterQual': 4, 'ExterCond': 5, 'Foundation': 6, 'BsmtQual': 4, 'BsmtCond': 4, 'BsmtExposure': 4, 'BsmtFinType1': 6, 'BsmtFinType2': 6, 'Heating': 4, 'HeatingQC': 5, 'CentralAir': 2, 'Electrical': 4, 'KitchenQual': 4, 'Functional': 7, 'FireplaceQu': 5, 'GarageType': 6, 'GarageFinish': 3, 'GarageQual': 4, 'GarageCond': 5, 'PavedDrive': 3, 'PoolQC': 2, 'Fence': 4, 'MiscFeature': 3, 'SaleType': 9, 'SaleCondition': 6}
 
### Columns with nan values 
{'LotFrontage': 227, 'Alley': 1352, 'MasVnrType': 16, 'MasVnrArea': 15, 'BsmtQual': 44, 'BsmtCond': 45, 'BsmtExposure': 44, 'BsmtFinType1': 42, 'BsmtFinType2': 42, 'Electrical': 0, 'FireplaceQu': 730, 'GarageType': 76, 'GarageYrBlt': 78, 'GarageFinish': 78, 'GarageQual': 78, 'GarageCond': 78, 'PoolQC': 1456, 'Fence': 1169, 'MiscFeature': 1408}
 
