#%%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pickle
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import copulae
from copulae import EmpiricalCopula, pseudo_obs
from copulae.datasets import load_marginal_data
import yfinance as yf
from yahoofinancials import YahooFinancials
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.api import VAR
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, f_regression
import geopandas as pgd
import branca
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
import seaborn as sns
import tslearn
#%%

data = pd.read_csv(
    "./markettrends0.csv",
    dtype={
        "state": "str",
        "sa3_name16": "str",
        "sa4_name16": "str",
        "postcode": "str",
        "state": "str",
        "property_type": "str",
    },
)
ndata = data.fillna(
    {"Volume of new rental listings (1 month)": 0, "Volume of sales (1 month)": 0}
).dropna(subset=["postcode"])
ndatahouses = ndata[:][ndata.property_type == "Houses"]
ndataunits = ndata[:][ndata.property_type == "Units"]


with open('DataFrameDict_postcode.pickle', 'rb') as f:
     DataFrameDict_postcode =  pickle.load(f) 
        
aveHPI_lv3 = []
lst = ["postcode", "sa4", "state", "logHPI", "logHPIdiff"]
# Calling DataFrame constructor on list
for key in DataFrameDict_postcode.keys():
    df = pd.DataFrame([], columns=lst)
    df["Description"] = DataFrameDict_postcode[key]["value_at_date"].loc[
        ("2021-06" > DataFrameDict_postcode[key]["value_at_date"])
        & (DataFrameDict_postcode[key]["value_at_date"] > "1999-12")
    ]
    df["logHPI"] =np.log2(
        DataFrameDict_postcode[key]["Hedonic Home Value Index"].loc[
            ("2021-06" > DataFrameDict_postcode[key]["value_at_date"])
            & (DataFrameDict_postcode[key]["value_at_date"] > "1999-12")
        ]
    )
    df["postcode"] = key
    df["sa4"] = DataFrameDict_postcode[key]["sa4_name16"].loc[
        ( DataFrameDict_postcode[key]["postcode"] == key)]
    df["state"] = DataFrameDict_postcode[key]["state"].loc[
        ( DataFrameDict_postcode[key]["postcode"] == key)]
    
    df["logHPIdiff"] = df["logHPI"].diff(1)
    aveHPI_lv3.append(df)

aveHPIdf_lv3 = pd.concat(aveHPI_lv3)
aveHPIdf_lv3 = aveHPIdf_lv3.reset_index(drop=True)
aveHPIdf_lv3['date'] = pd.to_datetime(aveHPIdf_lv3['Description'],errors = 'coerce')
aveHPIdf_lv3['year'] = pd.DatetimeIndex(aveHPIdf_lv3['date']).year

aveHPIdf_lv3 = aveHPIdf_lv3[aveHPIdf_lv3['date'] > '2000-12-31']

avg_1 = aveHPIdf_lv3.groupby(["date"])["logHPIdiff"].agg(["mean"]).reset_index()
avg_1 = avg_1.rename(columns = {'mean':'l0'}).dropna().reset_index(drop = True)

sum_1 = aveHPIdf_lv3.groupby(["date"])["logHPIdiff"].agg(["sum"]).reset_index()
sum_1 = sum_1.rename(columns = {'sum':'l0_s'}).dropna().reset_index(drop = True)
sum_1 = sum_1.reset_index(drop=True)

#sa4 
avg_2 = aveHPIdf_lv3.groupby(["sa4","date"])["logHPIdiff"].agg(["mean"]).reset_index()
avg_2 = avg_2.rename(columns = {'mean':'l01'}).dropna().reset_index(drop = True)

sum_2 = aveHPIdf_lv3.groupby(["sa4","date"])["logHPIdiff"].agg(["sum"]).reset_index()
sum_2 = sum_2.rename(columns = {'sum':'l01_s'}).dropna().reset_index(drop = True)
sum_2 = sum_2.reset_index(drop=True)

#postcode
avg_3 = aveHPIdf_lv3.groupby(["postcode","date"])["logHPIdiff"].agg(["mean"]).reset_index()
avg_3 = avg_3.rename(columns = {'mean':'l012'}).dropna().reset_index(drop = True)

# Set the 'date' column to a datetime format
avg_2['date'] = pd.to_datetime(avg_2['date'], format='%Y-%m-%d')

# Create a new dataframe with monthly dates from '2000-01-31' to '2021-05-31'
date_range = pd.date_range('2001-01-31', '2021-05-31', freq='M')
new_df = pd.DataFrame({'date': date_range})

# Use pivot to reshape the 'avg_2' dataframe
pivot_df = avg_2.pivot(index='date', columns='sa4', values='l01')

# Merge the 'new_df' and 'pivot_df' dataframes
merged_df = new_df.merge(pivot_df, on='date', how='outer')

# Sort the columns alphabetically by column name
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# Fill missing values with NaN
avg_2_pivot = merged_df.fillna(np.nan).drop('date', axis=1)

# Set the 'date' column to a datetime format
sum_2['date'] = pd.to_datetime(sum_2['date'], format='%Y-%m-%d')

# Create a new dataframe with monthly dates from '2000-01-31' to '2021-05-31'
date_range = pd.date_range('2001-01-31', '2021-05-31', freq='M')
new_df = pd.DataFrame({'date': date_range})

# Use pivot to reshape the 'sum_2' dataframe
pivot_df = sum_2.pivot(index='date', columns='sa4', values='l01_s')

# Merge the 'new_df' and 'pivot_df' dataframes
merged_df = new_df.merge(pivot_df, on='date', how='outer')

# Sort the columns alphabetically by column name
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# Fill missing values with NaN
sum_2_pivot = merged_df.fillna(np.nan).drop('date', axis=1)

# Set the 'date' column to a datetime format
avg_3['date'] = pd.to_datetime(avg_3['date'], format='%Y-%m-%d')

# Create a new dataframe with monthly dates from '2000-01-31' to '2021-05-31'
date_range = pd.date_range('2001-01-31', '2021-05-31', freq='M')
new_df = pd.DataFrame({'date': date_range})

# Use pivot to reshape the 'avg_3' dataframe
pivot_df = avg_3.pivot(index='date', columns='postcode', values='l012')

# Merge the 'new_df' and 'pivot_df' dataframes
merged_df = new_df.merge(pivot_df, on='date', how='outer')

# Sort the columns alphabetically by column name
merged_df = merged_df.reindex(sorted(merged_df.columns), axis=1)

# Fill missing values with NaN
avg_3_pivot = merged_df.fillna(np.nan).drop('date', axis=1)

avg1 = avg_1.set_index("date")
avg2 = avg_2.set_index("date")
avg3 = avg_3.set_index("date")
df_dis =  pd.concat([avg1 , avg2.pivot(columns='sa4', values='l01'), avg3.pivot(columns='postcode', values='l012')],axis = 1)
#%% Exo National

xls1 = pd.read_excel("f01hist.xls", sheet_name="Data", header=2)
xls1 = xls1.drop(xls1.index[0:8], axis=0)
xls1["Description"] = pd.to_datetime(xls1["Description"])
xls1["Description"] = xls1["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
xls3 = pd.read_excel("f11hist-1969-2009.xls", sheet_name="Data", header=2)
xls3 = xls3.drop(xls3.index[0:8], axis=0)
xls3["Description"] = pd.to_datetime(xls3["Description"])
xls3["Description"] = xls3["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
xls4 = pd.read_excel("f11hist.xls", sheet_name="Data", header=2)
xls4 = xls4.drop(xls4.index[0:8], axis=0)
xls4["Description"] = pd.to_datetime(xls4["Description"])
xls4["Description"] = xls4["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
xls34 = pd.concat([xls3, xls4], axis=0)
xls5 = pd.read_excel("g01hist.xls", sheet_name="Data", header=2)
xls5 = xls5.drop(xls5.index[0:8], axis=0)
xls5["Description"] = pd.to_datetime(xls5["Description"])
xls5["Description"] = xls5["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
xls6 = pd.read_excel("h01hist.xls", sheet_name="Data", header=2)
xls6 = xls6.drop(xls6.index[0:8], axis=0)
xls6["Description"] = pd.to_datetime(xls6["Description"])
xls6["Description"] = xls6["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
xls7 = pd.read_excel("h03hist.xls", sheet_name="Data", header=2)
xls7 = xls7.drop(xls7.index[0:8], axis=0)
xls7["Description"] = pd.to_datetime(xls7["Description"])
xls7["Description"] = xls7["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))
# Download ASX data from Yahoo Finance

ASX200data = yf.download('^AXJO', start='1999-01-01', end='2021-06-01', interval = "1mo", progress=False)
ASX200data = ASX200data.reset_index()
ASX200data["Description"] = pd.to_datetime(ASX200data["Date"])
ASX200data["Description"] = ASX200data["Description"].dt.date.apply(lambda x: x.strftime("%Y-%m"))

macvar = pd.DataFrame().assign(
    Description=xls7.loc[
        ("2021-06" > xls7["Description"]) & (xls7["Description"] > "1998-12")
    ]["Description"]
)
macvar.reset_index(drop=True, inplace=True)

macvar = pd.merge(
    macvar,
    xls1[["Description", "Cash Rate Target; monthly average"]],
    on="Description",
    how="left",
)
macvar = pd.merge(
    macvar,
    xls34[["Description", "AUD/USD Exchange Rate; see notes for further detail."]],
    on="Description",
    how="left",
)
macvar = pd.merge(
    macvar,
    xls5[["Description", "Consumer price index; All groups"]],
    on="Description",
    how="left",
)
macvar = pd.merge(
    macvar,
    xls6[["Description", "Gross domestic product (GDP); Chain volume"]],
    on="Description",
    how="left",
)
macvar = pd.merge(
    macvar,
    xls7[["Description", "Retail sales; All industries; Current price"]],
    on="Description",
    how="left",
)
macvar = pd.merge(
    macvar,
    xls7[["Description", "Private dwelling approvals"]],
    on="Description",
    how="left",
)

macvar = pd.merge(
    macvar,
    ASX200data[["Description", "Adj Close"]],
    on="Description",
    how="left",
)

macvar = macvar.set_axis(
    ["date", "ir", "exr", "cpi", "gdp", "rs", "pda", "asx"], axis=1, inplace=False
)
macvar["ir"] = macvar["ir"].astype(float, errors="raise")
macvar["exr"] = macvar["exr"].astype(float, errors="raise")
macvar["cpi"] = macvar["cpi"].astype(float, errors="raise")
macvar["gdp"] = macvar["gdp"].astype(float, errors="raise")
macvar["rs"] = macvar["rs"].astype(float, errors="raise")
macvar["pda"] = macvar["pda"].astype(float, errors="raise")
macvar["asx"] = macvar["asx"].astype(float, errors="raise")
macvar = macvar.interpolate()

threshold = 0.01

def selectStationaySeries(variable_tar):
        if adfuller(variable_tar.dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar
            suffix =  "original"
            print(suffix)
        elif adfuller(variable_tar.pct_change(1).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(1).dropna(how="all")
            suffix = "1storderdiff"
            print(suffix)
        elif adfuller(variable_tar.pct_change(3).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(3).dropna(how="all")
            suffix = "seasonaldiff"
            print(suffix)
        elif adfuller(variable_tar.pct_change(12).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(12).dropna(how="all")
            suffix = "annualdiff" 
            print(suffix)    
        elif adfuller(variable_tar.pct_change(1).diff().dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(1).diff().dropna(how="all")
            suffix = "2ndorderdiff"
            print(suffix)
        else:
            print("not found")
        return(pd.DataFrame(stationary_variable))
    
macvarst = macvar["date"] 
for vn in macvar.columns.values[1:]:
    macvarst = pd.concat([macvarst, selectStationaySeries(macvar[vn])], axis=1)

macvarsta = macvarst  
matrixmac = macvarsta.corr().round(2)
macdata = macvarsta.loc[
    ("2021-06" > macvarsta["date"]) & (macvarsta["date"] > "1999-12")
]

pca = PCA()
X_train = macdata.loc[:, macdata.columns != "date"]
# poly= PolynomialFeatures(degree=2)
scaler = StandardScaler()

#X_train_poly = poly.fit_transform(X_train)
#X_train_std = scaler.fit_transform(X_train_poly)
X_train_std = scaler.fit_transform(X_train)

pca.fit(X_train_std)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.9) + 1
print(d)

pca_d = PCA(n_components=d)
pca_d.fit(X_train_std)
X_pca_d = pca_d.transform(X_train_std)



df_loadings_nat = pd.DataFrame(pca_d.components_)
df_loadings_nat.columns =  macvarst.columns[1:] 
PCnames = []
for i in range(1, d + 1, 1):
    PCnames.append(f"PC{i}")
PC_nat = pd.DataFrame(X_pca_d, columns=PCnames)

PC_nat['date'] = macdata['date'].values


#%%
with open('DataFrameDict_Hsa4.pickle', 'rb') as f:
    DataFrameDict_Hsa4 =  pickle.load(f)
with open('dfHILDA.pickle', 'rb') as f:
    dfHILDA =  pickle.load(f)  

with open('dfHILDA_cut.pickle', 'rb') as f:
      dfHILDA_cut =  pickle.load(f)  
#%%
threshold = 0.01
df = dfHILDA_cut.set_index("date")

df1 = df.resample("Y").mean().iloc[:, :]
vrst = pd.DataFrame([])

# need a dataframe to writedown the way to stabilize all variables
df_marks = pd.DataFrame([],columns=['Variable Names', 'Approach'])

for vn in df1.columns.values:
    variable_tar = df1[vn]
    if adfuller(variable_tar.interpolate("bfill").interpolate("ffill"))[1] < threshold:
        stationary_variable = variable_tar.interpolate("bfill").interpolate("ffill")
        suffix =  "original"
        new_row = {'Variable Names': vn,'Approach': suffix}
        df_marks = df_marks.append(new_row, ignore_index=True)
        newseries = pd.DataFrame(stationary_variable)
        vrst = pd.concat([vrst, newseries], axis=1)

    
    elif adfuller(variable_tar.diff().interpolate("bfill").interpolate("ffill"))[1] < threshold:
        stationary_variable = variable_tar.diff().interpolate("bfill").interpolate("ffill")
        suffix = "1storderdiff"
        new_row = {'Variable Names': vn,'Approach': suffix}
        df_marks = df_marks.append(new_row, ignore_index=True)
        newseries = pd.DataFrame(stationary_variable)
        vrst = pd.concat([vrst, newseries], axis=1)

    elif adfuller(variable_tar.diff().diff().interpolate("bfill").interpolate("ffill"))[1] < threshold:
        stationary_variable = variable_tar.diff().diff().interpolate("bfill").interpolate("ffill")
        suffix = "2ndorderdiff"
        new_row = {'Variable Names': vn,'Approach': suffix}
        df_marks = df_marks.append(new_row, ignore_index=True)
        newseries = pd.DataFrame(stationary_variable)
        vrst = pd.concat([vrst, newseries], axis=1)
#%%
Start_time = '2005-12-31'
Start_Year = 2005

vrdata = vrst[vrst.index> Start_time ].reset_index()
    
pca = PCA()
X_train = vrdata.loc[:, vrdata.columns != "date"]
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)

pca.fit(X_train_std)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d2 = np.argmax(cumsum >= 0.9) + 1
print(d2)

pca_d_sa4 = PCA(n_components=d2)
pca_d_sa4.fit(X_train_std)
X_pca_d_sa4 = pca_d_sa4.transform(X_train_std)

PCnames = []
for i in range(1, d2 + 1, 1):
    PCnames.append(f"sa4PC{i}")
    
pcadf = pd.DataFrame(X_pca_d_sa4, columns=PCnames)
pcadf['date'] =vrdata['date']  
pcadf["date"] = pd.to_datetime(pcadf["date"]) - pd.offsets.MonthEnd(6)
pcadf_sa4_date = pcadf


#%%

marks1 =df_marks['Variable Names'].loc[df_marks['Approach'] == 'original'].values
marks2 =df_marks['Variable Names'].loc[df_marks['Approach'] == "1storderdiff"].values
marks3 =df_marks['Variable Names'].loc[df_marks['Approach'] == "2ndorderdiff"].values
def selectStationaySeries3(variable_tar, vn):

    if vn in marks1:
        stationary_variable = variable_tar.interpolate("bfill").interpolate("ffill")
        suffix =  "original"
        #print(suffix)
    
    elif vn in marks2:
        stationary_variable = variable_tar.diff().interpolate("bfill").interpolate("ffill")
        suffix = "1storderdiff"
        #print(suffix)   

    elif vn in marks3:
        stationary_variable = variable_tar.diff().diff().interpolate("bfill").interpolate("ffill")
        suffix = "2ndorderdiff"
        #print(suffix）       
    
    return(pd.DataFrame(stationary_variable))
# %%
def HILDAvarExt(key):
    df = DataFrameDict_Hsa4[key].interpolate("bfill").interpolate("ffill")
    df = df.set_index("date")
    df1 = df.resample("Y").mean().iloc[:, :]
    vrst = pd.DataFrame([], columns = df_marks['Variable Names'].values)
    lst = list(set(df_marks['Variable Names'].values) & set(df1.columns.values))
    for vn in lst:
        vrst[vn] = selectStationaySeries3(df1[vn], vn)
    stat_var = vrst.reset_index().fillna(value = 0) 
    return(stat_var) 
# %% Respect your own work. 
def PCA_val(gamma1,vrdata0):
    tmp = pd.DataFrame([])
    for key in DataFrameDict_Hsa4.keys(): 
        df = HILDAvarExt(key)
        df1= df[vrdata0.columns.tolist()]
        df2 = df1.drop(df1.columns[0], axis=1).iloc[(Start_Year - 2020):,:].reset_index(drop = True)
        
        vrdata1 = vrdata0.drop(vrdata0.columns[0],axis=1)
        vrdata2= vrdata1.apply(lambda x: x*gamma1)
        
        df_add = df2.add(vrdata2, fill_value=0)
        scaler = StandardScaler()
        df_add = pd.DataFrame(scaler.fit_transform(df_add))
        tmp = pd.concat([tmp, df_add],axis = 0)

    pca = PCA()
    X_train_std = tmp
    pca.fit(X_train_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d3 = np.argmax(cumsum >= 0.9) + 1

    pca_d_sa4 = PCA(n_components=d3)
    pca_d_sa4.fit(X_train_std)
    # X_pca_d_sa4 = pca_d_sa4.transform(X_train_std)

    # PCnames = []
    # for i in range(1, d3 + 1, 1):
    #     PCnames.append(f"sa4PC{i}")

    # pcadf = pd.DataFrame(X_pca_d_sa4, columns=PCnames)
    # pcadf_sa4_date = pcadf
    # pcadf_sa4_date
    return  d3, pca_d_sa4


# %%
PCnumbers = []
x = range(0,400,2)
for gamma in x:
    gamma1 = gamma
    d2, pcafit = PCA_val(gamma1,vrdata)
    PCnumbers.append(d2)
    print(gamma)


#%%
# Create an empty list to store the indices of different elements
diff_indices = []

# Iterate through the PCnumbers array and compare each element to the previous one
for i in range(1, len(PCnumbers)):
    if PCnumbers[i] != PCnumbers[i-1]:
        diff_indices.append(i)

# Add 0 at the beginning of the diff_indices array
diff_indices.insert(0, 0)

# Multiply every element in the array by 2
gamma_indices = [x * 2 for x in diff_indices]

#%%
Dict_pcd_algo = {}
for gamma in gamma_indices:
    Dict_pcd_algo[gamma]  = {}
    d, pcafit = PCA_val(gamma,vrdata)
    Dict_pcd_algo[gamma]['d']  =   d
    Dict_pcd_algo[gamma]['pcafit']  =  pcafit


#%%

# from scipy.interpolate import make_interp_spline, BSpline
# # represents number of points to make between x.min and x.max
# xnew =np.linspace(0, 1, 150)
# spl = make_interp_spline(x, PCnumbers, k=1)  # type: BSpline
# PCno_smooth = spl(xnew)

# color1 = "#0085c3"
# color2 = "#7ab800"
# color3 = "#dc5034"

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111)
# ax.step(x, PCnumbers,where='post',color = color3, label = '$d_{90\%}$')
# ax.grid(ls=":", color="gray", alpha=0.6)

# plt.axhline(y=5, color=color1, linestyle='--',  label = '$d_{L_1}$')
# plt.axhline(y=3, color=color2, linestyle='--', label = '$d_{L_2}$')

# plt.xlabel(r'$\gamma$')
# plt.ylabel('Number of risk factors at the SA4s level')
# plt.ylim(0, 23)
# plt.xlim(0, 700)
# plt.xticks(rotation=45, fontsize=12)
# plt.legend(loc='upper right', fontsize=12)
# plt.yticks(fontsize=12)

#%%
avg1 = avg_1.set_index("date")
avg2 = avg_2.set_index("date")
avg3 = avg_3.set_index("date")
df_dis =  pd.concat([avg1 , avg2.pivot(columns='sa4', values='l01'), avg3.pivot(columns='postcode', values='l012')],axis = 1)
# %% transform fucntion 
def transform_df(df0):
    df = df0.copy()
    # create a copy of df_dis to avoid modifying the original dataframe
    df.index = pd.to_datetime(df.index)
     # subtract the value of each column in group 3 by the value of the corresponding column in group 2
    
    for col in avg_3_pivot.columns:
        df[col] = df[col].sub(df['l0'], axis=0)
    
    for col in avg_2_pivot.columns:
        df[col] = df[col].sub(df['l0'], axis=0)

    for col in avg_2_pivot.columns:
# Group the data by year and calculate the annual average
        annual_avg = df[col].groupby(df.index.year).transform('mean')
        df.loc[:, col] = annual_avg

    for col in avg_3_pivot.columns:
        sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == col]['sa4'].iloc[0]
        df[col] = df[col].sub(df[sa4], axis=0)
    # subtract the value under column 'l0' from columns in group 2

    return df

df_l = transform_df(df_dis)
df_l.index = pd.to_datetime(df_l.index)
# Change the index to year-month format
df_l.index = df_l.index.to_period('M')


PC_l0 = PC_nat.set_index('date')
PC_l0.index = pd.to_datetime(PC_l0.index)
PC_l0.index = PC_l0.index.to_period('M')
df_combined = pd.concat([ PC_l0, df_l['l0']], axis=1)

# %%
def d2pca_SA4(gamma1,key):
    df = HILDAvarExt(key)
    df1= df[vrdata.columns.tolist()]   
    df2 = df1.drop(df1.columns[0], axis=1).iloc[Start_Year-2020:,:].reset_index(drop = True)     

    # vrdata1 = vrdata.drop(vrdata.columns[0],axis=1)
    # vrdata2 = vrdata1.apply(lambda x: x*gamma1)
        
    # df_add = df2.add(vrdata2)

    df_add = df2

    d4 = Dict_pcd_algo[gamma1]['d']
    pca_d_sa4 = Dict_pcd_algo[gamma1]['pcafit']

    X_train = df_add 
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    
    X_pca_d_sa4 = pca_d_sa4.transform(X_train_std)

    PCnames = []
    for i in range(1, d4 + 1, 1):
        PCnames.append(f"sa4PC{i}")

    pcadf = pd.DataFrame(X_pca_d_sa4, columns=PCnames)
    pcadf_sa4_date = pcadf

    new_index = pd.date_range(start=f'{Start_Year+1}', end='2021', freq='Y')
# Convert the index to the 'YYYY' format
    new_index = new_index.strftime('%Y')
# Assign the new index to the dataframe
    pcadf_sa4_date.index = new_index
    pcadf_sa4_date = pcadf_sa4_date.rename_axis('date')
    
    return pcadf_sa4_date


# %% Calculate AIC according to gamma and lag: 

def  K_best_VAR(gamma1,Kbest):
    results_dict = {}   
    for sa4 in DataFrameDict_Hsa4.keys():
        data  = d2pca_SA4(gamma1,sa4)
        l1 = pd.DataFrame(df_l_new[sa4].groupby(df_l_new.index.year).mean()).iloc[Start_Year-2020:,:]
        l1.index = l1_index
        l1  = l1.rename_axis('date')
        X = data
        y = l1.iloc[:,0]
        # Use SelectKBest to select the best 5 variables based on f_regression
        selector = SelectKBest(score_func=f_regression, k=Kbest)
        selector.fit(X, y)
        # Get the indices of the selected variables
        selected_indices = selector.get_support(indices=True)
        results_dict[sa4] = selected_indices
        
    results_df = pd.DataFrame.from_dict(results_dict, orient='index')
    value_counts = results_df.stack().value_counts()
    # Get the top 10 values and their counts
    top_values = value_counts.head(Kbest).index.tolist()
    # Display the top values and their counts

    return top_values 
#%%
l1_index = pd.date_range(start=f'{Start_Year+1}', end='2021', freq='Y')
l1_index = l1_index.strftime('%Y')
df_l_new = df_l[df_l.index < '2021-01']

def gamma2AIC_VAR(gamma1,lag0,d_U):
    d2 = Dict_pcd_algo[gamma1]['d']
    aic_max = []
    selected_indices = K_best_VAR(gamma1,d_U)
    for d in range(1,d_U+1):
        try:
            aic_rec = [] 
            for sa4 in DataFrameDict_Hsa4.keys():
                colnames = d2pca_SA4(gamma1,sa4).columns[selected_indices]
                df = d2pca_SA4(gamma1,sa4)[colnames]
                data  = df.iloc[:,0:d]
                l1 = pd.DataFrame(df_l_new[sa4].groupby(df_l_new.index.year).mean()).iloc[Start_Year-2020:,:]
                l1.index = l1_index
                l1  = l1.rename_axis('date')
                df_combined = pd.concat([data,l1],axis = 1)
                df = df_combined.dropna()
                df = df.reset_index(drop = True)
                train_data = df
                model = VAR(train_data)
                results = model.fit(maxlags = lag0)
                aic_rec.append(results.aic)
            avg_aic_d = np.mean(aic_rec)
            aic_max.append(avg_aic_d)             
        except (np.linalg.LinAlgError): 
            avg_aic_d = 9999
            print("Error occurred. Exiting loop.")
            aic_max.append(avg_aic_d)    
            continue        
    return  aic_max 
# %%
Dict_lag1 = {}
for gamma_new in gamma_indices:
    Dict_lag1[gamma_new] = gamma2AIC_VAR(gamma_new,1,5)
    
#%%

# Initialize variables to store the minimum value and its corresponding key and index
min_val = float('inf')
min_key = None
min_idx = None

# Iterate through the dictionary and its arrays to find the minimum value and its corresponding key and index
for key, arr in Dict_lag1.items():
    for i, val in enumerate(arr):
        if val < min_val:
            min_val = val
            min_key = key
            min_idx = i

# Print the results
print(f"The smallest element in {min_key} is {min_val}, found at index {min_idx}.")




#%%
df_AIC_L1 = pd.DataFrame.from_dict(Dict_lag1, orient='index')
new_index = list(range(19, 7, -1))

# Set the new index values
df_AIC_L1 = df_AIC_L1.set_index(pd.Index(new_index))

# Create a dictionary of new column names
new_columns = {old_col: str(i+1) for i, old_col in enumerate(df_AIC_L1.columns)}

# Rename the columns using the dictionary
df_AIC_L1 = df_AIC_L1.rename(columns=new_columns)

#%%import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# Create a figure and 3D axis
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set the x, y, and z data
y_data = df_AIC_L1.index.values
x_data = df_AIC_L1.columns.values.astype(int)
z_data = df_AIC_L1.values

# Create a mesh grid for the x and y data
X, Y = np.meshgrid(x_data, y_data)

# max_values_rows = df_AIC_L1.max(axis=1)
# max_values_coords = [(i, j, max_values_rows[i - df_AIC_L1.index[0]],) for i in df_AIC_L1.index.values for j in df_AIC_L1.columns.values]
# # for coords in max_values_coords:
# #     ax.scatter(coords[1], coords[0], coords[2], marker='o', s=50, color='blue')

# # Plot the surface
surf = ax.plot_surface(X, Y, z_data, cmap='viridis', linewidth=0.5, edgecolor='k')

# Set the axis labels
ax.set_ylabel('$d_{90\%}$')
ax.set_xlabel('$d^*$')


ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Show the plot
plt.show()

# %%
Dict_lag2 = {}
for gamma_new in gamma_indices:
    Dict_lag2[gamma_new] = gamma2AIC_VAR(gamma_new,2,3)



# Initialize variables to store the minimum value and its corresponding key and index
min_val = float('inf')
min_key = None
min_idx = None

# Iterate through the dictionary and its arrays to find the minimum value and its corresponding key and index
for key, arr in Dict_lag2.items():
    for i, val in enumerate(arr):
        if val < min_val:
            min_val = val
            min_key = key
            min_idx = i

# Print the results
print(f"The smallest element in {min_key} is {min_val}, found at index {min_idx}.")




# %%
df_AIC_L2 = pd.DataFrame.from_dict(Dict_lag2, orient='index')
new_index = list(range(19, 7, -1))

# Set the new index values
df_AIC_L2 = df_AIC_L2.set_index(pd.Index(new_index))

# Create a dictionary of new column names
new_columns = {old_col: str(i+1) for i, old_col in enumerate(df_AIC_L2.columns)}

# Rename the columns using the dictionary
df_AIC_L2 = df_AIC_L2.rename(columns=new_columns)


df_AIC_L2['3']  = df_AIC_L2['3'] 
# %%



fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Set the x, y, and z data
y_data = df_AIC_L2.index.values
x_data = df_AIC_L2.columns.values.astype(int)
z_data = df_AIC_L2.values

# Create a mesh grid for the x and y data
X, Y = np.meshgrid(x_data, y_data)

# max_values_rows = df_AIC_L1.max(axis=1)
# max_values_coords = [(i, j, max_values_rows[i - df_AIC_L1.index[0]],) for i in df_AIC_L1.index.values for j in df_AIC_L1.columns.values]
# # for coords in max_values_coords:
# #     ax.scatter(coords[1], coords[0], coords[2], marker='o', s=50, color='blue')

# # Plot the surface
surf = ax.plot_surface(X, Y, z_data, cmap='viridis', linewidth=0.5, edgecolor='k')

# Set the axis labels
ax.set_ylabel('$d_{90\%}$')
ax.set_xlabel('$d^*$')


ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Show the plot
plt.show()
# %%

# %%

finald, finalpca = PCA_val(44,vrdata)
finalKBest =  K_best_VAR(44,5)
# %%
df_loadings = pd.DataFrame(finalpca.components_[finalKBest])
df_loadings.columns = vrdata.columns[1:]
# # %%
# df_loadings_t = df_loadings.transpose()

# # %%
# df_loadings_t.columns = (["fin_rf_" + str(i) for i in range(1, 6)])

# #%%
# # convert the DataFrame to a LaTeX tabular environment
# df_latex = df_loadings_t.to_latex()

# # print the LaTeX code
# print(df_latex)
# # %%

#%%
selected_indices = K_best_VAR(44,5)

# %%
postcode_deleted = list(aveHPIdf_lv3.loc[aveHPIdf_lv3['sa4'] == 'Wide Bay']['postcode'].unique())
listpostcode = list(avg_3_pivot.columns)
listpostcode = [x for x in listpostcode if x not in postcode_deleted]



# %%
train_data_nat  = pd.concat([ PC_l0, df_l['l0']], axis=1).loc['2006-01':'2020-12']
model_VAR_nat  = VAR(train_data_nat)
results_nat = model_VAR_nat.fit(maxlags = 1)
para_VAR_nat= results_nat.params

# %%
Dict_VARpara_SA4 = {}
for sa4 in DataFrameDict_Hsa4.keys():
    trainingdf2 = d2pca_SA4(44,sa4) 
    colnames = trainingdf2.columns[selected_indices]
    trainingdf2 = trainingdf2[colnames] 
    l1 = pd.DataFrame(df_l_new[sa4].groupby(df_l_new.index.year).mean()).iloc[Start_Year-2020:,:]
    l1.index = l1_index
    l1  = l1.rename_axis('date')
    df_combined = pd.concat([trainingdf2 ,l1],axis = 1)
    df = df_combined.dropna()
    df = df.reset_index(drop = True)
    train_data = df
    model_VAR  = VAR(train_data)
    results = model_VAR.fit(maxlags = 1)
    Dict_VARpara_SA4[sa4] = results.params

# %%
Dict_std_l1 = {}
for sa4 in DataFrameDict_Hsa4.keys():
     l1 = pd.DataFrame(df_l_new[sa4].groupby(df_l_new.index.year).mean()).iloc[Start_Year-2020:,:]
     Dict_std_l1[sa4] = l1.std(ddof = 0)[0]



# %%



# pc = '0810'
# sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
# traningY = df_l.loc['2006-01':'2020-12'][pc]


# trainingdf1 = PC_l0.loc['2006-01':'2020-12'].reset_index(drop=True)

# trainingdf2 = d2pca_SA4(44,sa4) 
# colnames = trainingdf2.columns[selected_indices]
# trainingdf2 = trainingdf2[colnames] 
# df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
# trainingdf2_n = df_repeated.reset_index(drop=True)

# Y = traningY.values
# X = pd.concat([trainingdf1,trainingdf2_n], axis=1).values
# X_train = sm.add_constant(X)
# model_full = sm.OLS(Y,X_train)
# reg_full = model_full.fit()
# para_linear = reg_full.params


# para_linear_nat = np.append(para_linear[1:7],1)
# para_linear_sa4 = np.append(para_linear[7:12],1)

# para_xpan_nat = np.dot(para_VAR_nat,para_linear_nat)[1:-1] 
# para_xpan_sa4 = np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[1:-1] 

# varible_coef_nat = np.dot(para_xpan_nat.reshape(-1,1).T, df_loadings_nat)

# varible_coef_sa4 = np.dot(para_xpan_sa4.reshape(-1,1).T, df_loadings)

# const  = para_linear[0] +  np.dot(para_VAR_nat,para_linear_nat)[0]  + np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[0] 

# l0_coef = np.dot(para_VAR_nat,para_linear_nat)[-1]

# l1_coef =  np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[-1] 


# %%
l0_std = df_l['l0'].loc['2006-01':'2020-12'].std(ddof = 0)
def pc2para(pc):
    sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
    traningY = df_l.loc['2006-01':'2020-12'][pc]

    trainingdf1 = PC_l0.loc['2006-01':'2020-12'].reset_index(drop=True)

    trainingdf2 = d2pca_SA4(44,sa4) 
    colnames = trainingdf2.columns[selected_indices]
    trainingdf2 = trainingdf2[colnames] 
    df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
    trainingdf2_n = df_repeated.reset_index(drop=True)

    Y = traningY.values
    X = pd.concat([trainingdf1,trainingdf2_n], axis=1).values
    X_train = sm.add_constant(X)
    model_full = sm.OLS(Y,X_train)
    reg_full = model_full.fit()
    para_linear = reg_full.params

    para_linear_nat = np.append(para_linear[1:7],1)
    para_linear_sa4 = np.append(para_linear[7:12],1)

    para_xpan_nat = np.dot(para_VAR_nat,para_linear_nat)[1:-1] 
    para_xpan_sa4 = np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[1:-1] 

    # varible_coef_nat = np.dot(para_xpan_nat.reshape(-1,1).T, df_loadings_nat)
    varible_coef_nat = para_xpan_nat
    # varible_coef_sa4 = np.dot(para_xpan_sa4.reshape(-1,1).T, df_loadings)
    varible_coef_sa4  = para_xpan_sa4

    const  = para_linear[0] +  np.dot(para_VAR_nat,para_linear_nat)[0]  + np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[0] 

    l0_coef = np.dot(para_VAR_nat,para_linear_nat)[-1]*l0_std 

    l1_coef =  np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[-1] * Dict_std_l1[sa4]

    return const,l0_coef,l1_coef, varible_coef_nat, varible_coef_sa4

# %%
const_arr = []
l0_coef_arr = []
l1_coef_arr = []
variable_coef_nat_arr = [] 
variable_coef_sa4_arr = []

for pc_ in listpostcode:
    const_,l0_coef_,l1_coef_, variable_coef_nat_, variable_coef_sa4_ = pc2para(pc_)
    const_arr.append(const_)
    l0_coef_arr.append(l0_coef_)
    l1_coef_arr.append(l1_coef_)
    variable_coef_nat_arr.append(variable_coef_nat_)
    variable_coef_sa4_arr.append(variable_coef_sa4_)



# %%
# create a sample array
# arr =np.array(variable_coef_sa4_arr)

# max_idx = np.argmax(arr, axis=1)

# # count the occurrences of each column index
# counts = np.bincount(max_idx, minlength=arr.shape[1])

# # sort the column indices by their counts
# sorted_idx = np.argsort(counts)[::-1]

# # print the sorted column indices and their counts
# for i, idx in enumerate(sorted_idx):
#     count = counts[idx]
#     print(f"Rank {i+1}: Column {idx}, count = {count}")





# %%
postcodecsv = pd.read_csv("au_postcodes.csv")

from statistics import mean, median


ndf = pd.read_json("POA_2016_AUST.json")
ndf1 = ndf["features"].apply(pd.Series)
ndf2 = ndf1["geometry"].apply(pd.Series)
ndf3 = ndf1["properties"].apply(pd.Series)
border = ndf2["coordinates"]

depth = lambda L: isinstance(L, list) and max(map(depth, L)) + 1


def flatten(d, l=1):
    for i in d:
        yield from ([i] if l == 1 else flatten(i, l - 1))


def centerlocation(index):
    tmplst = list(flatten(border[index], l=depth(border[index])))
    longi = median([i for i in tmplst if i > 0])
    lati = median([i for i in tmplst if i < 0])
    return [longi, lati]


postcodedf = []
i = 0
for i in range(len(ndf)):
    if depth(border[i]) > 0:
        tpd = {
            "postcode": [ndf3["POA_CODE16"].iloc[i]],
            "longitude": [centerlocation(i)[0]],
            "latitude": [centerlocation(i)[1]],
        }
        tpdf = pd.DataFrame(tpd)
        postcodedf.append(tpdf)
    else:
        print(i)
        continue
    i = i + 1
postcodedf = pd.concat(postcodedf, ignore_index=True)

ndf = pd.read_json("AUS_2021_AUST_GDA2020.json")
ndf1 = ndf["features"].apply(pd.Series)
ndf2 = ndf1["geometry"].apply(pd.Series)
ndf3 = ndf1["properties"].apply(pd.Series)

boundary = []
for index in range(0,len(ndf2['coordinates'][0])):
    tmplst = list(flatten(ndf2['coordinates'][0][index], l=depth(ndf2['coordinates'][0][index])))
    boundary = boundary+tmplst 
df_boundary = pd.DataFrame(np.reshape(boundary ,(96,2)), columns = ['longitude','latitude']) 


#%%
postcodename = postcodecsv[['postcode','place_name']]
postcodename['postcode'] = postcodename['postcode'].astype('str')


#%%

df_coef_con = pd.DataFrame({'postcode': np.array(listpostcode), 'constant': const_arr})

df_con_map = pd.merge(postcodedf ,df_coef_con, on='postcode')


df_df1_map1 = df_con_map[["latitude","longitude","constant"]]

length = len(df_df1_map1)
X = df_df1_map1[['longitude','latitude']]
y = df_df1_map1['constant']
regressor = LinearRegression()  
regressor.fit(X, y) #training the algorithm
X_test = df_boundary[['longitude','latitude']]
y_pred = regressor.predict(X_test)
df_boundary['constant'] = y_pred 
df_df1_map2 = pd.concat([df_df1_map1, df_boundary], ignore_index=True, sort=False)


temp_mean = mean(df_df1_map2.constant)
temp_std  = np.std(df_df1_map2.constant)
debug     = False

# Setup colormap
colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
vmin   = temp_mean - 2 * temp_std
vmax   = temp_mean + 2 * temp_std
levels = len(colors)
cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)

x_orig = np.asarray(df_df1_map2.longitude.tolist())
y_orig = np.asarray(df_df1_map2.latitude.tolist())
z_orig = np.asarray(df_df1_map2.constant.tolist())

# Make a grid
x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 2000)
y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 2000)
x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
# sigma = [5, 5]
# z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')
path = Path(np.reshape(boundary ,(96,2)))
patch = PathPatch(path, facecolor ='none',edgecolor='none')

fig, ax = plt.subplots() 
ax.add_patch(patch)

im = ax.imshow(z_mesh, interpolation ='nearest',
               origin ='lower', extent =[112, 155, -45, -10],
               clip_path = patch, clip_on = True)
im.set_clip_path(patch)

plt.colorbar(im)


plt.xticks([])
plt.yticks([])
plt.show()



#%%
df_con_map_od = df_con_map.sort_values(by='constant', ascending=False)


# create two new dataframes containing the first and last ten rows of the sorted dataframe
df_top10 = df_con_map_od.head(10).reset_index(drop = True)
df_bottom10 = df_con_map_od.tail(10).reset_index(drop = True)


#%%
df_top10 = pd.merge( df_top10,postcodename.groupby('postcode')['place_name'].agg(list), on='postcode')
df_top10['constant'] = df_top10['constant'].apply(lambda x: x * 1000)
df_top10['longitude'] = df_top10['longitude'].apply(lambda x: '{:.2f}'.format(x))
df_top10['latitude'] = df_top10['latitude'].apply(lambda x: '{:.2f}'.format(x))
df_top10['constant'] = df_top10['constant'].map('{:.2f}'.format)

print(df_top10.to_latex(index = False))

#%%
df_bottom_10 =  pd.merge( df_bottom10,postcodename.groupby('postcode')['place_name'].agg(list), on='postcode')
df_bottom_10['constant'] = df_bottom_10['constant'].apply(lambda x: x * 1000)
# format the 'longitude', 'latitude', and 'constant' columns
df_bottom_10['longitude'] = df_bottom_10['longitude'].apply(lambda x: '{:.2f}'.format(x))
df_bottom_10['latitude'] = df_bottom_10['latitude'].apply(lambda x: '{:.2f}'.format(x))
df_bottom_10['constant'] = df_bottom_10['constant'].map('{:.2f}'.format)
print(df_bottom_10.to_latex(index = False))
#%%
# df_loadings_s = df_loadings_s.applymap(lambda x: x + np.random.normal(loc=0, scale=0.001))
# %%
df_coef_con = pd.DataFrame({'postcode': np.array(listpostcode), 'constant': l0_coef_arr})

df_con_map = pd.merge(postcodedf ,df_coef_con, on='postcode')


df_df1_map1 = df_con_map[["latitude","longitude","constant"]]

length = len(df_df1_map1)
X = df_df1_map1[['longitude','latitude']]
y = df_df1_map1['constant']
regressor = LinearRegression()  
regressor.fit(X, y) #training the algorithm
X_test = df_boundary[['longitude','latitude']]
y_pred = regressor.predict(X_test)
df_boundary['constant'] = y_pred 
df_df1_map2 = pd.concat([df_df1_map1, df_boundary], ignore_index=True, sort=False)


temp_mean = mean(df_df1_map2.constant)
temp_std  = np.std(df_df1_map2.constant)
debug     = False

# Setup colormap
colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
vmin   = temp_mean - 2 * temp_std
vmax   = temp_mean + 2 * temp_std
levels = len(colors)
cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)

x_orig = np.asarray(df_df1_map2.longitude.tolist())
y_orig = np.asarray(df_df1_map2.latitude.tolist())
z_orig = np.asarray(df_df1_map2.constant.tolist())

# Make a grid
x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 2000)
y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 2000)
x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)

z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
# sigma = [5, 5]
# z_mesh = sp.ndimage.filters.gaussian_filter(z_mesh, sigma, mode='constant')
path = Path(np.reshape(boundary ,(96,2)))
patch = PathPatch(path, facecolor ='none',edgecolor='none')

fig, ax = plt.subplots() 
ax.add_patch(patch)

im = ax.imshow(z_mesh, interpolation ='nearest',
               origin ='lower', extent =[112, 155, -45, -10],
               clip_path = patch, clip_on = True)
im.set_clip_path(patch)

plt.colorbar(im)


plt.xticks([])
plt.yticks([])
plt.show()


# %%
df_con_map_od = df_con_map.sort_values(by='constant', ascending=False)


# create two new dataframes containing the first and last ten rows of the sorted dataframe
df_top10 = df_con_map_od.head(10).reset_index(drop = True)
df_bottom10 = df_con_map_od.tail(10).reset_index(drop = True)


df_top10 = pd.merge( df_top10,postcodename.groupby('postcode')['place_name'].agg(list), on='postcode')
df_top10['constant'] = df_top10['constant'].apply(lambda x: x * 1000)
df_top10['longitude'] = df_top10['longitude'].apply(lambda x: '{:.2f}'.format(x))
df_top10['latitude'] = df_top10['latitude'].apply(lambda x: '{:.2f}'.format(x))
df_top10['constant'] = df_top10['constant'].map('{:.2f}'.format)

print(df_top10.to_latex(index = False))

df_bottom_10 =  pd.merge( df_bottom10,postcodename.groupby('postcode')['place_name'].agg(list), on='postcode')
df_bottom_10['constant'] = df_bottom_10['constant'].apply(lambda x: x * 1000)
# format the 'longitude', 'latitude', and 'constant' columns
df_bottom_10['longitude'] = df_bottom_10['longitude'].apply(lambda x: '{:.2f}'.format(x))
df_bottom_10['latitude'] = df_bottom_10['latitude'].apply(lambda x: '{:.2f}'.format(x))
df_bottom_10['constant'] = df_bottom_10['constant'].map('{:.2f}'.format)
print(df_bottom_10.to_latex(index = False))
# %%
arr1, arr2, arr3, arr4, arr5, arr6 = np.split(np.array(variable_coef_nat_arr), 6, axis=1)

#%%
arrays = [arr1, arr2, arr3, arr4, arr5, arr6]

fig, axs = plt.subplots(3, 2, figsize=(10, 10)) # create a 3x2 subplot layout
axs = axs.flatten() # flatten the axs array to be able to iterate through it

# iterate through the arrays and plot the data on the corresponding subplot
for i, arr in enumerate(arrays):
    new_arr = arr.flatten()
    df_coef_con = pd.DataFrame({'postcode': np.array(listpostcode), 'constant': new_arr})
    df_con_map = pd.merge(postcodedf ,df_coef_con, on='postcode')
    df_df1_map1 = df_con_map[["latitude","longitude","constant"]]
    length = len(df_df1_map1)
    X = df_df1_map1[['longitude','latitude']]
    y = df_df1_map1['constant']
    regressor = LinearRegression()  
    regressor.fit(X, y) #training the algorithm
    X_test = df_boundary[['longitude','latitude']]
    y_pred = regressor.predict(X_test)
    df_boundary['constant'] = y_pred 
    df_df1_map2 = pd.concat([df_df1_map1, df_boundary], ignore_index=True, sort=False)
    temp_mean = mean(df_df1_map2.constant)
    temp_std  = np.std(df_df1_map2.constant)
    debug     = False
    # Setup colormap
    colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
    vmin   = temp_mean - 2 * temp_std
    vmax   = temp_mean + 2 * temp_std
    levels = len(colors)
    cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)
    x_orig = np.asarray(df_df1_map2.longitude.tolist())
    y_orig = np.asarray(df_df1_map2.latitude.tolist())
    z_orig = np.asarray(df_df1_map2.constant.tolist())
    # Make a grid
    x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 2000)
    y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 2000)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
    path = Path(np.reshape(boundary ,(96,2)))
    patch = PathPatch(path, facecolor ='none',edgecolor='none')
    axs[i].add_patch(patch)
    im = axs[i].imshow(z_mesh, interpolation ='nearest',
                   origin ='lower', extent =[112, 155, -45, -10],
                   clip_path = patch, clip_on = True)
    im.set_clip_path(patch)
    plt.colorbar(im, ax=axs[i])
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(f"Risk Factor {i+1}") # add the caption to the subplot
    
plt.show()
# %%
SA4_rf_Dict = {}
for sa4 in DataFrameDict_Hsa4.keys():
    SA4_rf_df = d2pca_SA4(44,sa4).iloc[:,selected_indices] 
    SA4_rf_Dict[sa4] =  SA4_rf_df 

# %%
new_dict = {}

# iterate through the keys and data frames in the original dictionary
for key, df in SA4_rf_Dict.items():
    
    # iterate through the columns in each data frame
    for col in df.columns:
        
        # if the column is already a key in the new dictionary, append the column to the data frame
        if col in new_dict:
            new_dict[col][key] = df[col]
        
        # otherwise, create a new data frame with the column as the key and append it to the new dictionary
        else:
            new_dict[col] = pd.DataFrame({key: df[col]})

# display the new dictionary

RF_sa4_dict  = new_dict.copy()
 
# %%
import warnings
from tslearn.clustering import TimeSeriesKMeans
from matplotlib.pyplot import figure
random.seed(123)
warnings.filterwarnings("ignore", category=FutureWarning) 
# %%
# sim_times = 100
# k = 3
# Dict_clu_PC_map = {}
# rdf_centroids = pd.DataFrame()
# rdf_clu_PC = pd.DataFrame()
# www = 0

# figure(figsize = (40, 20), dpi = 180)

# for pc in RF_sa4_dict.keys():
#     data_array = RF_sa4_dict[pc].T.values
#     X_train = data_array
#     for i in range(0,sim_times):
#         model = TimeSeriesKMeans(n_clusters=k, metric="softdtw", max_iter=10)
#         model.fit(data_array)
#         y_pred = model.predict(data_array)
#         rdf_clu_PC[f's{i+1}'] = y_pred
#         for yi in range(k):
#             rdf_centroids[f'c{yi}_s{i+1}'] = model.cluster_centers_[yi].ravel()        
#     df_clu_PC = rdf_clu_PC.copy()
#     df_centroids= rdf_centroids.copy()
#     data_array_cen = np.array(df_centroids.T.values)
#     X_train_cen = data_array_cen
#     model_cen = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10)
#     model_cen.fit(data_array_cen)
#     y_pred_cen = model_cen.predict(data_array_cen)
#     y_pred_cen_rs = y_pred_cen.reshape(-1,k)
#     df_clu_PC_map = pd.DataFrame()
#     for i in range(0,sim_times):
#         arr_pred_cen_rs = y_pred_cen_rs[i]
#         def mapping(ele):
#             return arr_pred_cen_rs[ele]
#         df_clu_PC_map[f'sim{i+1}'] =list(map(mapping,df_clu_PC.iloc[:,i].values))
#     Dict_clu_PC_map[pc] = df_clu_PC_map 
#     sz = X_train_cen.shape[1]

# for www in range(0,5):
#     for yi in range(k):
#             plt.subplot(5,k, www*3+yi+1)
#             for xx in X_train_cen[y_pred_cen == yi]:
#                 plt.plot(xx.ravel(), "k-", alpha=.2)
#                 plt.text(0.55, 0.85,'Cluster %d' % (yi + 1), fontsize=18,transform=plt.gca().transAxes)
#                 if yi == 1:
#                     plt.title(f"PC{www+1}",fontsize = 1) 
#             plt.plot(model_cen.cluster_centers_[yi].ravel(), "r-")
#             plt.xlim(0, sz)
#             plt.ylim(-4, 4)
#     www = www+1
# plt.subplots_adjust(hspace=0.5)        
# plt.show()

#%%
unique_postcodes = listpostcode
pc_sa4_df = pd.DataFrame({})
pc_sa4_df['postcode'] =  listpostcode
# create an empty list to store the sa4 values
sa4_list = []

# loop through each postcode and find the corresponding sa4 value from aveHPIdf_lv3
for pc in unique_postcodes:
    sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
    sa4_list.append(sa4)

# add the sa4 column to the DataFrame
pc_sa4_df['sa4'] = sa4_list

# %%
RF_sa4_dict_keys = list(RF_sa4_dict.keys())
i = 0 
RF_sa4_dict[RF_sa4_dict_keys[i]]

arr1, arr2, arr3, arr4, arr5 = np.split(np.array(variable_coef_sa4_arr), 5, axis=1)
arrays = [arr1, arr2, arr3, arr4, arr5]
arr = arrays[i]
new_arr = arr.flatten()
df_coef_con = pd.DataFrame({'postcode': np.array(listpostcode), 'constant': new_arr})
df_con_map = pd.merge(postcodedf ,df_coef_con, on='postcode')

df_con_map = pd.merge(df_con_map ,pc_sa4_df, on='postcode')

df_sa4_sum = pd.DataFrame(RF_sa4_dict[RF_sa4_dict_keys[i]].sum(axis = 0))
df_sa4_sum = df_sa4_sum.reset_index()
df_sa4_sum.columns = ['sa4','sum']

# %%
start_date = '2006-01-01'
end_date = '2020-12-31'
date_range = pd.date_range(start=start_date, end=end_date, freq='M')


# %%
def pc_contri(pc):


    l0_std = df_l['l0'].loc['2006-01':'2020-12'].std(ddof = 0)

    sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
    traningY = df_l.loc['2006-01':'2020-12'][pc]

    trainingdf1 = PC_l0.loc['2006-01':'2020-12'].reset_index(drop=True)

    trainingdf2 = d2pca_SA4(44,sa4) 
    colnames = trainingdf2.columns[selected_indices]
    trainingdf2 = trainingdf2[colnames] 
    df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
    trainingdf2_n = df_repeated.reset_index(drop=True)

    Y = traningY.values
    X = pd.concat([trainingdf1,trainingdf2_n], axis=1).values
    X_train = sm.add_constant(X)
    model_full = sm.OLS(Y,X_train)
    reg_full = model_full.fit()
    para_linear = reg_full.params

    para_linear_nat = np.append(para_linear[1:7],1)
    para_linear_sa4 = np.append(para_linear[7:12],1)

    para_xpan_nat = np.dot(para_VAR_nat,para_linear_nat)[1:-1] 
    para_xpan_sa4 = np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[1:-1] 

    # varible_coef_nat = np.dot(para_xpan_nat.reshape(-1,1).T, df_loadings_nat)
    varible_coef_nat = para_xpan_nat
    # varible_coef_sa4 = np.dot(para_xpan_sa4.reshape(-1,1).T, df_loadings)
    varible_coef_sa4  = para_xpan_sa4

    const  = para_linear[0] +  np.dot(para_VAR_nat,para_linear_nat)[0]  + np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[0] 

    l0_coef = np.dot(para_VAR_nat,para_linear_nat)[-1]

    l1_coef =  np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[-1]

    contri_df = pd.DataFrame({}, index=date_range)
    contri_df['constant'] = np.repeat(const, 180)
    contri_df['l0'] = (df_l['l0'].loc['2006-01':'2020-12'] * l0_coef).values
    contri_df[['RF1','RF2','RF3','RF4','RF5','RF6']]  = (trainingdf1 * varible_coef_nat).values
    contri_df['l1'] = (traningY * l1_coef).values
    contri_df[['RF1_sa4','RF2_sa4','RF3_sa4','RF4_sa4','RF5_sa4']]= (trainingdf2_n * varible_coef_sa4 ).values
    contri_df_by_year = contri_df.resample('Y').sum()
    #contri_df_by_year = contri_df_by_year.drop(contri_df_by_year.index[-1])
    # row_sums = contri_df_by_year.sum(axis=1)
    avg_l0_con = contri_df_by_year ['l0'].values
    contri_df_by_year_pct = contri_df_by_year.apply(lambda x: (x /avg_l0_con ) * 100, axis=0)
    mean = contri_df_by_year_pct.mean(axis = 0)
    return mean[-6],mean[-5],mean[-4],mean[-3],mean[-2],mean[-1]
# %%

l1_arr = [] 
rfsa4_1_arr = []
rfsa4_2_arr = []
rfsa4_3_arr = []
rfsa4_4_arr = []
rfsa4_5_arr = []

for pc_ in listpostcode:
    l1_, rfsa4_1_, rfsa4_2_, rfsa4_3_, rfsa4_4_, rfsa4_5_=  pc_contri(pc_)
    l1_arr.append(l1_)
    rfsa4_1_arr.append(rfsa4_1_)
    rfsa4_2_arr.append(rfsa4_2_)
    rfsa4_3_arr.append(rfsa4_3_)
    rfsa4_4_arr.append(rfsa4_4_)
    rfsa4_5_arr.append(rfsa4_5_)


# %%
rfnamesa4 = ['$l_j$','RF1','RF2','RF3','RF4','RF5']
# %%
from matplotlib.ticker import FuncFormatter
def percent_formatter(x, pos):
    return '{:.0f}%'.format(x)
arrayss =[l1_arr ,rfsa4_1_arr ,rfsa4_2_arr ,rfsa4_3_arr ,rfsa4_4_arr ,rfsa4_5_arr]

fig, axs = plt.subplots(3, 2, figsize=(10, 10)) # create a 3x2 subplot layout
axs = axs.flatten() # flatten the axs array to be able to iterate through it

# iterate through the arrays and plot the data on the corresponding subplot
for i, arr in enumerate(arrayss):
    new_arr = arr
    df_coef_con = pd.DataFrame({'postcode': np.array(listpostcode), 'constant': new_arr})
    df_con_map = pd.merge(postcodedf ,df_coef_con, on='postcode')
 

    # Calculate the mean and standard deviation of the constant column
    mean = df_con_map['constant'].mean()
    std = df_con_map['constant'].std()

    # Define the threshold for extreme values
    threshold1 = mean + 3 * std
    threshold2 = mean - 3 * std

    # Delete rows where the value in the constant column exceeds the threshold
    df_con_map = df_con_map[df_con_map['constant'] < threshold1]
    df_con_map = df_con_map[df_con_map['constant'] > threshold2]

    df_df1_map1 = df_con_map[["latitude","longitude","constant"]]
    

    length = len(df_df1_map1)
    X = df_df1_map1[['longitude','latitude']]
    y = df_df1_map1['constant']
    regressor = LinearRegression()  
    regressor.fit(X, y) #training the algorithm
    X_test = df_boundary[['longitude','latitude']]
    y_pred = regressor.predict(X_test)
    df_boundary['constant'] = y_pred 
    df_df1_map2 = pd.concat([df_df1_map1, df_boundary], ignore_index=True, sort=False)
    temp_mean = np.mean(df_df1_map2.constant)
    temp_std  = np.std(df_df1_map2.constant)
    debug     = False
    # Setup colormap
    colors = ['#d7191c',  '#fdae61',  '#ffffbf',  '#abdda4',  '#2b83ba']
    vmin   = temp_mean - 2 * temp_std
    vmax   = temp_mean + 2 * temp_std
    levels = len(colors)
    cm     = branca.colormap.LinearColormap(colors, vmin=vmin, vmax=vmax).to_step(levels)
    x_orig = np.asarray(df_df1_map2.longitude.tolist())
    y_orig = np.asarray(df_df1_map2.latitude.tolist())
    z_orig = np.asarray(df_df1_map2.constant.tolist())
    # Make a grid
    x_arr          = np.linspace(np.min(x_orig), np.max(x_orig), 2000)
    y_arr          = np.linspace(np.min(y_orig), np.max(y_orig), 2000)
    x_mesh, y_mesh = np.meshgrid(x_arr, y_arr)
    z_mesh = griddata((x_orig, y_orig), z_orig, (x_mesh, y_mesh), method='linear')
    path = Path(np.reshape(boundary ,(96,2)))
    patch = PathPatch(path, facecolor ='none',edgecolor='none')
    axs[i].add_patch(patch)
    im = axs[i].imshow(z_mesh, interpolation ='nearest',
                   origin ='lower', extent =[112, 155, -45, -10],
                   clip_path = patch, clip_on = True)
    im.set_clip_path(patch)
    plt.colorbar(im, ax=axs[i],format=FuncFormatter(percent_formatter))
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title(rfnamesa4[i] ) # add the caption to the subplot
    
plt.show()
# %%
def pc2cumsum(pc):
    sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
    traningY = df_l.loc['2006-01':'2020-12'][pc]

    trainingdf1 = PC_l0.loc['2006-01':'2020-12'].reset_index(drop=True)

    trainingdf2 = d2pca_SA4(44,sa4) 
    colnames = trainingdf2.columns[selected_indices]
    trainingdf2 = trainingdf2[colnames] 
    df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
    trainingdf2_n = df_repeated.reset_index(drop=True)

    Y = traningY.values
    X = pd.concat([trainingdf1,trainingdf2_n], axis=1).values
    X_train = sm.add_constant(X)
    model_full = sm.OLS(Y,X_train)
    reg_full = model_full.fit()
    para_linear = reg_full.params

    para_linear_nat = np.append(para_linear[1:7],1)
    para_linear_sa4 = np.append(para_linear[7:12],1)

    para_xpan_nat = np.dot(para_VAR_nat,para_linear_nat)[1:-1] 
    para_xpan_sa4 = np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[1:-1] 

    # varible_coef_nat = np.dot(para_xpan_nat.reshape(-1,1).T, df_loadings_nat)
    varible_coef_nat = para_xpan_nat
    # varible_coef_sa4 = np.dot(para_xpan_sa4.reshape(-1,1).T, df_loadings)
    varible_coef_sa4  = para_xpan_sa4

    const  = para_linear[0] +  np.dot(para_VAR_nat,para_linear_nat)[0]  + np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[0] 

    l0_coef = np.dot(para_VAR_nat,para_linear_nat)[-1]

    l1_coef =  np.dot(Dict_VARpara_SA4[sa4],para_linear_sa4)[-1]

    contri_df = pd.DataFrame({}, index=date_range)
    contri_df['constant'] = np.repeat(const, 180)
    contri_df['l0'] = (df_l['l0'].loc['2006-01':'2020-12'] * l0_coef).values
    contri_df[['RF1','RF2','RF3','RF4','RF5','RF6']]  = (trainingdf1 * varible_coef_nat).values
    contri_df['l1'] = (traningY * l1_coef).values
    # contri_df[['RF1_sa4','RF2_sa4','RF3_sa4','RF4_sa4','RF5_sa4']]= (trainingdf2_n * varible_coef_sa4 ).values
    df = trainingdf2_n * varible_coef_sa4 
    for i in range(len(df)):
        df.iloc[i] = df.iloc[i]/ ((i+1))
    contri_df[['RF1_sa4','RF2_sa4','RF3_sa4','RF4_sa4','RF5_sa4']] = df.values
    contri_df_by_year = contri_df.resample('Y').sum()
    mean_df = contri_df_by_year.cumsum()
    for i in range(len(mean_df)):
        mean_df.iloc[i] = mean_df.iloc[i] / (i+1)
    return mean_df

# %%
Dict_mean_df = {}
for pc_ in listpostcode:
    Dict_mean_df[pc_] = pc2cumsum(pc_)

# %%
fig, axs = plt.subplots(7, 2, figsize=(12, 18))
mean_names = ['Constant', '$l_0$','National RF1','National RF2','National RF3','National RF4','National RF5','National RF6','$l_j$','SA4 RF1','SA4 RF2','SA4 RF3','SA4 RF4','SA4 RF5' ]
# iterate over each column
for i in range(14):
    # get the row and column index of the subplot
    row = i // 2
    col = i % 2
    
    # iterate over each dataframe in the dictionary
    for key, df in Dict_mean_df.items():
        # plot the column as a line graph
        axs[row, col].plot(df.iloc[:, i])
    
    # set the title and legend
    axs[row, col].set_title(mean_names[i])
    axs[row, col].set_xticklabels([])
   
    
# adjust spacing and display the plot
fig.tight_layout()
plt.show()
# %%
