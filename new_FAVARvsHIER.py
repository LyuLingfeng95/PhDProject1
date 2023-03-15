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

#%% Choose the best order for FAVAR Model 
# for r in range(7, 26):
#     pca = PCA(n_components=r)
#     C = pca.fit_transform(data_s)
#     reg = LinearRegression().fit(data_s.iloc[:, :6], C)
#     F_hat = (C - data_s.iloc[:, :6].dot(reg.coef_.T)).values
#     F_hat = pd.DataFrame(F_hat, columns=[f'Latent_{i}' for i in range(1, r+1)])
#     data_var = pd.concat([F_hat, data_s.iloc[:, :6]], axis=1)
#     X = data_var.values
#     AIC = 0  
#     BIC = 0 
#     for i in range(1, 465):
#         Y = observations.iloc[:, i].values
#         X_new = sm.add_constant(X)
#         model2 = sm.OLS(Y,X_new)
#         reg2 = model2.fit()
#         AIC += reg2.aic
#         BIC += reg2.bic
#     AICavg =  AIC/464
#     BICavg = BIC/464
#     df_cr = df_cr.append({"N": r-5, "AIC": AICavg, "BIC": BICavg}, ignore_index=True)

# df_cr.loc[0, :] = [1, -1280.382, -1228.971]

# plt.plot(df_cr["N"], df_cr["AIC"], color="green", label="AIC")
# plt.plot(df_cr["N"], df_cr["BIC"], color="red", label="BIC")
# plt.xlabel("K")
# plt.ylabel("Information Criteria")
# plt.legend()






#%% Read the data set: 
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

aveHPIdf_lv3 = aveHPIdf_lv3[aveHPIdf_lv3['date'] > '2009-12-31']

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
date_range = pd.date_range('2010-01-31', '2021-05-31', freq='M')
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
date_range = pd.date_range('2010-01-31', '2021-05-31', freq='M')
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
date_range = pd.date_range('2010-01-31', '2021-05-31', freq='M')
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



#%%
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
#%%
threshold = 0.01

def selectStationaySeries(variable_tar):
        if adfuller(variable_tar.dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar
            suffix =  "original"
        elif adfuller(variable_tar.pct_change(1).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(1).dropna(how="all")
            suffix = "1storderdiff"
        elif adfuller(variable_tar.pct_change(3).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(3).dropna(how="all")
            suffix = "seasonaldiff"
        elif adfuller(variable_tar.pct_change(12).dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(12).dropna(how="all")
            suffix = "annualdiff"     
        elif adfuller(variable_tar.pct_change(1).diff().dropna(how="all"))[1] < threshold:
            stationary_variable = variable_tar.pct_change(1).diff().dropna(how="all")
            suffix = "2ndorderdiff"
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
PCnames = []
for i in range(1, d + 1, 1):
    PCnames.append(f"PC{i}")
PC_nat = pd.DataFrame(X_pca_d, columns=PCnames)

PC_nat['date'] = macdata['date'].values


#%%
observations = avg_3_pivot
knownvariables = PC_nat.loc[PC_nat['date']>'2009-12']
knownvariables = knownvariables.drop('date', axis=1).reset_index(drop = True)

data = pd.concat([knownvariables, observations], axis=1)
data_s = (data - data.mean()) / data.std() # standardize the data


#%%
def FAVAR_rfs(r):
    pca = PCA(n_components=r)
    C = pca.fit_transform(data_s)
    reg = LinearRegression().fit(data_s.iloc[:, :6], C)
    F_hat = (C - data_s.iloc[:, :6].dot(reg.coef_.T)).values
    F_hat = pd.DataFrame(F_hat, columns=[f'Latent_{k}' for k in range(1, r+1)])
    data_var = pd.concat([F_hat, data_s.iloc[:, :6]], axis=1)
    return data_var 
# %% Using rolling window to generate the errors 
def Forecasts_FAVAR(r,i):
    Data = FAVAR_rfs(r) 
    train_data = Data.iloc[i-120:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    forecast_values = results.forecast(train_data.values[-1:], steps=137-i)
    results = pd.DataFrame({})
    for postcode in observations.columns:
        Y = observations.iloc[i-120:i, :][postcode].values
        X = train_data.values
        X_train = sm.add_constant(X)
        model2 = sm.OLS(Y,X_train)
        reg_2 = model2.fit()
        X_test = sm.add_constant(forecast_values) 
        if forecast_values.shape[0]==1:
            X_test = np.insert(forecast_values, 0, 1, axis=1)    
        predicted_FAVAR = reg_2.predict(X_test)    
        Y_test = observations.iloc[i:, :][postcode].values
        errors = Y_test - predicted_FAVAR 
        if len(errors) < 17:
            errors = np.pad(errors, (0, 17-len(errors)), mode='constant')
        results[postcode] = errors
    return results

#%%
FAVAR_error_step_Dict = {}
for i in range(120,137):
    FAVAR_error_step_Dict[i] = Forecasts_FAVAR(10,i)


# %%
FAVAR_error_Dict = {}
for column in observations.columns:
    errors_df = pd.DataFrame({})
    for m in range(1, 18):
        error_values = FAVAR_error_step_Dict[m+120-1][column]
        errors_df[f'window{m}'] = error_values
    FAVAR_error_Dict[column] =  errors_df


# %% error_average_FAVAR
def error_average_FAVAR(Dict_Name, h, M):
    col_names = avg_3_pivot.columns
    dfs = []
    for col in col_names:
        try:
            df = Dict_Name[col]
            dfs.append(df)
        except KeyError:
            print(f"Warning: Could not find dataframe for key {col}.")


    # calculate RMSE for each dataframe and find average by row
    rmse_by_row = []
    for df in dfs:
        df_error = df.iloc[0: h+1, 0:len(df_dis)-M-h].copy()
        # calculate RMSE by row
        rmse_by_row.append(np.sqrt(np.mean((df_error.values)**2)))

    # calculate average of average RMSE by row
    if rmse_by_row:
        average_rmse = np.mean([np.mean(rmse) for rmse in rmse_by_row])
    else:
        average_rmse = np.nan

    return average_rmse
# %%

error_average_FAVAR(FAVAR_error_Dict, 1, 120)


#%% fitting_error
def Fitted_FAVAR(r,i):
    Data = FAVAR_rfs(r) 
    train_data = Data.iloc[i-120:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    forecast_values = results.fittedvalues
    errors = pd.DataFrame({})
    for postcode in observations.columns:
        Y = observations.iloc[i-120:i, :][postcode].values
        X = train_data.values
        X_train = sm.add_constant(X)
        model2 = sm.OLS(Y,X_train)
        reg_2 = model2.fit()
        X_test = sm.add_constant(forecast_values)   
        predicted_FAVAR = reg_2.predict(X_test).values
        Y_test = observations.iloc[i-120+1:i, :][postcode].values
        error = Y_test - predicted_FAVAR 
        errors[postcode] = error
    error =  np.sqrt(np.mean(np.mean(errors**2)))

    return error

fitted_error_array = []
for i in range(120,137):
    fitted_error_array.append(Fitted_FAVAR(10,i))

fitted_error_array_10  = np.mean(fitted_error_array)

fitted_error_array = []
for i in range(120,137):
    fitted_error_array.append(Fitted_FAVAR(6,i))

fitted_error_array_6  = np.mean(fitted_error_array)


# %%
with open('DataFrameDict_Hsa4.pickle', 'rb') as f:
    DataFrameDict_Hsa4 =  pickle.load(f)
# %%
with open('dfHILDA_cut.pickle', 'rb') as f:
      dfHILDA_cut =  pickle.load(f)  
# %%
with open('dfHILDA.pickle', 'rb') as f:
      dfHILDA =  pickle.load(f)    

# %%
threshold = 0.01
df = dfHILDA_cut.set_index("date")
df1 = df.resample("Y").mean().iloc[:, :]
vrst = pd.DataFrame([])

# need a dataframe to writedown the way to stabilize all variables
df_marks = pd.DataFrame([],columns=['Variable Names', 'Approach'])

#%%

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
# for vn in df1.columns.values:
    
#     variable_tar = df1[vn]
    
#     count1 = np.isinf(variable_tar.diff()).values.sum()
#     count2 = np.isinf(variable_tar.diff()).diff().values.sum()
    
#     if adfuller(variable_tar.interpolate("bfill").interpolate("ffill"))[1] < threshold:
#             stationary_variable = variable_tar.interpolate("bfill").interpolate("ffill")
#             suffix =  "original"
#             new_row = {'Variable Names': vn,'Approach': suffix}
#             df_marks = df_marks.append(new_row, ignore_index=True)
#             newseries = pd.DataFrame(stationary_variable)
#             vrst = pd.concat([vrst, newseries], axis=1)

    
#     elif count1 == 0:
#         try:
#             if adfuller(variable_tar.diff().interpolate("bfill").interpolate("ffill"))[1] < threshold:
#                 stationary_variable = variable_tar.diff().interpolate("bfill").interpolate("ffill")
#                 suffix = "1storderdiff"
#                 new_row = {'Variable Names': vn,'Approach': suffix}
#                 df_marks = df_marks.append(new_row, ignore_index=True)
#                 newseries = pd.DataFrame(stationary_variable)
#                 vrst = pd.concat([vrst, newseries], axis=1)
    
#             elif count2 == 0:
#                 try: 
#                     if adfuller(variable_tar.diff().diff().interpolate("bfill").interpolate("ffill"))[1] < threshold:
#                         stationary_variable = variable_tar.diff().diff().interpolate("bfill").interpolate("ffill")
#                         suffix = "2ndorderdiff"
#                         new_row = {'Variable Names': vn,'Approach': suffix}
#                         df_marks = df_marks.append(new_row, ignore_index=True)
#                         newseries = pd.DataFrame(stationary_variable)
#                         vrst = pd.concat([vrst, newseries], axis=1)
#                 except np.linalg.LinAlgError as e1:
#                     print('except:', e1)

#         except np.linalg.LinAlgError as e1:
#             print('except:', e1) 
    
#     else:
#         print('not found')
# %%
    
    
vrdata = vrst.reset_index()
    
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
pcadf['date'] = dfHILDA['date']  
pcadf["date"] = pd.to_datetime(pcadf["date"]) - pd.offsets.MonthEnd(6)
pcadf_sa4_date = pcadf

# %%
def selectStationaySeries3(variable_tar, vn, marks):
 
    marks1 =marks['Variable Names'].loc[marks['Approach'] == 'original'].values
    marks2 =marks['Variable Names'].loc[marks['Approach'] == "1storderdiff"].values
    marks3 =marks['Variable Names'].loc[marks['Approach'] == "2ndorderdiff"].values
    if vn in marks1:
            stationary_variable = variable_tar.interpolate("bfill").interpolate("ffill")
            suffix =  "original"
            #print(suffix)
            return(pd.DataFrame(stationary_variable))
    
    elif vn in marks2:
        try:
            if adfuller(variable_tar.diff().interpolate("bfill").interpolate("ffill"))[1] < threshold:
                stationary_variable = variable_tar.diff().interpolate("bfill").interpolate("ffill")
                suffix = "1storderdiff"
                #print(suffix)
                return(pd.DataFrame(stationary_variable))
        except np.linalg.LinAlgError as e1:
            print('except:', e1)
    
    elif vn in marks3:
        try: 
            if adfuller(variable_tar.diff().diff().dropna(how="all"))[1] < threshold:
                stationary_variable = variable_tar.diff().diff().dropna(how="all")
                suffix = "2ndorderdiff"
                #print(suffix)
                return(pd.DataFrame(stationary_variable))
        except np.linalg.LinAlgError as e1:
            print('except:', e1)  

# %%
def HILDAvarExt(key, marks):
    df = DataFrameDict_Hsa4[key].interpolate("bfill").interpolate("ffill")
    df = df.set_index("date")
    df1 = df.resample("Y").mean().iloc[:, :]
    vrst = pd.DataFrame([], columns = df_marks['Variable Names'].values)
    lst = list( set(df_marks['Variable Names'].values) & set(df1.columns.values[1:]))
    for vn in lst:
        vrst[vn] = selectStationaySeries3(df1[vn], vn, marks)
    vrdata = vrst.reset_index().fillna(value = 0) 
    return(vrdata)
# %% Respect your own work. 
def PCA_val(gamma1,marks,vrdata):
    tmp = pd.DataFrame([], columns = marks['Variable Names'].values)
    for key in DataFrameDict_Hsa4.keys(): 
        df = HILDAvarExt(key,marks)
        df1= df[vrdata.columns.tolist()]
        df2 = df1.drop(df1.columns[0], axis=1)
        
        vrdata1 = vrdata.drop(vrdata.columns[0],axis=1)
        vrdata2= vrdata1.apply(lambda x: x*gamma1)
        
        df_add = df2.add(vrdata2, fill_value=0)
        tmp = pd.concat([tmp, df_add],axis = 0)
        
    pca = PCA()
    X_train = tmp
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)

    pca.fit(X_train_std)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    d2 = np.argmax(cumsum >= 0.9) + 1

    pca_d_sa4 = PCA(n_components=d2)
    pca_d_sa4.fit(X_train_std)
    X_pca_d_sa4 = pca_d_sa4.transform(X_train_std)

    PCnames = []
    for i in range(1, d2 + 1, 1):
        PCnames.append(f"sa4PC{i}")
    
    pcadf = pd.DataFrame(X_pca_d_sa4, columns=PCnames)
    pcadf_sa4_date = pcadf
    return  d2, pca_d_sa4

# %%
PCnumbers = []
x = [i/10 for i in range(11)]
for gamma in x:
    gamma1 = gamma
    d2, pcafit = PCA_val(gamma1,df_marks,vrdata)
    PCnumbers.append(d2)
    print(gamma)
    
# %%
from scipy.interpolate import make_interp_spline, BSpline
# represents number of points to make between x.min and x.max
xnew =np.linspace(0, 1, 100)
spl = make_interp_spline(x, PCnumbers, k=1)  # type: BSpline
PCno_smooth = spl(xnew)

color1 = "#0085c3"
color2 = "#7ab800"
color3 = "#dc5034"

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
ax.step(x, PCnumbers,where='post',color = color3)
ax.grid(ls=":", color="gray", alpha=0.6)

plt.xlabel(r'$\gamma$')
plt.ylabel('Number of risk factors at the SA4s level')


plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)





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

#%%
df_l = transform_df(df_dis)
df_l.index = pd.to_datetime(df_l.index)
# Change the index to year-month format
df_l.index = df_l.index.to_period('M')
# %%

PC_l0 = PC_nat.set_index('date')
PC_l0.index = pd.to_datetime(PC_l0.index)
PC_l0.index = PC_l0.index.to_period('M')

df_combined = pd.concat([ PC_l0, df_l['l0']], axis=1)
# Create a boolean mask for rows to keep
mask = df_combined.index >= '2010-01-01'

# Filter the dataframe using the boolean mask
VAR_data_l0 = df_combined.loc[mask].reset_index(drop = True)
# %%
def VAR_data_l0_pre(i):
    train_data = VAR_data_l0.iloc[i-120:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    pre_values = results.fittedvalues
    return pre_values
# %%
def VAR_data_l0_for(i):
    train_data = VAR_data_l0.iloc[i-120:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    for_values = results.forecast(train_data.values[-1:], steps=132-i)
    return for_values

#%%
Dict_l0_pca = {}
for i in range(120,132):
    train_data = VAR_data_l0.iloc[i-120:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    pre_values = results.fittedvalues
    for_values = results.forecast(train_data.values[-1:], steps=132-i)
    Dict_l0_pca[i] = {}
    Dict_l0_pca[i]['Tdata'] = train_data
    Dict_l0_pca[i]['pre_values']  = pre_values
    Dict_l0_pca[i]['for_values']  =  for_values 


# %%
def d2pca_SA4(gamma1,key):
    df = HILDAvarExt(key,df_marks)
    df1= df[vrdata.columns.tolist()]
    df2 = df1.drop(df1.columns[0], axis=1)
    vrdata1 = vrdata.drop(vrdata.columns[0],axis=1)
    vrdata2 = vrdata1.apply(lambda x: x*gamma1)
    df_add = df2.add(vrdata2, fill_value=0)
    d2, pca_d_sa4 =  PCA_val(gamma1,df_marks,vrdata)

    pca = PCA()
    X_train = df_add 
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    
    pca_d_sa4.fit(X_train_std)
    X_pca_d_sa4 = pca_d_sa4.transform(X_train_std)

    PCnames = []
    for i in range(1, d2 + 1, 1):
        PCnames.append(f"sa4PC{i}")

    pcadf = pd.DataFrame(X_pca_d_sa4, columns=PCnames)
    pcadf_sa4_date = pcadf
    new_index = pd.date_range(start='2001', end='2021', freq='Y')
# Convert the index to the 'YYYY' format
    new_index = new_index.strftime('%Y')
# Assign the new index to the dataframe
    pcadf_sa4_date.index = new_index
    pcadf_sa4_date = pcadf_sa4_date.rename_axis('date')
    
    return pcadf_sa4_date

# %%
l1_index = pd.date_range(start='2010', end='2022', freq='Y')
l1_index = l1_index.strftime('%Y')

gamma1 = 40 

def VAR_data_l1_pre(i,d2,key):
    data  = d2pca_SA4(d2,key)
    l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
    l1.index = l1_index
    l1  = l1 .rename_axis('date')
    df_combined = pd.concat([data,l1],axis = 1)
    df = df_combined.dropna()
    df = df.reset_index(drop = True)
    train_data = df.iloc[i-10:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    pre_values = results.fittedvalues
    pre_values_index =  pd.date_range(start='2011', end='2020', freq='Y')
    pre_values_index =  pre_values_index.strftime('%Y')
    pre_values.index =  pre_values_index 
    return pre_values

def VAR_data_l1_for(i,d2,key):
    data  = d2pca_SA4(d2,key)
    l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
    l1.index = l1_index
    l1  = l1 .rename_axis('date')
    df_combined = pd.concat([data,l1],axis = 1)
    df = df_combined.dropna()
    df = df.reset_index(drop = True)
    train_data = df.iloc[i-10:i, :]
    model = VAR(train_data)
    results = model.fit(1)
    for_values = results.forecast(train_data.values[-1:], steps=11-i)
    for_values = pd.DataFrame(for_values)
    for_values_index =  pd.date_range(start='2020', end='2021', freq='Y')
    for_values_index =  for_values_index.strftime('%Y')
    for_values.index =  for_values_index
    return for_values

#%%
sa4_list  = list(avg_2_pivot.columns)
sa4_list_HILDA = list(DataFrameDict_Hsa4)
set_sa4_list = set(sa4_list)
set_sa4_list_HILDA = set(sa4_list_HILDA)

# Find the difference between the sets
diff = set_sa4_list.symmetric_difference(set_sa4_list_HILDA)

# Print the different elements and where they come from
print("Different elements:")
for element in diff:
    if element in set_sa4_list:
        print(f"{element} is in sa4_list but not in sa4_list_HILDA")
    else:
        print(f"{element} is in sa4_list_HILDA but not in sa4_list")

#%%
Dict_l1_pca = {}
for i in range(10,11):
    Dict_l1_pca[i] = {}
    for sa4 in DataFrameDict_Hsa4.keys():
        Dict_l1_pca[i][sa4] = {}
        data  = d2pca_SA4(d2,sa4)
        l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
        l1.index = l1_index
        l1  = l1 .rename_axis('date')
        df_combined = pd.concat([data,l1],axis = 1)
        df = df_combined.dropna()
        df = df.reset_index(drop = True)
        train_data = df.iloc[i-10:i, :]
        model = VAR(train_data)
        results = model.fit(1)

        pre_values = results.fittedvalues
        pre_values_index =  pd.date_range(start='2011', end='2020', freq='Y')
        pre_values_index =  pre_values_index.strftime('%Y')
        pre_values.index =  pre_values_index 

        for_values = results.forecast(train_data.values[-1:], steps=11-i)
        for_values = pd.DataFrame(for_values)
        for_values_index =  pd.date_range(start='2020', end='2021', freq='Y')
        for_values_index =  for_values_index.strftime('%Y')
        for_values.index =  for_values_index
        
        Dict_l1_pca[i][sa4]['Tdata'] = train_data
        Dict_l1_pca[i][sa4]['pre_values'] =  pre_values
        Dict_l1_pca[i][sa4]['for_values']  =  for_values
        Dict_l1_pca[i][sa4]['data'] = df


        l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
        l1.index = l1_index
        l1  = l1.rename_axis('date')
        df_combined = pd.concat([data,l1],axis = 1)
        df = df_combined.dropna()
        df = df.reset_index(drop = True)
        train_data = df
        model = VAR(train_data)
        results = model.fit(1)
        pre_values2 = results.fittedvalues
        pre_values_index2 =  pd.date_range(start='2011', end='2021', freq='Y')
        pre_values_index2 =  pre_values_index2.strftime('%Y')
        pre_values2.index =  pre_values_index2
        Dict_l1_pca[i][sa4]['pre_values2'] =  pre_values2

        


# %%

postcode_deleted = list(aveHPIdf_lv3.loc[aveHPIdf_lv3['sa4'] == 'Wide Bay']['postcode'].unique())
listpostcode = list(avg_3_pivot.columns)
listpostcode = [x for x in listpostcode if x not in postcode_deleted]
mse_arr_hier = []
for i in range(120,132):
    error_array = []
    for pc in listpostcode:
        sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
        traningY = df_dis[pc][i-120:i]

        trainingdf1 = Dict_l0_pca[i]['Tdata'].reset_index(drop = True) 
        l0_true = trainingdf1[trainingdf1.columns[-1]]
        trainingdf1 = trainingdf1.drop(trainingdf1.columns[-1], axis=1)

        trainingdf2 = Dict_l1_pca[10][sa4]['data']# Repeat each row n times
        df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
        df_repeated = df_repeated.reset_index(drop=True)
        trainingdf2_n =  df_repeated.iloc[i-120:i,].reset_index(drop = True) 
        l1_true = trainingdf2_n[ trainingdf2_n.columns[-1]]

        trainingdf2_n = trainingdf2_n.drop(trainingdf2_n.columns[-1], axis=1)
        df_combined = pd.concat([trainingdf1,trainingdf2_n], axis=1)

        Y = traningY.values
        X = df_combined.values
        X_train = sm.add_constant(X)
        model1 = sm.OLS(Y,X_train)
        reg_1 = model1.fit()
            
        # Prediction 
        trainingdf1_pre = Dict_l0_pca[i]['pre_values'].reset_index(drop = True)
        trainingdf1_pre = trainingdf1_pre.iloc[131-i:,].reset_index(drop = True)
        l0_pre = trainingdf1_pre[trainingdf1_pre.columns[-1]]
        trainingdf1_pre = trainingdf1_pre.drop(trainingdf1_pre.columns[-1], axis=1)

        trainingdf2_pre = Dict_l1_pca[10][sa4]['pre_values2']# Repeat each row n times
        
        df_repeated_pre = trainingdf2_pre.loc[trainingdf2_pre.index.repeat(12)]
        df_repeated_pre = df_repeated_pre.reset_index(drop=True)
        trainingdf2_n_pre =  df_repeated_pre.iloc[0:i-12,].reset_index(drop = True) 
        l1_pre = trainingdf2_n_pre[ trainingdf2_n_pre.columns[-1]]

        trainingdf2_n_pre = trainingdf2_n_pre.drop(trainingdf2_n_pre.columns[-1], axis=1)
        df_combined_pre = pd.concat([trainingdf1_pre,trainingdf2_n_pre], axis=1)

        X_test1 = sm.add_constant(df_combined_pre) 
        if df_combined_pre.shape[0]==1:
            X_test = np.insert(df_combined_pre, 0, 1, axis=1)    
        
        predicted_Hier= reg_1.predict(X_test1)    
        errors =  l0_pre + l1_pre + predicted_Hier.values
        error_array.append(errors**2)

    mse_arr_hier.append(np.sqrt(np.mean(error_array)))
        
np.mean(mse_arr_hier)




# %%
#%% Half Hierarchical: 
Dict_l1_pca = {}
for i in range(10,11):
    Dict_l1_pca[i] = {}
    for sa4 in DataFrameDict_Hsa4.keys():
        Dict_l1_pca[i][sa4] = {}
        data  = d2pca_SA4(d2,sa4)
        l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
        l1.index = l1_index
        l1  = l1 .rename_axis('date')
        df_combined = pd.concat([data,l1],axis = 1)
        df = df_combined.dropna()
        df = df.reset_index(drop = True)
        train_data = df.iloc[i-10:i, :]
        model = VAR(train_data)
        results = model.fit(1)

        pre_values = results.fittedvalues
        pre_values_index =  pd.date_range(start='2011', end='2020', freq='Y')
        pre_values_index =  pre_values_index.strftime('%Y')
        pre_values.index =  pre_values_index 

        for_values = results.forecast(train_data.values[-1:], steps=11-i)
        for_values = pd.DataFrame(for_values)
        for_values_index =  pd.date_range(start='2020', end='2021', freq='Y')
        for_values_index =  for_values_index.strftime('%Y')
        for_values.index =  for_values_index
        
        Dict_l1_pca[i][sa4]['Tdata'] = train_data
        Dict_l1_pca[i][sa4]['pre_values'] =  pre_values
        Dict_l1_pca[i][sa4]['for_values']  =  for_values
        Dict_l1_pca[i][sa4]['data'] = df


        l1 = pd.DataFrame(df_l[key].groupby(df_l.index.year).mean())
        l1.index = l1_index
        l1  = l1.rename_axis('date')
        df_combined = pd.concat([data,l1],axis = 1)
        df = df_combined.dropna()
        df = df.reset_index(drop = True)
        train_data = df
        model = VAR(train_data)
        results = model.fit(1)
        pre_values2 = results.fittedvalues
        pre_values_index2 =  pd.date_range(start='2011', end='2021', freq='Y')
        pre_values_index2 =  pre_values_index2.strftime('%Y')
        pre_values2.index =  pre_values_index2
        Dict_l1_pca[i][sa4]['pre_values2'] =  pre_values2

        
postcode_deleted = list(aveHPIdf_lv3.loc[aveHPIdf_lv3['sa4'] == 'Wide Bay']['postcode'].unique())
listpostcode = list(avg_3_pivot.columns)
listpostcode = [x for x in listpostcode if x not in postcode_deleted]
mse_arr_hier = []
for i in range(120,132):
    error_array = []
    for pc in listpostcode:
        sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == pc]['sa4'].iloc[0]
        traningY = df_dis[pc][i-120:i]

        trainingdf1 = Dict_l0_pca[i]['Tdata'].reset_index(drop = True) 
        l0_true = trainingdf1[trainingdf1.columns[-1]]
        trainingdf1 = trainingdf1.drop(trainingdf1.columns[-1], axis=1)

        trainingdf2 = Dict_l1_pca[10][sa4]['data']# Repeat each row n times
        df_repeated = trainingdf2.loc[trainingdf2.index.repeat(12)]
        df_repeated = df_repeated.reset_index(drop=True)
        trainingdf2_n =  df_repeated.iloc[i-120:i,].reset_index(drop = True) 
        l1_true = trainingdf2_n[ trainingdf2_n.columns[-1]]

        trainingdf2_n = trainingdf2_n.drop(trainingdf2_n.columns[-1], axis=1)
        df_combined = pd.concat([trainingdf1,trainingdf2_n], axis=1)

        Y = traningY.values
        X = df_combined.values
        X_train = sm.add_constant(X)
        model1 = sm.OLS(Y,X_train)
        reg_1 = model1.fit()
            
        # Prediction 
        trainingdf1_pre = Dict_l0_pca[i]['pre_values'].reset_index(drop = True)
        trainingdf1_pre = trainingdf1_pre.iloc[131-i:,].reset_index(drop = True)
        l0_pre = trainingdf1_pre[trainingdf1_pre.columns[-1]]
        trainingdf1_pre = trainingdf1_pre.drop(trainingdf1_pre.columns[-1], axis=1)

        trainingdf2_pre = Dict_l1_pca[10][sa4]['pre_values2']# Repeat each row n times
        
        df_repeated_pre = trainingdf2_pre.loc[trainingdf2_pre.index.repeat(12)]
        df_repeated_pre = df_repeated_pre.reset_index(drop=True)
        trainingdf2_n_pre =  df_repeated_pre.iloc[0:i-12,].reset_index(drop = True) 
        l1_pre = trainingdf2_n_pre[ trainingdf2_n_pre.columns[-1]]

        trainingdf2_n_pre = trainingdf2_n_pre.drop(trainingdf2_n_pre.columns[-1], axis=1)
        df_combined_pre = pd.concat([trainingdf1_pre,trainingdf2_n_pre], axis=1)

        X_test1 = sm.add_constant(df_combined_pre) 
        if df_combined_pre.shape[0]==1:
            X_test = np.insert(df_combined_pre, 0, 1, axis=1)    
        
        predicted_Hier= reg_1.predict(X_test1)    
        errors =  l0_pre + l1_pre + predicted_Hier.values
        error_array.append(errors**2)

    mse_arr_hier.append(np.sqrt(np.mean(error_array)))
        
np.mean(mse_arr_hier)









