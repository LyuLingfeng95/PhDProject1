#%%
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


# %%
# The name of the original Data is changed by me. 
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


# %%
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

# %%
aveHPIdf_lv3 = aveHPIdf_lv3[aveHPIdf_lv3['state'] == 'NSW']
aveHPIdf_lv3 = aveHPIdf_lv3[aveHPIdf_lv3['date'] > '2009-12-31']

#NSW(The scope of data is narrowed down to the NSW from the national level)
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

#%%
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

# %%
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

#%% 
avg1 = avg_1.set_index("date")
avg2 = avg_2.set_index("date")
avg3 = avg_3.set_index("date")
df_dis =  pd.concat([avg1 , avg2.pivot(columns='sa4', values='l01'), avg3.pivot(columns='postcode', values='l012')],axis = 1)

# #%%
# df_l01 = df_dis.copy()

# def transform_predicted(df):
#     df = df.reset_index(drop=True)
#     results = pd.DataFrame({})
#     for col in avg_2_pivot.columns:
#             training_data = df[col].values 
#             model = auto_arima(training_data, d = None, seasonal=False, suppress_warnings=True, error_action="ignore", stepwise=True, trace=False)
#             arima = ARIMA(training_data, order=model.order)
#             arima_fit = arima.fit()
#             prediction = arima_fit.predict()  
#             results[col] = prediction


#     for col in avg_3_pivot.columns:
#         sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == col]['sa4'].iloc[0]
#         results[col] = df[col].sub(results[sa4], axis=0).values

#     return results   

#%%
# df_pre_dif  = transform_predicted(df_l01)

               
# %%
sa4_names = avg_2_pivot.columns

def ARIMA_generator(M,i,colname):
    data = df_dis[colname].values
    training_data_ARMA = data[i-M:i]
    arima_model = auto_arima(training_data_ARMA , seasonal=False, suppress_warnings=True)
    return arima_model,training_data_ARMA 

def strtpoint2ARIMAModel_Dict_init(M,i):
    ARIMAModel_Dict = {}
    for col in sa4_names:
        model,training_data = ARIMA_generator(M,i,col)
        ARIMAModel_Dict[col] = {'Model':model, 'Data':  training_data }   
    return ARIMAModel_Dict
# %%
# Dict_strt2Arimas_sa4 = {}
# for i in range(120, df_dis.shape[0]):
#      Dict_strt2Arimas_sa4[i] = strtpoint2ARIMAModel_Dict_init(120, i)

#%%
# Define the file name for saving the dictionary
# filename = 'Dict_strt2Arimas_sa4.pkl'

# # Open a file in binary mode for writing
# with open(filename, 'wb') as file:
#     # Use pickle.dump() to write the dictionary to the file
#     pickle.dump(Dict_strt2Arimas_sa4, file)

with open(filename, 'rb') as file:
     # Use pickle.load() to read the dictionary from the file
     Dict_strt2Arimas_sa4 = pickle.load(file)


# %%
# Generate the one-step forecasts 
df_onestep = pd.DataFrame({})
for col in avg_3_pivot.columns:
    arr_1stepf = []
    for i in  range(120, df_dis.shape[0]):
         sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == col]['sa4'].iloc[0]
         model = Dict_strt2Arimas_sa4[i][sa4]['Model']
         onestepf = model.predict(n_periods=1, return_conf_int=False)[0]
         arr_1stepf.append(onestepf)
    df_onestep[col] = arr_1stepf
# %%

df_dis_se = df_dis.reset_index(drop = True)

df_dis_test = df_dis_se[avg_3_pivot.columns].iloc[-17:].reset_index(drop =True)

error_copula = df_onestep - df_dis_test

(np.sqrt((error_copula **2).mean())).mean()
# %%
df_emp =  error_copula
error_mean =  error_copula.mean()
u = pseudo_obs(df_emp)
emp_cop = EmpiricalCopula(u, smoothing="beta")
df_vol = EmpiricalCopula.to_marginals(emp_cop.random(10000, seed=10), df_emp)

#%%
vol_shift = df_vol.mean()['2010']
adjusted_f_s= df_onestep['2010'].values - vol_shift
error_vol_s = np.sqrt(np.mean((adjusted_f_s - df_dis_test['2010'].values)**2))
error_wovol_s = np.sqrt(np.mean((df_onestep['2010'] - df_dis_test['2010'].values)**2))

error_vol_s - error_wovol_s
#%%
all1stepRMSE = []
for pc in avg_3_pivot.columns:
    vol_shift = df_vol.mean()[pc]
    adjusted_f_s= df_onestep[pc].values - vol_shift
    error_vol_s = np.sqrt(np.mean((adjusted_f_s - df_dis_test[pc].values)**2))
    error_wovol_s = np.sqrt(np.mean((df_onestep[pc] - df_dis_test[pc].values)**2))
    all1stepRMSE.append(error_vol_s - error_wovol_s)

np.mean(all1stepRMSE)





#%%
def generate_matrix_S(sa4_lvls, postcode_lvls, aveHPIdf_lv3):
    # Initialize a matrix of zeros with the same shape as the desired matrix S
    S = np.zeros((len(sa4_lvls), len(postcode_lvls)))

    # Loop over the rows of S
    for i in range(len(sa4_lvls)):
        # Get a boolean array indicating which rows of aveHPIdf_lv3 have sa4 equal to sa4_lvls[i]
        sa4_match = aveHPIdf_lv3['sa4'] == sa4_lvls[i]

        # Loop over the columns of S
        for j in range(len(postcode_lvls)):
            # Set S[i,j] to 1 if aveHPIdf_lv3['postcode'] is equal to postcode_lvls[j] and sa4_match is True
            if sa4_match.any() and (aveHPIdf_lv3['postcode'][sa4_match] == postcode_lvls[j]).any():
                S[i,j] = 1
    top_row = np.ones((1, len(postcode_lvls)))
    bottom_row = np.eye(len(postcode_lvls))
    Sfinal = np.vstack([top_row, S, bottom_row])

            

    # Convert the matrix to a dataframe with the appropriate row and column names
#     S = pd.DataFrame(S, index=sa4_lvls, columns=postcode_lvls)

    return Sfinal 

sa4_lvls= list(avg_2_pivot.columns)
postcode_lvls = list(avg_3_pivot.columns)
S = generate_matrix_S(sa4_lvls, postcode_lvls, aveHPIdf_lv3)
S_u =S[:S.shape[0]- S.shape[1],:]
S_u = S_u / np.sum(S_u, axis=1, keepdims=True)


# %%
def Forecasts_sim(M, i, ARIMAModel_Dict0):
    ARIMAModel_Dict = ARIMAModel_Dict0.copy()
    arr_of_arrs = []
    for steps in range (0, df_dis.shape[0] - i):
        arr = []
        random_int_vol = random.randint(0, 9999)
        for postcode in list(avg_3_pivot.columns):
            sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == postcode]['sa4'].iloc[0]
            model = ARIMAModel_Dict[sa4]['Model']
            test_Y_pred = model.predict(n_periods=1, return_conf_int=False)[0]
            l012_next =  test_Y_pred - 1 * df_vol[postcode].iloc[random_int_vol]
            arr.append(l012_next)
        arr_newdata = np.dot(S_u, arr)[1:]
        for i in range(0,len(arr_newdata)):
            key = list(avg_2_pivot.columns)[i]
            ARIMAModel_Dict[key]['Model'] = ARIMAModel_Dict[key]['Model'].update(arr_newdata[i])
        arr_of_arrs.append(arr)  
    return arr_of_arrs



# %%
M0 =120
Forecasts_dict = {}
num_cols = df_dis.shape[0] - M0
# Iterate over the columns in avg_3_pivot
for colname in avg_3_pivot.columns:
    new_df = pd.DataFrame({}, columns=['window'+str(k) for k in range(1, num_cols+1)])
    Forecasts_dict[colname] = new_df

#%%
def calculate_sim_forecast(i):
    arr_of_arrs_of_arrs = []
    fname = 'Dict_strt2Arimas_sa4.pkl'
    for sim in range (0,Sim_Times):
        with open(fname, 'rb') as file:
            Dict_strt2Arimas_sa4 = pickle.load(file)
        ARIMAModel_Dict0 = Dict_strt2Arimas_sa4[i]  
        arr_of_arrs = Forecasts_sim(M0, i, ARIMAModel_Dict0)
        arr_of_arrs_of_arrs.append(arr_of_arrs)
    d3_arr = np.array(arr_of_arrs_of_arrs)
    avg_arr = np.mean(d3_arr, axis=0)
    arrays = [avg_arr[:, j] for j in range(avg_3_pivot.shape[1])]
    desired_length = df_dis.shape[0] - M0
    k = 0

    for key in Forecasts_dict.keys():
        # Get the current length of the array (assuming it's a 1D NumPy array)
        my_array = arrays[k]
        current_length = len(my_array )
        # If the current length is smaller than the desired length, pad with zeros
        if current_length < desired_length:
            zeros_to_add = desired_length - current_length
            my_array = np.pad(my_array, (0, zeros_to_add), mode='constant', constant_values=0)
        
        Forecasts_dict[key]['window'+str(i - M0 + 1)] = my_array
        k = k +1 

# %%
Sim_Times = 50
M0 =120
random.seed(42)
for i in range(M0, df_dis.shape[0]):
    calculate_sim_forecast(i)
    print(i) 


# %%
def error_average_hier(Dict_Name, h, M):
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
        df_error = df.iloc[0: h+1, 0:len(df_l)-M-h].copy()
        # calculate RMSE by row
        rmse_by_row.append(np.sqrt(np.mean((df_error.values)**2)))

    # calculate average of average RMSE by row
    if rmse_by_row:
        average_rmse = np.mean([np.mean(rmse) for rmse in rmse_by_row])
    else:
        average_rmse = np.nan

    return average_rmse


#%%
# Define the file name for saving the dictionary
Hier_Errors_dict = {}
for key in Forecasts_dict.keys():
    test_df = pd.DataFrame({})
    data = df_dis[key].values
    # Add a column for each window
    for w in range(1, 18):
        p = 18 - w
        last_p = data[-p:]
        new_array = np.zeros(17 - p)
        new_array = np.concatenate((last_p, new_array))
        test_df[f"window{w}"] = new_array
    # Add the new dataframe to Hier_Errors_dict with the same key as in Forecasts_dict
#     errors_df = Forecasts_dict[key].iloc[t]  -  test_df
    errors_df =  Forecasts_dict[key] -  test_df
    Hier_Errors_dict[key] = errors_df

Hier_Errors_dict_vol = Hier_Errors_dict
# Hier_Errors_dict_wo_vol = Hier_Errors_dict

#%%
filename = 'Hier_Errors_dict_vol.pickle'

# Open a file in binary mode and write the dictionary using the pickle.dump() method
with open(filename, 'wb') as f:
    pickle.dump(Hier_Errors_dict_vol, f)

# filename = 'Hier_Errors_dict_wo_vol.pickle'

# # Open a file in binary mode and write the dictionary using the pickle.dump() method
# with open(filename, 'wb') as f:
#     pickle.dump(Hier_Errors_dict_wo_vol, f)
# %%
error_average_hier(Hier_Errors_dict_vol, 12, M0)
# %%
error_average_hier(Hier_Errors_dict_wo_vol, 12, M0)

#%%
# initialize an empty array to store the results
array_h_v = []

# loop through values of i from 1 to 12
for i in range(1, 13):
    # call the error_average_hier function with arguments Hier_Errors_dict_vol and M0
    result = error_average_hier(Hier_Errors_dict_vol, i, M0)
    # append the result to the array
    array_h_v.append(result)

# %%

array_h_wv = []

for i in range(1, 13):
    # call the error_average_hier function with arguments Hier_Errors_dict_vol and M0
    result = error_average_hier(Hier_Errors_dict_wo_vol, i, M0)
    # append the result to the array
    array_h_wv.append(result)


# %%

df_yt = pd.concat([sum_1, sum_2_pivot, avg_3_pivot], axis=1)

# Convert the 'date' column to a datetime type
df_yt['date'] = pd.to_datetime(df_yt['date'])

# Set the 'date' column as the index
df_yt = df_yt.set_index('date')

# %%

def base_errors(colname, M):
    # Get the data for the specified column
    data = df_yt[colname].values
    error_df = pd.DataFrame()

    # Initialize a matrix to store the forecasts
#     errors = np.zeros((N, data.shape[0] - M))

    # Loop over the rolling window
    for i in range(M, data.shape[0] ):
        N = data.shape[0] - M
        training_data = data[i-M:i]
        test_data = data[i:i+N]

        # Fit ARIMA model to training data
        model = auto_arima(training_data, d = None, seasonal=False, suppress_warnings=True, error_action="ignore", stepwise=True, trace=False)
        arima = ARIMA(training_data, order=model.order)
        arima_fit = arima.fit()
    
        # Make N-step-ahead forecast 
        forecast = arima_fit.forecast(steps=N)
        # Check if the forecast is longer than the test_data
        if len(forecast) > len(test_data):
        # Calculate the difference between the lengths of the two arrays
            difference = len(forecast) - len(test_data)
    
        # Add the extra values from the forecast array to the end of the test_data array
            test_data = np.append(test_data, forecast[-difference:])
        error_df['sim'+str(i)] = forecast - test_data
        # Calculate average MSE for each series
    return  error_df
# %%
colnames = df_yt.columns
M0  = 120
Dict_Errors = {}

for i, colname in enumerate(colnames):
    error_results = base_errors(colname,M0)
    Dict_Errors[colname] = error_results
    print(f'Progress: {i+1}/{len(colnames)}')

#%%
# with open('Dict_Errors.pkl', 'wb') as f:
#     pickle.dump(Dict_Errors, f)
# %%
def forecast_base(M, h, Dict_Errors, df_yt):
    forecasts = pd.DataFrame({}, columns=df_yt.columns, index=df_yt.index)

    # Loop over all dataframes in Dict_Errors
    for key in Dict_Errors:
        # Find the row in Dict_Errors with index h-1 and extract all values before the last h column
        errors = Dict_Errors[key].iloc[h-1, :Dict_Errors[key].shape[1]-h+1]

        # Calculate the length of the extracted values from Dict_Errors
        length = len(errors)

        # Cut df_yt[key] into two parts
        first_part = df_yt[key].iloc[:-length]
        second_part = df_yt[key].iloc[-length:]

        # Add the second part of df_yt[key] and the extracted values from Dict_Errors to the forecast dataframe
        forecasts.loc[second_part.index, key] = second_part.values + errors.values


    return forecasts

#%%
# Extract the first row of each dataframe in Dict_Errors and combine them as new columns in a dataframe called error_1step
error_1step = pd.concat([Dict_Errors[key].iloc[0, :] for key in Dict_Errors], axis=1)

# Rename the columns of error_1step to match the keys in Dict_Errors
error_1step.columns = Dict_Errors.keys()
# %%
sa4_lvls= list(avg_2_pivot.columns)
postcode_lvls = list(avg_3_pivot.columns)
# %%

def mint_shrink(base_forecasts, base_error_1step):
    """
    Implements the MinT (Shrink) algorithm for forecast reconciliation.
    Given a set of N-step-ahead base forecasts and the in-sample one-step-ahead
    base forecast errors, returns the reconciled forecasts using the MinT (Shrink) method.
    """
    # Get the number of time series and the forecast horizon


    # Compute the empirical covariance matrix of the base forecast errors
    Sigma = np.cov(base_error_1step.T)
    diagonal_entries = np.diag(Sigma)
    D = np.diag(diagonal_entries)
    correlation_matrix = base_error_1step.corr()

    # Calculate the variance of the correlation matrix
    correlation_matrix_var = correlation_matrix.apply(lambda x: (1 - x**2) / (len(base_error_1step) - 3)).values

    # Calculate the sum of the variances of the correlation estimates
    variance_sum = np.sum(correlation_matrix_var[np.triu_indices(len(base_error_1step), k=1)])

    # Calculate the sum of the squares of the correlation estimates
    correlation_sum = np.sum(correlation_matrix.values[np.triu_indices(len(base_error_1step), k=1)]**2)

    # Calculate the estimator for lambda_D
    lambda_D_hat = variance_sum / correlation_sum

    W_h = lambda_D_hat* D + (1 - lambda_D_hat) * Sigma

    m_star = S.shape[0] - S.shape[1]

    C = S[:len(sa4_lvls)+1,: ]

    U = np.hstack([np.identity(m_star), -C]).transpose()

    J = np.concatenate((np.zeros(( S.shape[1], m_star)), np.eye(S.shape[1])), axis=1)

    J_W_U = np.dot(np.dot(J, W_h), U)

    UWUinv = np.linalg.inv(np.dot(np.dot(U.transpose(), W_h),U) )

    J_W_UUWUinv =  np.dot(J_W_U,UWUinv )

    coefficient = np.dot(S, J - np.dot(J_W_UUWUinv ,U.transpose()))
    
    reconciled_forecasts =np.dot(coefficient,base_forecasts.values.transpose())
    return reconciled_forecasts

# %%
Dict_recon_Errors = {key: pd.DataFrame() for key in Dict_Errors}

for h in range(1, len(df_yt) - M0 + 1):
    mat = mint_shrink(forecast_base(M0, h, Dict_Errors, df_yt).dropna(), error_1step)
    i = 0
    for key in  Dict_recon_Errors.keys():
        forecasts = mat[i,:] 
        length = len(forecasts)
        testset = df_yt[key].iloc[-length:]
        new_row = (forecasts  -   testset).values
        new_row_length = len(new_row)
        if new_row_length < len(df_yt) - M0 :
            missing_cols = len(df_yt) - M0  - new_row_length
            new_row = np.append(new_row, np.full(missing_cols, np.nan))
        new_df = pd.DataFrame([new_row], columns=df_yt.index[-len(df_yt) + M0 :])

# concatenate the new dataframe to the original dataframe
        Dict_recon_Errors[key] = Dict_recon_Errors[key].append(new_df, ignore_index=True)
        i = i + 1
        
       
# %%
from sklearn.metrics import mean_squared_error

def error_average(Dict_Name, h, Level):
    # check if level is valid
    if Level not in ['High', 'Mid', 'Low']:
        raise ValueError("Level must be 'High', 'Mid', or 'Low'.")

    # extract column names based on level
    if Level == 'Low':
        col_names = avg_3_pivot.columns
    elif Level == 'Mid':
        col_names = avg_2_pivot.columns
    else:
        col_names = ['l0_s']

    # extract dataframes from Dict_Name using column names as keys
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
        df_error = df.iloc[0: h+1, 0:len(df_yt)-M0-h].copy()
        # calculate RMSE by row
        rmse_by_row.append(np.sqrt(np.mean((df_error.values)**2)))

    # calculate average of average RMSE by row
    if rmse_by_row:
        average_rmse = np.mean([np.mean(rmse) for rmse in rmse_by_row])
    else:
        average_rmse = np.nan

    return average_rmse


# %%
array_b = []
for i in range(1, 13):
    # call the error_average_hier function with arguments Hier_Errors_dict_vol and M0
    result = error_average(Dict_Errors, i, 'Low')
    # append the result to the array
    array_b.append(result)

# %%
array_mint = []
for i in range(1, 13):
    # call the error_average_hier function with arguments Hier_Errors_dict_vol and M0
    result = error_average(Dict_recon_Errors, i, 'Low')
    # append the result to the array
    array_mint.append(result)
# %%
table_data = np.vstack((array_b, array_mint, array_h_wv, array_h_v))
new_table_data = np.vstack((table_data[0], table_data[1:]/table_data[0]-1))
# Print the table as LaTeX code
print('\\begin{table}[h]')
print('\\centering')
print('\\begin{tabular}{|c|c|c|c|}')
print('\\hline')
print(' & Model B & Model MINT & Hierarchical Model (with volatility) & Hierarchical Model (without volatility) \\\\')
print('\\hline')
for i in range(new_table_data.shape[0]):
    row = ' & '.join([f'{100 * val :.2f}' for val in new_table_data[i]])
    row = f'{i+1} & {row} \\\\'
    print(row)
    print('\\hline')
print('\\end{tabular}')
print('\\caption{Table caption goes here.}')
print('\\label{table:my_table}')
print('\\end{table}')
# %%