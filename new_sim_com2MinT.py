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

# with open(filename, 'rb') as file:
#     # Use pickle.load() to read the dictionary from the file
#     Dict_strt2Arimas_sa4 = pickle.load(file)


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
            l012_next =  test_Y_pred +0 * df_vol[postcode].iloc[random_int_vol]
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
Sim_Times = 1
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

#%%
filename = 'Hier_Errors_dict_vol.pickle'

# Open a file in binary mode and write the dictionary using the pickle.dump() method
with open(filename, 'wb') as f:
    pickle.dump(Hier_Errors_dict_vol, f)
# %%
error_average_hier(Hier_Errors_dict_vol, 12, M0)
# %%


# %%
