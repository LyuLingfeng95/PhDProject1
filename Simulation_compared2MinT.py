#%%
import pickle
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import random

#%%
with open("Dict_strt2Arimas.pickle", "rb") as file:
      Dict_strt2Arimas = pickle.load(file)

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
# %%
def transform_df(df_dis):
    # create a copy of df_dis to avoid modifying the original dataframe
    df_transformed = df_dis.copy()
     # subtract the value of each column in group 3 by the value of the corresponding column in group 2
    for col in avg_3_pivot.columns:
        sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == col]['sa4'].iloc[0]
        df_transformed[col] = df_transformed[col].sub(df_transformed[sa4], axis=0)
    # subtract the value under column 'l0' from columns in group 2
    df_transformed[avg_2_pivot.columns] = df_transformed[avg_2_pivot.columns].sub(df_transformed['l0'], axis=0)
    return df_transformed
# %%
def reverse_transform_df(df_transformed):
    
    # create a copy of df_transformed to avoid modifying the original dataframe
    df_dis = df_transformed.copy()
    
    # add the value under column 'l0' back to columns in group 2
    df_dis[avg_2_pivot.columns] = df_dis[avg_2_pivot.columns].add(df_dis['l0'], axis=0)
    
    # add the value of each column in group 3 back to the value of the corresponding column in group 2
    for col in avg_3_pivot.columns:
        sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == col]['sa4'].iloc[0]
        df_dis[col] = df_dis[col].add(df_dis[ sa4], axis=0)
    
    return df_dis


def reverse_transform_df_12(df_transformed):
    
    # create a copy of df_transformed to avoid modifying the original dataframe
    df_dis = df_transformed.copy()
    
    # add the value under column 'l0' back to columns in group 2
    df_dis[avg_2_pivot.columns] = df_dis[avg_2_pivot.columns].add(df_dis['l0'], axis=0)

    
    return df_dis
# %%
df_l = transform_df(df_dis)
df_ll = df_l.copy().reset_index(drop = True)
# Combine the dataframes
combined_df = pd.concat([avg1, avg_2_pivot], axis=1)
hm_names = list(combined_df.columns)

def ARIMA_generator(M,i,colname):
    data = df_l[colname].values
    training_data_ARMA = data[i-M:i]
    arima_model = auto_arima(training_data_ARMA , seasonal=False, suppress_warnings=True)
    return arima_model,training_data_ARMA 
# %%
def strtpoint2ARIMAModel_Dict_init(M,i):
    ARIMAModel_Dict = {}
    for col in hm_names:
        model,training_data = ARIMA_generator(M,i,col)
        ARIMAModel_Dict[col] = {'Model':model, 'Data':  training_data }   
    return ARIMAModel_Dict

#%%
# Dict_strt2Arimas = {}

# for i in range(M0, df_l.shape[0]):
#     Dict_strt2Arimas[i] = strtpoint2ARIMAModel_Dict_init(M0, i)

with open("Dict_strt2Arimas.pickle", "wb") as file:
    pickle.dump(Dict_strt2Arimas, file)


#%%
df_len = 17

# Create an empty dictionary to store the columns
columns = {}

# Loop through the column names and create an empty list for each one
for col_name in hm_names:
    columns[col_name] = []

# Populate the columns with the desired length (17)
for i in range(df_len):
    for col_name in hm_names:
        columns[col_name].append(None)

# Create the dataframe from the dictionary of columns
new_df = pd.DataFrame(columns)


# Loop through each row index in new_df
for R in range(0,17):
    # Loop through each column name in new_df
    for key in new_df.columns:
        # Get the corresponding ARIMA model from Dict_strt2Arimas and predict one period
        arima_model = Dict_strt2Arimas[R + 120][key]['Model']
        prediction = arima_model.predict(n_periods=1, return_conf_int=False)[0]
        # Set the value in new_df to the predicted value
        new_df.loc[R, key] = prediction

df_onestep_p1 = new_df

#%%
df_test= df_dis[hm_names].iloc[-17:,].reset_index(drop= True)
dff1 = reverse_transform_df_12(df_onestep_p1)

((dff1-df_test)**2).mean().mean() - (df_test**2).mean().mean()




#%%
def str2key(colname,ARIMAModel_Dict):
    key_to_find = colname

# Convert the keys of the dictionary to a list
    keys_list = list(ARIMAModel_Dict.keys())

# Use the index() method of the list to get the position of the key
    if key_to_find in keys_list:
        key_position = keys_list.index(key_to_find)
        return key_position    
    else:
        print("Key {} is not present in the dictionary".format(key_to_find))






# %%
df_onestep
# %%
df_onestep_ori = reverse_transform_df(df_onestep)
df_onestep_ori = df_onestep_ori.filter(avg_3_pivot.columns, axis=1)
df_true = df_dis.copy().reset_index(drop = True)
df_true = df_true.tail(df_onestep_ori.shape[0]).reset_index(drop = True)
error_copula = (df_onestep_ori - df_true).dropna(axis = 1) 
# %%
import copulae
from copulae import EmpiricalCopula, pseudo_obs
from copulae.datasets import load_marginal_data
df  = error_copula

df_emp = df
u = pseudo_obs(df)
emp_cop = EmpiricalCopula(u, smoothing="none")
df_vol = EmpiricalCopula.to_marginals(emp_cop.random(2000, seed=10), df)
# %% import copulae

df  = error_copula

df_emp = df
u = pseudo_obs(df)
emp_cop = EmpiricalCopula(u, smoothing="none")
df_vol = EmpiricalCopula.to_marginals(emp_cop.random(2000, seed=10), df)

# %%
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

# %%
sa4_lvls= list(avg_2_pivot.columns)
postcode_lvls = list(avg_3_pivot.columns)
S = generate_matrix_S(sa4_lvls, postcode_lvls, aveHPIdf_lv3)
S_u =S[:S.shape[0]- S.shape[1],:]
S_u = S_u / np.sum(S_u, axis=1, keepdims=True)


# %%
df_ll = df_l.copy().reset_index(drop = True)
# Combine the dataframes
combined_df = pd.concat([avg1, avg_2_pivot], axis=1)
hm_names = list(combined_df.columns)
# %%

        
#%%
def Forecasts_sim(M, i, ARIMAModel_Dict0):
    ARIMAModel_Dict = ARIMAModel_Dict0.copy()
    arr_of_arrs = []
    for steps in range (0, df_l.shape[0] - i):
        l_before_aj = []
        
        for key in ARIMAModel_Dict.keys():
            model = ARIMAModel_Dict[key]['Model']
            l_before_aj.append(model.predict(n_periods=1, return_conf_int=False)[0])
            
        arr = []
        random_int_vol = random.randint(0, 1999)
        for postcode in list(avg_3_pivot.columns):
            sa4 = aveHPIdf_lv3.loc[aveHPIdf_lv3['postcode'] == postcode]['sa4'].iloc[0]
            N = df_ll.shape[0] - M
            train_start = i - M
            train_end = i - 1
            test_start = i
            train_X = df_ll.loc[train_start:train_end+1,['l0', sa4]].values
            train_Y = df_ll.loc[train_start:train_end+1,postcode]
            model = LinearRegression().fit(train_X, train_Y)           
            test_X = np.array([l_before_aj[0],l_before_aj[str2key(sa4, ARIMAModel_Dict)]]).reshape(1,-1)
            test_Y_pred = model.predict(test_X) 
            l012_next = test_Y_pred + np.sum(test_X) + df_vol[postcode].iloc[random_int_vol]
            arr.append(l012_next[0])
        
        arr_newdata = np.dot(S_u, arr)
        first_element = arr_newdata[0]
        for k in range(1, len(arr_newdata)):
            arr_newdata[k] = arr_newdata[k] - first_element  
    
        l_before_aj = []
        for i in range(0,len(arr_newdata)):
            key = list(df_l.columns)[i]
            ARIMAModel_Dict[key]['Model'] = ARIMAModel_Dict[key]['Model'].update(arr_newdata[i])
            l_new =ARIMAModel_Dict[key]['Model'].predict(n_periods=1, return_conf_int=False)[0]
            l_before_aj.append(l_new)
    


        arr_of_arrs.append(arr)  
    return arr_of_arrs



#%%
# for i in range(M0, df_l.shape[0]):
#     ARIMAModel_Dict0 = Dict_strt2Arimas[i]   
#     arr_of_arrs_of_arrs = []
#     for sim in range (0,Sim_Times):
#         arr_of_arrs = Forecasts_sim(M0, i, ARIMAModel_Dict0 )
#         arr_of_arrs_of_arrs.append(arr_of_arrs)
#     d3_arr = np.array(arr_of_arrs_of_arrs)
#     avg_arr = np.mean(d3_arr, axis=0)
#     arrays = [avg_arr[:, j] for j in range(avg_3_pivot.shape[1])]
#     desired_length = df_l.shape[0] - M0
#     k = 0
    
#     for key in Forecasts_dict.keys():
#         # Get the current length of the array (assuming it's a 1D NumPy array)
#         my_array = arrays[k]
#         current_length = len(my_array )
#         # If the current length is smaller than the desired length, pad with zeros
#         if current_length < desired_length:
#             zeros_to_add = desired_length - current_length
#             my_array = np.pad(my_array, (0, zeros_to_add), mode='constant', constant_values=0)
        
#         Forecasts_dict[key]['window'+str(i - M0 + 1)] = my_array
#         k = k +1 
#%%
with open("Dict_strt2Arimas.pickle", "rb") as file:
    Dict_strt2Arimas = pickle.load(file)

# %%
Forecasts_dict = {}
num_cols = df_l.shape[0] - M0
# Iterate over the columns in avg_3_pivot
for colname in avg_3_pivot.columns:
    new_df = pd.DataFrame({}, columns=['window'+str(k) for k in range(1, num_cols+1)])
    Forecasts_dict[colname] = new_df

#%%


def calculate_sim_forecast(i):
    ARIMAModel_Dict0 = Dict_strt2Arimas[i]  
    arr_of_arrs_of_arrs = []
    for sim in range (0,Sim_Times):
        arr_of_arrs = Forecasts_sim(M0, i, ARIMAModel_Dict0 )
        arr_of_arrs_of_arrs.append(arr_of_arrs)
    d3_arr = np.array(arr_of_arrs_of_arrs)
    avg_arr = np.mean(d3_arr, axis=0)
    arrays = [avg_arr[:, j] for j in range(avg_3_pivot.shape[1])]
    desired_length = df_l.shape[0] - M0
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

#%%
random.seed(42)
Sim_Times =10
for i in range(M0, df_l.shape[0]):
    calculate_sim_forecast(i)
    print(i) 

#%%
Forecasts_dict

#%%
# Sim_Times =1
# import ray

# ray.get([ calculate_sim_forecast(inp) for inp in range(M0, df_l.shape[0])])

#%%
# for i in range(M0, df_l.shape[0]):
#     ARIMAModel_Dict0 = Dict_strt2Arimas[i]   
#     arr_of_arrs_of_arrs = []
#     for sim in range (0,Sim_Times):
#         arr_of_arrs = Forecasts_sim(M0, i, ARIMAModel_Dict0 )
#         arr_of_arrs_of_arrs.append(arr_of_arrs)
#     d3_arr = np.array(arr_of_arrs_of_arrs)
#     avg_arr = np.mean(d3_arr, axis=0)
#     arrays = [avg_arr[:, j] for j in range(avg_3_pivot.shape[1])]
#     desired_length = df_l.shape[0] - M0
#     k = 0
#     for key in Forecasts_dict.keys():
#         # Get the current length of the array (assuming it's a 1D NumPy array)
#         my_array = arrays[k]
#         current_length = len(my_array )
#         # If the current length is smaller than the desired length, pad with zeros
#         if current_length < desired_length:
#             zeros_to_add = desired_length - current_length
#             my_array = np.pad(my_array, (0, zeros_to_add), mode='constant', constant_values=0)
        
#         Forecasts_dict[key]['window'+str(i - M0 + 1)] = my_array
#         k = k +1 
        
#%%
Hier_Errors_dict = Forecasts_dict.copy()
for key in Hier_Errors_dict.keys():
    length = np.shape(Hier_Errors_dict[key])[0]
    for t in range(0,length):
        Hier_Errors_dict[key].iloc[:length-t,t]  = - df_dis[key].iloc[-length+t:].reset_index(drop = True)
        Hier_Errors_dict[key].iloc[t]  =  Forecasts_dict[key].iloc[t] +  Hier_Errors_dict[key].iloc[t] 



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

# %%
error_average_hier(Hier_Errors_dict, 12, M0)





# %%
