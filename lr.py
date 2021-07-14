import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
random.seed(123)

# Global variables
# phase can be set to either "train" or "eval" or "test"


""" 
You are allowed to change the names of function "arguments" as per your convenience, 
but it should be meaningful.

E.g. y, y_train, y_test, output_var, target, output_label, ... are acceptable
but abc, a, b, etc. are not.
"""

def one_hot(df1, col_name, vals):
    for val in vals:
        df1[col_name + "_" + val] = df1[col_name].apply(lambda x: 1 if x == val else 0)
    return df1.drop(col_name, axis=1)

def preprocess_train(df1):
    global seats_mode, power_mean, mean_engine, mean_mileage
    global year_min, year_max
    global km_min, km_max
    global mileage_min, mileage_max
    global engine_min, engine_max
    global power_min, power_max
    global seats_min, seats_max

    #dropping few columns
    df1 = df1.drop(["Index","torque"], axis = 1)

    #one-hot encoding some features
    df1["company"] = df1["name"].apply(lambda x: x.split()[0])

    fuel_types = ["Diesel", "Petrol", "CNG", "LPG"]
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner', 'Second Owner', 'Third Owner',
                   'Fourth & Above Owner', 'Test Drive Car']
    company_types = ['Maruti', 'Skoda', 'Honda', 'Hyundai',
                     'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
                     'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz',
                     'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
                     'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
                     'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

    col_dict = {
        "fuel": fuel_types,
        "seller_type": seller_types,
        "transmission": transmission_types,
        "owner": owner_types,
        "company": company_types
    }

    for col_name in col_dict.keys():
        df1 = one_hot(df1, col_name, col_dict[col_name])

    #dropping name
    df1 = df1.drop(["name"], axis = 1)

    #filling empty values of seats with its mode in the train dataset
    seats_mode = df1["seats"].mode()
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["seats"]):
            df1.at[i,"seats"] = seats_mode
    df1["seats"] = df1["seats"].astype(int)

    #filling empty values of max_power with its mean in the train dataset
    power_mean = df1["max_power"].mean()
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["max_power"]):
            df1.at[i,"max_power"] = power_mean

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["engine"]):
            df1.at[i, "engine"] = 0 
        else:
            df1.at[i,"engine"] = df1.at[i,"engine"][0:-3]

    df1["engine"] = df1["engine"].astype(int)

    no_of_non_zero_engine = 4500 - 136
    mean_engine = df1["engine"].sum()/no_of_non_zero_engine
    for i in range(0,df1.shape[0]):
        if df1.at[i, "engine"] == 0:
            df1.at[i, "engine"] = mean_engine

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["mileage"]):
             df1.at[i, "mileage"] = 0 
        elif "kmpl" in df1.at[i, "mileage"]:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-5]
        else:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-6]

    df1["mileage"] = df1["mileage"].astype(float)

    no_of_non_zero_mileage = 4500 - 136
    mean_mileage = df1["mileage"].sum()/no_of_non_zero_mileage
    for i in range(0,df1.shape[0]):
        if df1.at[i, "mileage"] == 0.0:
            df1.at[i, "mileage"] = mean_mileage

    #calculating min and max of features in the train dataset for normalising 
    year_min = df1["year"].min()
    year_max = df1["year"].max()
    km_min = df1["km_driven"].min()
    km_max = df1["km_driven"].max()
    mileage_min = df1["mileage"].min()
    mileage_max = df1["mileage"].max()
    engine_min = df1["engine"].min()
    engine_max = df1["engine"].max()
    power_min = df1["max_power"].min()
    power_max = df1["max_power"].max()
    seats_min = df1["seats"].min()
    seats_max = df1["seats"].max()

    #normalising few columns
    df1["year"] = (df1["year"] - year_min) / (year_max - year_min) 
    df1["km_driven"] = (df1["km_driven"] - km_min) / (km_max - km_min)
    df1["mileage"] = (df1["mileage"] - mileage_min) / (mileage_max - mileage_min)
    df1["engine"] = (df1["engine"] - engine_min) / (engine_max - engine_min)
    df1["max_power"] = (df1["max_power"] - power_min) / (power_max - power_min)
    df1["seats"] = (df1["seats"] - seats_min) / (seats_max - seats_min) 

    return df1

def preprocess_val(df1):

    #dropping few columns
    df1 = df1.drop(["Index","torque"], axis = 1)

    #one-hot encoding some features
    df1["company"] = df1["name"].apply(lambda x: x.split()[0])

    fuel_types = ["Diesel", "Petrol", "CNG", "LPG"]
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner', 'Second Owner', 'Third Owner',
                   'Fourth & Above Owner', 'Test Drive Car']
    company_types = ['Maruti', 'Skoda', 'Honda', 'Hyundai',
                     'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
                     'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz',
                     'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
                     'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
                     'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

    col_and_type_dict = {
        "fuel": fuel_types,
        "seller_type": seller_types,
        "transmission": transmission_types,
        "owner": owner_types,
        "company": company_types
    }

    for col_name in col_and_type_dict.keys():
        df1 = one_hot(df1, col_name, col_and_type_dict[col_name])

    #dropping name
    df1 = df1.drop(["name"], axis = 1)

    #filling empty values of seats with its mode in the train dataset
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["seats"]):
            df1.at[i,"seats"] = seats_mode
    df1["seats"] = df1["seats"].astype(int)

    #filling empty values of max_power with its mean in the train dataset
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["max_power"]):
            df1.at[i,"max_power"] = power_mean

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["engine"]):
            df1.at[i, "engine"] = 0 
        else:
            df1.at[i,"engine"] = df1.at[i,"engine"][0:-3]

    df1["engine"] = df1["engine"].astype(int)

    for i in range(0,df1.shape[0]):
        if df1.at[i, "engine"] == 0:
            df1.at[i, "engine"] = mean_engine

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["mileage"]):
             df1.at[i, "mileage"] = 0 
        elif "kmpl" in df1.at[i, "mileage"]:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-5]
        else:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-6]

    df1["mileage"] = df1["mileage"].astype(float)

    for i in range(0,df1.shape[0]):
        if df1.at[i, "mileage"] == 0.0:
            df1.at[i, "mileage"] = mean_mileage

    #normalising few columns
    df1["year"] = (df1["year"] - year_min) / (year_max - year_min) 
    df1["km_driven"] = (df1["km_driven"] - km_min) / (km_max - km_min)
    df1["mileage"] = (df1["mileage"] - mileage_min) / (mileage_max - mileage_min)
    df1["engine"] = (df1["engine"] - engine_min) / (engine_max - engine_min)
    df1["max_power"] = (df1["max_power"] - power_min) / (power_max - power_min)
    df1["seats"] = (df1["seats"] - seats_min) / (seats_max - seats_min)

    return df1

def preprocess_test(df1):

    #dropping few columns
    df1 = df1.drop(["Index","torque"], axis = 1)

    #one-hot encoding some features
    df1["company"] = df1["name"].apply(lambda x: x.split()[0])

    fuel_types = ["Diesel", "Petrol", "CNG", "LPG"]
    seller_types = ['Individual', 'Dealer', 'Trustmark Dealer']
    transmission_types = ['Manual', 'Automatic']
    owner_types = ['First Owner', 'Second Owner', 'Third Owner',
                   'Fourth & Above Owner', 'Test Drive Car']
    company_types = ['Maruti', 'Skoda', 'Honda', 'Hyundai',
                     'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata',
                     'Chevrolet', 'Fiat', 'Datsun', 'Jeep', 'Mercedes-Benz',
                     'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan',
                     'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
                     'Kia', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel']

    col_and_type_dict = {
        "fuel": fuel_types,
        "seller_type": seller_types,
        "transmission": transmission_types,
        "owner": owner_types,
        "company": company_types
    }

    for col_name in col_and_type_dict.keys():
        df1 = one_hot(df1, col_name, col_and_type_dict[col_name])

    #dropping name
    df1 = df1.drop(["name"], axis = 1)

    #filling empty values of seats with its mode in the train dataset
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["seats"]):
            df1.at[i,"seats"] = seats_mode
    df1["seats"] = df1["seats"].astype(int)

    #filling empty values of max_power with its mean in the train dataset
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["max_power"]):
            df1.at[i,"max_power"] = power_mean

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["engine"]):
            df1.at[i, "engine"] = 0 
        else:
            df1.at[i,"engine"] = df1.at[i,"engine"][0:-3]

    df1["engine"] = df1["engine"].astype(int)

    for i in range(0,df1.shape[0]):
        if df1.at[i, "engine"] == 0:
            df1.at[i, "engine"] = mean_engine

    #processing the engine column
    for i in range(0,df1.shape[0]):
        if pd.isna(df1.iloc[i]["mileage"]):
             df1.at[i, "mileage"] = 0 
        elif "kmpl" in df1.at[i, "mileage"]:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-5]
        else:
            df1.at[i,"mileage"] = df1.at[i,"mileage"][0:-6]

    df1["mileage"] = df1["mileage"].astype(float)

    for i in range(0,df1.shape[0]):
        if df1.at[i, "mileage"] == 0.0:
            df1.at[i, "mileage"] = mean_mileage

    #normalising few columns
    df1["year"] = (df1["year"] - year_min) / (year_max - year_min) 
    df1["km_driven"] = (df1["km_driven"] - km_min) / (km_max - km_min)
    df1["mileage"] = (df1["mileage"] - mileage_min) / (mileage_max - mileage_min)
    df1["engine"] = (df1["engine"] - engine_min) / (engine_max - engine_min)
    df1["max_power"] = (df1["max_power"] - power_min) / (power_max - power_min)
    df1["seats"] = (df1["seats"] - seats_min) / (seats_max - seats_min) 

    return df1

def get_features(file_path):
	# Given a file path , return feature matrix and target labels 
    global phase

    df = pd.read_csv(file_path)

    if phase == "train":
        phi = preprocess_train(df.drop("selling_price", axis=1))
        y = df["selling_price"].to_numpy()
        return phi, y
    if phase == "eval":
        phi = preprocess_val(df.drop("selling_price", axis=1))
        y = df["selling_price"].to_numpy()
        return phi, y
    if phase == "test":
        phi = preprocess_test(df)
        return phi, None

def get_features_basis(file_path):
	# Given a file path , return feature matrix and target labels 
    global phase
	
    df = pd.read_csv(file_path)

    if phase == "train":
        phi = preprocess_train(df.drop("selling_price", axis=1))
        phi["year"] = phi["year"].apply(lambda x: x**(5))
        phi["km_driven"] = phi["km_driven"].apply(lambda x: x**4)
        y = df["selling_price"].to_numpy()
        return phi, y
    if phase == "eval":
        phi = preprocess_val(df.drop("selling_price", axis=1))
        phi["year"] = phi["year"].apply(lambda x: x**(5))
        phi["km_driven"] = phi["km_driven"].apply(lambda x: x**4)
        y = df["selling_price"].to_numpy()
        return phi, y
    if phase == "test":
        phi = preprocess_test(df)
        phi["year"] = phi["year"].apply(lambda x: x**(4))
        phi["km_driven"] = phi["km_driven"].apply(lambda x: x**(4))
        return phi, None

def compute_RMSE(phi, w , y) :
    # Root Mean Squared Error
    diff = phi@w - y
    se = (diff**2).sum()
    mse = se/len(diff)
    error = np.sqrt(mse)
    return error

def generate_output(phi_test, w):
	# writes a file (output.csv) containing target variables in required format for Submission.
    df = pd.DataFrame(phi_test@w)
    df[0] = df[0].apply(lambda x: max(-x, x))
    df.to_csv("output.csv")

def closed_soln(phi, y):
    # Function returns the solution w for Xw=y.
    return np.linalg.pinv(phi).dot(y)
	
def gradient_descent(phi, y, phi_dev, y_dev) :
	# Implement gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    w = np.random.normal(0, 1, phi.shape[1])
    max_epochs = 100000  #maximum epochs
    lr = 0.0001          #learning rate
    prev_rmse = compute_RMSE(phi_dev, w, y_dev)
    for epoch in range(0, max_epochs):
        w = w - lr * (phi.T@(phi@w - y))
        current_rmse = compute_RMSE(phi_dev, w, y_dev)
        if prev_rmse < current_rmse:
            break
        prev_rmse = current_rmse
    return w

def sgd(phi, y, phi_dev, y_dev) :
	# Implement stochastic gradient_descent using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence

    w = np.random.normal(0, 1, phi.shape[1])
    max_epochs = 100000        #maximum epochs
    lr = 0.001                 #learning rate
    for epoch in range(0, max_epochs):
        data_index = random.randint(0,phi.shape[0]-1)    #select random data entry
        w = w - lr * phi.iloc[data_index, :]*(phi@w - y)[data_index]
    return w

def pnorm(phi, y, phi_dev, y_dev, p) :
	# Implement gradient_descent with p-norm regularisation using Mean Squared Error Loss
    # You may choose to use the dev set to determine point of convergence
    w = np.random.normal(0, 1, phi.shape[1])
    max_epochs = 100000        #maximum epochs
    lr = 0.0001                #learning rate
    lambda2 = 0.01
    lambda4 = 1e-15
    prev_rmse = compute_RMSE(phi_dev, w, y_dev)
    for ep in range(0, max_epochs):
        if p == 2:
            w = w - lr*((phi.T@(phi@w - y)) + p*w**(p - 2)*lambda2*w)
        else:
            w = w - lr*((phi.T@(phi@w - y)) + p*w**(p - 2)*lambda4*w)
        current_rmse = compute_RMSE(phi_dev, w, y_dev)
        if prev_rmse < current_rmse:
            break
        prev_rmse = current_rmse
    return w	

def graph(phi, y, phi_dev, y_dev):
    errors = []
    lengths = [2000, 2500, 3000, 4500]
    for length in lengths:
        phi_new = phi[:length]
        y_new = y[:length]
        w = gradient_descent(phi_new, y_new, phi_dev, y_dev)
        errors.append(compute_RMSE(phi_dev, w, y_dev))
    plt.figure(figsize=(9,6))
    plt.title("Validation Set RMSE vs Training Size")
    plt.plot(lengths, errors)
    plt.xlabel("Training Size")
    plt.ylabel("Validation Set RMSE")
    plt.savefig("img.png")
    plt.show()

def main():
    #The following steps will be run in sequence by the autograder.
    global phase
    ######## Task 1 #########

    phase = "train"
    phi, y = get_features('df_train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features('df_val.csv')
    w1 = closed_soln(phi, y)
    w2 = gradient_descent(phi, y, phi_dev, y_dev)
    r1 = compute_RMSE(phi_dev, w1, y_dev)
    r2 = compute_RMSE(phi_dev, w2, y_dev)
    #print(r1)
    #print(r2)
    print('1a: ')
    print(abs(r1-r2))
    w3 = sgd(phi, y, phi_dev, y_dev)
    r3 = compute_RMSE(phi_dev, w3, y_dev)
    #print(r3)
    print('1c: ')
    print(abs(r2-r3))

    ######## Task 2 #########
    w_p2 = pnorm(phi, y, phi_dev, y_dev, 2)  
    w_p4 = pnorm(phi, y, phi_dev, y_dev, 4)  
    r_p2 = compute_RMSE(phi_dev, w_p2, y_dev)
    r_p4 = compute_RMSE(phi_dev, w_p4, y_dev)
    print('2: pnorm2')
    print(r_p2)
    print('2: pnorm4')
    print(r_p4)
    
    ######## Task 3 #########
    phase = "train"
    phi_basis, y = get_features_basis('df_train.csv')
    phase = "eval"
    phi_dev, y_dev = get_features_basis('df_val.csv')
    w_basis = pnorm(phi_basis, y, phi_dev, y_dev, 2)
    rmse_basis = compute_RMSE(phi_dev, w_basis, y_dev)
    print('Task 3: basis')
    print(rmse_basis)

    ########## Task 4 #########
    #for generating the graph

    #graph(phi, y, phi_dev, y_dev)

    ########## Task 5 #########
    #check the column with less absolute value of the weights

    #print(abs(w1))

    ########## Task 6 #########
    #used P2 norm with basis features
    #phase = "test"
    #phi_test, _ = get_features_basis('df_test.csv')
    #generate_output(phi_test, w_basis)

main()
