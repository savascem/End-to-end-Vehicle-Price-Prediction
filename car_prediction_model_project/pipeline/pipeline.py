import re
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from io import StringIO
import warnings
import pandas as pd
import time
import requests
from bs4 import BeautifulSoup
from lxml import etree
import pickle
from lists import df_cols, features, paint_list, transmission_list, engine_list, one_hot_encoding_list, scaler_list, drive_train_, brand_, model_1_, fuel_type_, model_2_



def pipeline(url):

    prediction_df  = pd.DataFrame(columns=df_cols, index=[0], data=0)
    prediction_df.shape

    req = requests.get(url)
    soup = BeautifulSoup(req.text, "html.parser")
    tree = etree.HTML(req.text)

    # Primary Price
    price = soup.select_one("span.primary-price").text
    prediction_df.at[0, "Primary Price"] = float(price[1:].replace(",",""))


    #year, brand, model1

    title = soup.select_one("h1.listing-title").text

    value = title.split(" ")
    brand = value[1]
    prediction_df.at[0, "Year"] = int(value[0])
    val_list = value[2:]
    model_ = val_list[0]

    for col in brand_:
        col_ = col[6:]
        if col_ == brand:
            prediction_df.at[0, col] = 1

    for col in model_1_:
        col_ = col[8:]
        if col_ == model_:
            prediction_df.at[0, col] = 1

    if len(val_list) > 1:
        model__ = val_list[1]
        for col in model_2_:
            col_ = col[8:]
            if col_ == model__:
                prediction_df.at[0, col] = 1

    #mileage:
    prediction_df.at[0, "Mileage"] = int(soup.select_one("div.listing-mileage").text[:-4].replace(",",""))


    #Accidents or damage:
    acc_or_damage = tree.xpath('/html/body/section/div/div/section/dl/dd[1]/text()')[-1]

    if acc_or_damage == 'None reported':
        prediction_df.at[0, "Accidents or damage"] = 0
    else:
        prediction_df.at[0, "Accidents or damage"] = 1


    #1-owner vehicle:
    owner_1 = tree.xpath('/html/body/section/div/div/section/dl/dd[2]/text()')[-1]

    if owner_1 == 'Yes':
        prediction_df.at[0, "1-owner vehicle"] = 1


    #Personal use only:
    personal_use = tree.xpath('/html/body/section/div/div/section/dl/dd[3]/text()')[-1]

    if personal_use == 'Yes':
        prediction_df.at[0, "Personal use only"] = 1


    # features

    all_features = []

    ul_elements = soup.select("ul.vehicle-features-list")
    for ul_element in ul_elements:
        li_elements = ul_element.select('li')
        for li_element in li_elements:
            all_features.append(li_element.text)

    for col in features:
        if col in all_features:
            prediction_df.at[0, col] = 1

    # Exterior color 

    ext_color = tree.xpath('/html/body/section/div/div/section/dl/dd[1]/text()')[0]

    if ("Satin" in ext_color) or ("Tintcoat" in ext_color) or ("Tri-Coat" in ext_color) or ("Carbon" in ext_color) or ("Ceramic" in ext_color):  
        prediction_df.at[0, "special_Paints"] = 1



    for col in paint_list:
        if col in ext_color:
            prediction_df.at[0, col] = 1

    if "Metallic" in ext_color:
        prediction_df.at[0, "is_Metallic"] = 1


    # Drivetrain 

    drivetrain = tree.xpath('/html/body/section/div/div/section/dl/dd[3]/text()')[0]
    drivetrain = drivetrain[1:]

    if drivetrain == 'FWD':
        drivetrain = 'Front-wheel Drive'
    elif drivetrain == 'RWD':
        drivetrain = 'Rear-wheel Drive'

    elif drivetrain in ['AWD', '4WD', 'All-wheel Drive']:
        drivetrain = 'Four-wheel Drive'

    elif drivetrain == '–':
        drivetrain = 'Unknown'
    else:
        None

    for col in drive_train_:
        col_ = col[11:]
        if col_ == drivetrain:
            print(col_)
            prediction_df[col] = 1


    # MPG 

    mpg = soup.select_one("span.sds-tooltip>span").text

    if "–" in str(mpg):
        value = str(mpg)
        value = value.replace("–", "-")  

        val_list = value.split("-")

        if val_list[0] != "":
            val_list[0] = float(val_list[0])
        else:
            val_list[0] = np.nan

        if val_list[1] != "":
            val_list[1] = float(val_list[1])
        else:
            val_list[1] = np.nan

        prediction_df.at[0, "min_MPG"] = min(val_list)
        prediction_df.at[0, "max_MPG"] = max(val_list)
    else:
        prediction_df.at[0, "min_MPG"] = float(value)
        prediction_df.at[0, "max_MPG"] = float(value)


    # Fuel type 

    fuel_type = tree.xpath('/html/body/section/div/div/section/dl/dd[5]/text()')[0]

    if fuel_type == 'Plug-In Hybrid':
        fuel_type = 'Hybrid'

    elif fuel_type in ['–', 'Other', 'Unspecified']:
        fuel_type = 'Unknown'

    else:
        None

    for col in fuel_type_:
        col_ = col[10:]
        
        if col_ == fuel_type:
            prediction_df.at[0, col] = 1


    # Transmission 

    transmission = tree.xpath('/html/body/section/div/div/section/dl/dd[6]/text()')[0]

    transmission = transmission.lower()

    val__ = transmission.split(" ")
    for val_ in val__:
        if "speed" in val_:
            val_ = val_.replace("speed", "")
            val_ = val_.replace("-", "")
            prediction_df.at[0, "speed_Transmission"] = int(val_)

    for val in transmission_list:
        new_name = val + "_transmission"
        if val in transmission:
            prediction_df.at[0 ,new_name] = 1
        else:
            None


    # Engine 

    engine = tree.xpath('/html/body/section/div/div/section/dl/dd[7]/text()')[0]
    engine = engine.split(" ")
    for value_engine_size in engine:
        if "L" in value_engine_size:
            prediction_df.at[0, "Engine Size"] = float(value_engine_size.replace("L",""))

    if "Turbo" in engine:
        prediction_df.at[0 ,"Turbo"] = 1

    if "GDI" in engine:
        prediction_df.at[0 ,"GDI_engine"] = 1

    if "MPFI" in engine:
        prediction_df.at[0 ,"MPFI_engine"] = 1

    if "DOHC" in engine:
        prediction_df.at[0 ,"DOHC_engine"] = 1

    for val in engine_list:
        val = val.lower()
        new_name = val + "_engine"
        if val in engine:
            prediction_df.at[0 ,new_name] = 1
        else:
            None

    prediction_df.isnull().sum()

    # Feature Engineering

    japan_cars = ['Honda', 'Toyota', 'Mazda', 'Mitsubishi', 'Nissan', 'Suzuki', 'Lexus']
    german_cars = ['Ford', 'Audi', 'BMW', 'MINI', 'Porsche', 'Mercedes-Benz', 'Volkswagen']
    italian_cars = ['FIAT', 'Maserati', 'Alfa']
    american_cars = ['Chevrolet', 'Jeep', 'Cadillac']
    lux_cars = ['Maserati', 'Lexus', 'Jaguar', 'Porsche', ]

    if brand in japan_cars:
        prediction_df.at[0, "japan_cars"] = 1

    if brand in german_cars:
        prediction_df.at[0, "german_cars"] = 1

    if brand in italian_cars:
        prediction_df.at[0, "italian_cars"] = 1

    if brand in american_cars:
        prediction_df.at[0, "american_cars"] = 1

    if brand in lux_cars:
        prediction_df.at[0, "lux_cars"] = 1

    prediction_df["NEW_Mileage_per_year"] = prediction_df["Mileage"] / prediction_df["Year"]

    prediction_df["feature_size"] = prediction_df[features].sum(axis=1)

    if owner_1 == "Yes":
        prediction_df["NEW_first_owner_no_acc"] = prediction_df["Accidents or damage"] + prediction_df["1-owner vehicle"]

        prediction_df["NEW_first_owner_no_acc"] = prediction_df["NEW_first_owner_no_acc"].replace({1: 1, 2: 0})

    prediction_df["NEW_mpg_mean"] = (prediction_df["min_MPG"] + prediction_df["max_MPG"]) / 2

    prediction_df["NEW_is_new_car"] = prediction_df["Year"].apply(lambda x : 1 if x > 2020 else 0)

    prediction_df["NEW_is_old_car"] = prediction_df["Year"].apply(lambda x : 1 if x < 2015 else 0)

    dv_val = drivetrain

    if dv_val == 'Front-wheel Drive':
        prediction_df.at[0, "NEW_fwd_turbo"] = 1 + prediction_df["Turbo"]
    else:
        prediction_df["NEW_fwd_turbo"] = 0 + prediction_df["Turbo"]

    prediction_df["NEW_fwd_turbo"] = prediction_df["NEW_fwd_turbo"].replace({2: 1, 1: 0})

    prediction_df["NEW_safety_size"] = prediction_df["Automatic Emergency Braking"] + prediction_df["Backup Camera"] + prediction_df["Stability Control"]

    prediction_df["NEW_Mileage"] = prediction_df["Mileage"].apply(lambda x: 9 if x < 10000
                          else 8 if (x < 20000) and (x > 10000)
                          else 7 if (x < 30000) and (x > 20000)
                          else 6 if (x < 40000) and (x > 30000)
                          else 5 if (x < 50000) and (x > 40000)
                          else 4 if (x < 60000) and (x > 50000)
                          else 3 if (x < 70000) and (x > 60000)
                          else 2 if (x < 80000) and (x > 70000)
                          else 1 if (x < 90000) and (x > 80000)
                          else 0)

    # X, y

    X = prediction_df.drop(["Primary Price"], axis=1)
    y = prediction_df["Primary Price"]
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X = X.to_numpy()

    with open(r"D:\miuul\final_project\car_prediction_model_project\pipeline\model.pkl", "rb") as f:
        model_dict = pickle.load(f)

    # Modeli ve özellik adlarını sözlükten almak için
    model = model_dict["model"]
    features_ = model_dict["features_"]    

    prediction = model.predict(X)

    print(f"\n\nThe list price of the {brand}, {model_} vehicle you are interested in '' ${y[0]}'' dir.\n\n")
    print(f"Based on our model's research, the price of the car you are interested in is '' ${int(prediction[0])} is predicted as ''.\n\n")

    if prediction[0] > y[0]:
        print(f"Based on the result of the prediction process; the list price of the vehicle you are interested in, \n compared to equivalent vehicles '' ${int(prediction[0])-y[0]} is as cheap as '''.")
        
    elif prediction[0] < y[0]:
        print(f"Based on the result of the prediction process, the list price of the vehicle you are interested in is '' ${y[0]-int(prediction[0])} is as expensive as '''.")
    
    else:
        print(f"According to the result of the prediction process; the list price and the predicted price of the car you are interested in are equal.")


def main():
    print("Note: The purpose of this model is to estimate the price of the vehicle you wish to purchase based on its characteristics.")
    print("In this way, you can find out whether the price of the car is higher or lower than it should be with the help of our model")
    print("and you can buy the car you are interested in at the best price...!!")
    url_ = input(r"Please enter the url address of the car you want to search for on cars.com :")
    pipeline(url=url_)

if __name__ == '__main__':
    main()