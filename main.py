
# A very simple Flask Hello World app for you to get started with...

from flask import Flask
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ppca import PPCA as PPCA
from os.path import dirname, join
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.mixture import GaussianMixture
import json

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Ini adalah aplikai klasifikasi mangga deployed!'


@app.route('/predict_std')
def predict_std():
    filename = join(dirname(__file__), "Dataset_RawSpectrum_NIRS_for_Intact_Mangoes.xlsx")
    df_original=pd.read_excel(filename)
    df = df_original
    df = df.dropna()
    df = df.drop("No", axis=1)
    df = df.drop("Mango Cultivars", axis=1)
    df = df.drop("label", axis=1)
    y = df_original["label"]
    y.to_list()
    scaler=StandardScaler()
    scaler.fit(df.values)
    scaled_data=scaler.transform(df.values)
    std_data_transpose = np.transpose(scaled_data)
    ppca_std = PPCA()
    ppca_std_data = ppca_std.fit(std_data_transpose)
    y_to_float = []
    for classes in y:
        if classes == "Cengkir":
            y_to_float.append(0)
        if classes == "Kweni":
            y_to_float.append(1)
        if classes == "Kent":
            y_to_float.append(2)
        if classes == "Palmer":
            y_to_float.append(3)
    ppca_std_transform = ppca_std.transform()
    ppca_std_tp = np.transpose(ppca_std_transform)
    em_std = GaussianMixture(n_components = 4)
    em_std.fit(ppca_std_tp)
    em_pred_std = em_std.predict(ppca_std_tp)
    y_to_name_std = []
    for classes in em_pred_std:
        if classes == 0:
            y_to_name_std.append("Cengkir")
        if classes == 1:
            y_to_name_std.append("Kweni")
        if classes == 2:
            y_to_name_std.append("Kent")
        if classes == 3:
            y_to_name_std.append("Palmer")
    df_std_ppca = pd.DataFrame(ppca_std_tp, columns = ["PC 1", "PC 2"])
    df_std_ppca['label'] = y_to_name_std
    df_std_ppca = df_std_ppca.sample(frac = 1)
    x_temp = df_std_ppca.drop("label", axis=1)
    y_temp = df_std_ppca["label"]
    x_std_train, x_std_test, y_std_train, y_std_test = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)
    std_svm = SVC(gamma='auto')
    std_svm.fit(x_std_train, y_std_train)
    y_std_pred = std_svm.predict(x_std_test)
    output = {
        "hasil": classification_report(y_std_test, y_std_pred),
    }
    return json.dumps(output)

@app.route('/predict_minmax')
def predict_minmax():
    filename = join(dirname(__file__), "Dataset_RawSpectrum_NIRS_for_Intact_Mangoes.xlsx")
    df_original=pd.read_excel(filename)
    df = df_original
    df = df.dropna()
    df = df.drop("No", axis=1)
    df = df.drop("Mango Cultivars", axis=1)
    df = df.drop("label", axis=1)
    y = df_original["label"]
    y.to_list()
    scaler=MinMaxScaler()
    scaler.fit(df.values)
    scaled_data=scaler.transform(df.values)
    minmax_data_transpose = np.transpose(scaled_data)
    ppca_minmax = PPCA()
    ppca_minmax_data = ppca_minmax.fit(minmax_data_transpose)
    y_to_float = []
    for classes in y:
        if classes == "Cengkir":
            y_to_float.append(0)
        if classes == "Kweni":
            y_to_float.append(1)
        if classes == "Kent":
            y_to_float.append(2)
        if classes == "Palmer":
            y_to_float.append(3)
    ppca_minmax_transform = ppca_minmax.transform()
    ppca_minmax_tp = np.transpose(ppca_minmax_transform)
    em_minmax = GaussianMixture(n_components = 4)
    em_minmax.fit(ppca_minmax_tp)
    em_pred_minmax = em_minmax.predict(ppca_minmax_tp)
    y_to_name_minmax = []
    for classes in em_pred_minmax:
        if classes == 0:
            y_to_name_minmax.append("Cengkir")
        if classes == 1:
            y_to_name_minmax.append("Kweni")
        if classes == 2:
            y_to_name_minmax.append("Kent")
        if classes == 3:
            y_to_name_minmax.append("Palmer")
    df_minmax_ppca = pd.DataFrame(ppca_minmax_tp, columns = ["PC 1", "PC 2"])
    df_minmax_ppca['label'] = y_to_name_minmax
    df_minmax_ppca = df_minmax_ppca.sample(frac = 1)
    x_temp = df_minmax_ppca.drop("label", axis=1)
    y_temp = df_minmax_ppca["label"]
    x_minmax_train, x_minmax_test, y_minmax_train, y_minmax_test = train_test_split(x_temp, y_temp, test_size=0.3, random_state=42)
    minmax_svm = SVC(gamma='auto')
    minmax_svm.fit(x_minmax_train, y_minmax_train)
    y_minmax_pred = minmax_svm.predict(x_minmax_test)
    output = {
        "hasil": classification_report(y_minmax_test, y_minmax_pred),
    }
    return json.dumps(output)




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)