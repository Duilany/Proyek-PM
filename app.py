from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import json

app = Flask(__name__)

# Load models and data
gbr_model = joblib.load('gbr_model.pkl')
lr_model = joblib.load('lr_model.pkl')
cluster_preprocessor = joblib.load('cluster_preprocessor.pkl')
kmeans = joblib.load('kmeans.pkl')
rata_produksi_provinsi = joblib.load('rata_produksi_provinsi.pkl')

with open('provinsi_prod_map.json', 'r') as f:
    provinsi_prod_map = json.load(f)

with open('metadata.json', 'r') as f:
    metadata = json.load(f)
feature_columns = metadata['feature_columns']
provinsi_columns = metadata['provinsi_columns']
tahun_min = metadata['tahun_min']
tahun_max = metadata['tahun_max']
provinsi_list = metadata['provinsi_list']
cluster_labels = metadata['cluster_labels']
metrics = metadata['metrics']

@app.route('/')
def intro():
    return render_template('dashboard.html', metrics=metrics)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    tahun_list = list(range(tahun_min, tahun_max + 1))
    show_result = False
    prediction_gbr = None
    prediction_lr = None
    cluster_label = None
    avg_cluster_prod = None
    error = None

    if request.method == 'POST':
        try:
            provinsi = request.form['provinsi']
            tahun = int(request.form['tahun'])
            luas = float(request.form['luas'])
            hujan = float(request.form['hujan'])
            kelembapan = float(request.form['kelembapan'])
            suhu = float(request.form['suhu'])

            if not (500 <= hujan <= 5000 and 60 <= kelembapan <= 90 and 25 <= suhu <= 32 and luas > 0):
                raise ValueError("Input di luar rentang realistis: Curah Hujan (500-5000 mm), Kelembapan (60-90%), Suhu (25-32°C), Luas Panen (>0 ha).")

            input_data = pd.DataFrame({
                'Luas Panen': [luas],
                'Curah hujan': [hujan],
                'Kelembapan': [kelembapan],
                'Suhu rata-rata': [suhu],
                'Hujan_Kelembapan': [hujan * kelembapan],
                'Luas_Hujan_Ratio': [luas / (hujan + 1)],
                'Suhu_Squared': [suhu ** 2],
                'Luas_Suhu_Interaction': [luas * suhu],
                'Rata_Produksi_Provinsi': [rata_produksi_provinsi.mean()],  # Use mean as fallback
                'Tahun_Norm': [(tahun - tahun_min) / (tahun_max - tahun_min)],
                'Luas_Panen_Log': [np.log1p(luas)],
                'Curah_Hujan_Log': [np.log1p(hujan)]
            })

            for prov in provinsi_list:
                prov_col = f'Provinsi_{prov}'
                input_data[prov_col] = 1 if prov == provinsi else 0

            input_data = input_data[feature_columns]

            # Predictions
            prediction_gbr_log = gbr_model.predict(input_data)[0]
            prediction_gbr = np.expm1(prediction_gbr_log)
            prediction_lr_log = lr_model.predict(input_data)[0]
            prediction_lr = np.expm1(prediction_lr_log)

            # Clustering
            cluster_input = cluster_preprocessor.transform(input_data)
            cluster = kmeans.predict(cluster_input)[0]
            cluster_label = cluster_labels[str(cluster)]
            avg_cluster_prod = provinsi_prod_map[cluster_label]

            show_result = True

        except Exception as e:
            error = f"Error: {str(e)}"

    return render_template('index.html', provinsi_list=provinsi_list, tahun_list=tahun_list,
                           show_result=show_result, prediction_gbr=prediction_gbr, prediction_lr=prediction_lr,
                           cluster_label=cluster_label, avg_cluster_prod=avg_cluster_prod, error=error)

if __name__ == '__main__':
    app.run(debug=True)