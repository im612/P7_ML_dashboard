# frontend/main.py

import requests
import streamlit as st
# import pandas as pd
import json
# import pickle
import plotly.figure_factory as ff
import plotly.graph_objects as go
# import shap
# from streamlit_shap import st_shap
# import aws_session
from pathlib import Path
import sklearn
# from requests_toolbelt.multipart.encoder import MultipartEncoder

BASE_DIR = Path(__file__).resolve(strict=True).parent

exec(Path("main_backend.py").read_text(), globals())

# Streamlit
st.set_page_config(layout="wide", page_title="Tableau de bord crédit clients", page_icon="📂")
st.title("Tableau de bord crédit clients - Pret à dépénser")

# print('hi')

# BASE_DIR = Path(__file__).resolve(strict=True).parent

#
# if my_file.is_file():
#     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig_S3.csv")
# else:
#     test_df = pd.read_csv(f"{BASE_DIR}/test_split_orig.csv")

urlname=st.secrets['config']['API_URL']

@st.cache_data  # 👈 Add the caching decorator
# def load_data():
#     # colnames = pd.read_csv(f"{BASE_DIR}/model_frontend/colnames.csv").columns.to_list()
#
#     # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig2.csv")
#     # # test_df = pd.read_csv(f"{BASE_DIR}/model_frontend/test_split_orig.csv")
#     # test_df = pd.DataFrame(test_df, columns=colnames)
#     # test_df['SK_ID_CURR'] = test_df['SK_ID_CURR'].astype(int)
#
#     # indnames = pd.DataFrame(test_df, columns=['SK_ID_CURR']).astype(int).values
#     # indnames = requests.post(url=f"https://p7a.herokuapp.com:8081/indnames")
#     indnames = requests.post(url=f"{urlname}/indnames")
#     # indnames = requests.post(url=URL)
#     # indnames = requests.post(url=f"http://p7a.herokuapp.com:8080/indnames")
#     # indnames = requests.post(url=f"http://im612-p7-deploy-main-9v49yi.streamlit.app:8080/indnames")
#
#     return indnames

# colnames, test_df, indnames = load_data()
# response = load_data()
# indnames = requests.post(url=f"{urlname}/indnames")

response = requests.post(url=f"{urlname}/indnames")
objind = response.json()
indnames = objind['listindnames']
#
# st.write(indnames)


# df = load_data("https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv")
# st.dataframe(df)

# Alleggerimento 2
# # SELECTION NUMERO CLIENT
id = st.selectbox("Saisir le code client :", [i for i in indnames])
st.header(f'Code client: {str(int(id))}')
#
# # APPEL AUX ENDPOINTS
# # https://stackoverflow.com/questions/72060222/how-do-i-pass-args-and-kwargs-to-a-rest-endpoint-built-with-fastapi
# q = {"id" : f'{id.tolist()[0]}'}
# q = {"id" : f"{id, id['id']}"}
# q = {"id" : f'{id["id"]}'}
st.write(id)
q = id
qj = json.dumps(q)
# # https://stackoverflow.com/questions/64057445/fast-api-post-does-not-recgonize-my-parameter
#
# # interact with FastAPI endpoint
#
# # fireto = 'fastapi'
# fireto = '0.0.0.0'
# # fireto = 'backend'
#
response = requests.post(url=f"{urlname}/probability", data=qj)
# # # response = requests.post(url=f"http://86.214.128.9:8080/probability", data=qj)
objind = response.json()
# prob = objind['probability']
st.write(objind)
# st.write(objind, prob)

# # #
# response = requests.post(url=f"{urlname}/prediction", data=qj)
# obj2 = response.json()
# pred = obj2['prediction']
# #
# response = requests.post(url=f"{urlname}/seuil", data=qj)
# obj3 = response.json()
# seuil = obj3['seuil']
# #
# st.divider()

# # ALLEGGERIMENTO 1
# # # col1, col2, col3 = st.columns(3)
# # col1, col3 = st.columns(2)
# # col1.metric("Code client", "%d" % id)
# # # col2.metric("Prédiction", "%d" % pred )
# # col3.metric("Probabilité de non solvabilité", "%.2f" % prob, "%.2f" % (seuil - prob))
# #
# # if pred < seuil:
# #     st.header('Le crédit est accordé :+1:')
# #     # https: // docs.streamlit.io / library / api - reference  # display-text
# # else:
# #     st.header('Le crédit est decliné :-1:')
# # st.write('Le crédit est refusé car la probabilité de non solvabilité dépasse %.2f' % seuil)
# # #
# # # Gauge chart
# # # https://plotly.com/python/gauge-charts/
# # # https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart
# #
# # probfig = float("%.2f" % prob)
# #
# # fig = go.Figure(go.Indicator(
# #     domain = {'x': [0, 1], 'y': [0, 1]},
# #     value = probfig,
# #     mode = "gauge+number",
# #     title = {'text': "Probabilité de non solvabilité"},
# #     delta = {'reference': 0.9},
# #     gauge = {'axis': {'range': [0.0, 1.0]},
# #              'steps' : [
# #                  {'range': [0.0, 0.9], 'color': "lightgreen"},
# #                  {'range': [0.9, 1.0], 'color': "red"}],
# #              'threshold' : {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value': probfig}}))
# #
# # st.plotly_chart(fig, use_container_width=True)
# #
# #
# # # Explainer
# # with open(f"{BASE_DIR}/model_frontend/explainer.pkl", "rb") as f:
# #     explainer = pickle.load(f)
# # f.close()
# #
# # #Jeu de données
# # X = test_df.drop(columns=['SK_ID_CURR', 'TARGET'])
# # # X = X.head(1000)
# #
# # ind = indnames.tolist().index(id)
# # # print(indnames)
# #
# # # shap_go = 1
# # shap_go = 0
# #
# # def ssp() # per la cache
# #
# # if shap_go == 0:
# #     with st.spinner('Je récupère les facteurs déterminants...'):
# #         shap_values = explainer(X)
# #     st.success('Fini ')
# #
# #     st.header('Facteurs globalement plus significatifs ')
# #     st_shap(shap.summary_plot(shap_values, X), height=800, width=2000)
# #
# #     st.header('Facteurs déterminants pour ce profil')
# #     st_shap(shap.plots.waterfall(shap_values[ind]), height=800, width=2000)
# #
# # st.divider()
# #
# # # Distribution des facteurs détérminants
# #
# # import numpy as np
# # # st.header(indnames, colnames)
# # # st.header(ind)
# # # st.header(shap_values[ind])
# # # st.header(shap_values.data[ind])
# #
# # top_shap = X.columns[np.argsort(np.abs(shap_values.values[ind]))[::-1][:9]]
# # ind_top_shap = np.argsort(np.abs(shap_values.values[ind]))[::-1][:9]
# # # https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order
# #
# # import plotly.graph_objects as go
# # import matplotlib.pyplot as plt
# #
# # st.header('Distribution des facteurs déterminants')
# #
# # # SELECTION NUMÉRO CLIENT
# # # for fi in range(0, len(top_shap)):
# # #     st.subheader(f'Nom variable: {top_shap[fi]}')
# # #     val_feature = '%.3f' % float(shap_values.data[ind][ind_top_shap[fi]])
# # #     shap_feature = float(shap_values.values[ind][ind_top_shap[fi]])
# # #
# # #     if shap_feature > 0:
# # #         st.subheader(f':warning: Contribution positive (%.2f): risque augmenté' % shap_feature)
# # #     elif shap_feature < 0:
# # #         st.subheader(f'Contribution négative (%.2f): risque diminué' % shap_feature)
# # #
# # #     data = X[top_shap[fi]]
# # #     n, _ = np.histogram(data)
# # #     fig, ax = plt.subplots()
# # #     _, _, bar_container = ax.hist(data,
# # #                                   fc="c", alpha=0.5)
# # #     media = data.mean()
# # #     media_acc = '%.2f' % media
# # #     mediana = data.median()
# # #     mediana_acc = '%.2f' % mediana
# # #     val_feature_acc = '%.2f' % float(val_feature)
# # #
# # #     plt.axvline(media, color='blue', linestyle='dashed', linewidth=1, alpha=0.5, label=f'moyenne : {media_acc}')
# # #     plt.axvline(mediana, color='darkgreen', linestyle='dashed', linewidth=1, alpha=0.5, label = f'mediane : {mediana_acc}')
# # #     plt.axvline(val_feature, color='red', linestyle='solid', linewidth=1, alpha=0.5, label = f'valeur client : {val_feature_acc}')
# # #
# # #     ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01),
# # #               ncol=3, fancybox=True)
# # #     plt.figure(figsize=(0.8, 0.8))
# # #     st.pyplot(fig=fig, use_container_width=False)
# # #     st.divider()
