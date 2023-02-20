import streamlit as st
import seaborn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle



"Vamos a trabajar con un dataset de inclusion sobre inclusion financiera en africa"


"A continuacion podemos ver como esta compuesto de set de datos"


df = pd.read_csv('Train.csv')

"Aqui tenemos un ejemplo de la informacion cruda disponible"

df



"Como vemos hay varias columnas, algunas con datos numericos, otros categoricos y algunos redundantes."

st.write("GRAFICO ILUSTRATIVO")

seaborn.countplot(data=data, y="marital_status",palette="rainbow" , order = data['marital_status'].value_counts().index).set_title("Estado civil",
                  fontdict = {'fontsize': 40,       # Tamaño
                              'fontweight': 'bold', # Estilo
                              'color': 'black'})      # Color

