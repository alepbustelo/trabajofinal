import streamlit as st
import seaborn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import io 
pip install stremlit_option_menu
#1. Import
df = pd.read_csv('Train.csv')

#2. Sidebar
with st.sidebar:
    selected = option_menu(
        menu_title='Main menu',
        options=['Home', 'Plots', 'Model'],
    )

if selected == 'Home':
    st.title('Bienvenidos a Financial inclusion!')
    st.write('Vamos a trabajar con un dataset de inclusion financiera en africa')
    st.write("A continuacion podemos ver como esta compuesto de set de datos")
    st.write("Aqui tenemos un ejemplo de la informacion cruda disponible")
    st.dataframe(df.head())
    st.write("Como vemos hay varias columnas, algunas con datos numericos, otros categoricos y algunos redundantes.")
    st.subheader("\n Descripcion de columnas")
    st.markdown("\n **country** :  En qué ciudad vive")
    st.markdown("\n **bank_account** :  Si posee cuenta de banco o no -- objetivo")
    st.markdown("\n **location_type** :  Si vive en zona rural o urbana")
    st.markdown("\n **cellphone_access** :  Si tiene acceso a teléfono móvil")
    st.markdown("\n **household_size** :  Cantidad de personas que viven en el hogar")
    st.markdown("\n **age_of_respondent** :  Edad de la persona entrevistada")
    st.markdown("\n **gender_of_respondent** :  Género de la persona (Male, Female)")
    st.markdown("\n **relationship_with_head** :  Relación con el responsable de la casa")
    st.markdown("\n **education_level** :  Nivel de educación")
    st.markdown("\n **job_type** :  Tipo de trabajo del entrevistado")

elif selected == 'Plots':
    st.title('Plots')
    
    def countplot(df): 
        g = seaborn.countplot(data=df, y="household_size",palette="rainbow" , order = df['household_size'].value_counts().index).set_title("Cantidad de personas por vivienda",
                          fontdict = {'fontsize': 40,      
                                      'fontweight': 'bold', 
                                      'color': 'black'})      
        g.axes.set_ylim(14)
        seaborn.set(rc={"figure.figsize":(20, 8)})
        return g
    


elif selected == 'Model':
    st.title('Ready to try your model?')
    def inputs():
        st.sidebar.header('Model inputs')
        country = st.sidebar.selectbox('Country', df.country.unique())	
        location_type = st.sidebar.selectbox('Location type', df.location_type.unique())
        cellphone_access = st.sidebar.selectbox('Cellphone access', df.cellphone_access.unique())
        household_size = st.sidebar.number_input('Household size', 1) 
        age_of_respondent = st.sidebar.number_input('Age', 16) 
        gender_of_respondent = st.sidebar.selectbox('Gender', df.gender_of_respondent.unique())
        relationship_with_head = st.sidebar.selectbox('Relationship with head', df.relationship_with_head.unique())
        marital_status = st.sidebar.selectbox('Marital status', df.marital_status.unique()) 
        education_level = st.sidebar.selectbox('Education level', df.education_level.unique())
        job_type = st.sidebar.selectbox('Job type', df.job_type.unique()) 
        button = st.sidebar.button('Try model!')
        return country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button

    def get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type):
            data_inputs = {'country': country, 
                    'location_type': location_type, 
                    'cellphone_access': cellphone_access, 
                    'household_size': household_size, 
                    'age_of_respondent':age_of_respondent, 
                    'gender_of_respondent': gender_of_respondent, 
                    'relationship_with_head': relationship_with_head, 
                    'marital_status': marital_status, 
                    'education_level': education_level, 
                    'job_type': job_type}
            data= pd.DataFrame(data_inputs, index=[0])
            return data



    def print_results():
        country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type, button = inputs()
        if button:
            st.header('Probando el modelo con los datos ingresados')
            st.write('') 
            trial_data = get_data(country, location_type, cellphone_access, household_size, age_of_respondent, gender_of_respondent, relationship_with_head, marital_status, education_level, job_type)
            
            # imprimimos el df con los inputs
            st.write('Estos son los datos que ingresaste:')
            st.dataframe(trial_data)

            # LLamo al pkl con el modelo
            with open('./financial_inc.pkl', 'rb') as clf_inclusion:
                modelo_inclusion = pickle.load(clf_inclusion)
            # Prediccion usando el trial data con lo insertado en el form
            if modelo_inclusion.predict(trial_data) == 1:
                st.write('---')
                st.markdown('<h4 style="text-align: center">El individuo ya se encuentra bancarizado</h4>',unsafe_allow_html=True)
                st.write('---')
            else:
                st.write('---')
                st.markdown('<h4 style="text-align: center">El individuo no se encuentra bancarizado</h4>', unsafe_allow_html=True)
                st.write('---')

                


    if __name__ == '__main__':
        print_results()
