import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
country_label = data['country_label']
edu_label = data['edu_label']

def show_predict_page():
    st.title('Software Developers Salary Prediction')

    st.write('''#### Predict the salary of various salaries depending on **Country**, **Education Level** and **Experience**''')

    countries = (
        'United States of America',                                                                              
        'Germany',                                                 
        'United Kingdom of Great Britain and Northern Ireland',   
        'India',                                                   
        'Canada',                                                  
        'France',                                                  
        'Brazil',                                                  
        'Spain',                                                    
        'Netherlands',                                              
        'Australia',                                                
        'Italy',                                                    
        'Poland',                                                   
        'Sweden',                                                   
        'Russian Federation',                                       
        'Switzerland',                                              
        'Turkey',                                                   
        'Israel',                                                   
        'Austria',                                                  
        'Norway',                                                   
        'Portugal',                                                 
        'Denmark',                                                  
        'Belgium',                                                  
        'Finland',                                                  
        'Mexico',                                                   
        'New Zealand',                                              
        'Greece',                                                   
        'South Africa',
    )

    education = (
        'Less than a Bachelors',
        "Bachelor's degree",
        "Master's degree",
        'Post Grad',  
    )

    country = st.selectbox('Country', countries)
    education = st.selectbox('Education Level', education)

    experience = st.slider('Years of Experience', 0, 50, 1)

    ok = st.button('Calculate Salary')
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = country_label.transform(X[:,0])
        X[:, 1] = edu_label.transform(X[:,1])
        X = X.astype(float)

        salary = model.predict(X)
        st.subheader(f'The estimated salary is ${salary[0]:.2f}')
