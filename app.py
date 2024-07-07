
import streamlit as st
import numpy as np
import pickle
import warnings
import sklearn

warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

st.set_page_config(page_title="HEART DISEASE PREDICTION", page_icon="https://your-icon-url.png", layout='centered',
                   initial_sidebar_state="collapsed")


def load_model(modelfile):
    loaded_model = pickle.load(open(modelfile, 'rb'))
    return loaded_model


def main():
    # title
    html_temp = """
    <div>
    <h1 style="color:MEDIUMSEAGREEN;text-align:center;"> HEART DISEASE PREDICTION 💓 </h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    col = st.columns(1)[0]

    with col:
        st.subheader("Check if you're at risk of heart disease ❤️")
        age = st.number_input("Age", 0, 150)
        sex = st.selectbox("Sex", ["Male", "Female"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 50, 250)
        chol = st.number_input("Serum Cholesterol (mg/dl)", 50, 700)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
        restecg = st.selectbox("Resting Electrocardiographic Results",
                               ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
        thalach = st.number_input("Maximum Heart Rate Achieved", 50, 250)
        exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
        oldpeak = st.number_input("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0)
        slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
        ca = st.number_input("Number of Major Vessels Colored by Flourosopy", 0, 4)
        thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

        if st.button('Predict'):
            loaded_model = load_model('heart_model.pkl')
            sex = 1 if sex == "Male" else 0
            cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-Anginal Pain": 2, "Asymptomatic": 3}
            cp = cp_dict[cp]
            fbs = 1 if fbs == "True" else 0
            restecg_dict = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
            restecg = restecg_dict[restecg]
            exang = 1 if exang == "Yes" else 0
            slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
            slope = slope_dict[slope]
            thal_dict = {"Normal": 0, "Fixed Defect": 1, "Reversible Defect": 2}
            thal = thal_dict[thal]

            single_pred = np.array(
                [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
            prediction = loaded_model.predict(single_pred)

            result = "You're at risk of heart disease! 😟" if prediction == 1 else "You're not at risk of heart disease. 😀"
            st.write('''
            ## Results 📊
            ''')
            st.success(result)

    hide_menu_style = """
    <style>
    .block-container {padding: 2rem 1rem 3rem;}
    #MainMenu {visibility: hidden;}
    </style>
    """

    st.markdown(hide_menu_style, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
