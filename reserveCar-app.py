import streamlit as st
import pandas as pd
import pickle
from sklearn import datasets

st.write("""
# ต้นทุนการใช้รถ
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    area = st.selectbox('พื้นที่', ('กรุงเทพฯและปริมณฑล', 'ต่างจังหวัด'))
    license_plate = st.sidebar.slider('ทะเบียน', 0, 47, 0,step=1)
    passenger = st.sidebar.slider('จำนวนผู้โดยสาร', 0, 120, 0,step=1)
    fuel = st.selectbox('ประเภทเชื้อเพลิง', ('NGV', 'NGV, แก๊สโซฮอล 95','แก๊สโซฮอล 95','แก๊สโซฮอล 95, เบนซิน','ดีเซล'))
    timing = st.sidebar.slider('ระยะเวลาการใช้บริการ', 0, 1000000, 0,step=1)

    data = {'area': area,
            'license_plate': license_plate,
            'passenger': passenger,
            'fuel': fuel,
            'timing': timing}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

# iris = datasets.load_iris()

clf = pickle.load(open('multilinear.pkl', 'rb'))

prediction = clf.predict(df)

if(prediction < 1):
    prediction = 0


st.subheader('forecast ต้นทุน')
st.write(prediction)

