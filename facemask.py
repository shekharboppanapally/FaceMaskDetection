import streamlit as st
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

st.title('MY MASK PROTECTS YOU,YOUR MASK PROTECTS ME')
st.subheader('#maskindia')
st.title('FACE MASK DETECTION APP')
st.text('upload an image')
model=pickle.load(open('facemask.p','rb'))

uploaded_file=st.file_uploader('choose an image',type='jpg')
if uploaded_file is not None:
  img=imread(uploaded_file)
  st.image(img,caption='uploaded image')
  if st.button('PREDICT'):
    CATAGORIES=['mask not weared','mask weared']
    st.write('Results')
    flat_data=[]
    img=np.array(img)
    img_resized=resize(img,(150,150,3))
    flat_data.append(img_resized.flatten())
    flat_data=np.array(flat_data)
    y_out=model.predict(flat_data)
    q=model.predict_proba(flat_data)
    for index,item in enumerate(CATAGORIES):
      st.write(f'{item} : {q[0][index]*100}')
    if y_out=='mask not weared':
      st.title('MASK NOT WEARED')
      st.subheader('WARNING!')
      st.write('PLEASE WEAR THE MASK')
    else:
      st.title('MASK WEARED')
      st.subheader('THANK YOU FOR WEARING THE MASK')  

st.write('HOW TO USE')
st.write('step1.first click on the option browswe files,it will open the camara and files in your mobile')
st.write('step2.choose any of the option camara or files and take a picture of your face')
st.write('step3.after loading the image,click on predict button then it will predict that mask is weared or not')
st.write('This is a face mask detection web app using streamlit made by shekhar_boppanapally')
st.write('It is made by using Support Vector Machine(SVM) Algorithm')
st.write('Accuracy of this model is :85%')
st.write('PLEASE WEAR A PROTECTIVE MASK TO PROTECT YOURSELF AND OTHERS')  