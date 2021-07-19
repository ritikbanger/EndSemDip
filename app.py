import streamlit as st 
from PIL import Image
import pickle
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model


html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
        Transformation
         """)

Direction = st.selectbox('Y',('X', 'Y'))

Transformation = st.selectbox('Reflection',('Shearing', 'Scaling', 'Translation', 'Reflection'))
Y_factor = st.number_input('Insert a Y Factor',0,200)
X_Factor = st.number_input('Insert a X Factor',0,200)
file= st.file_uploader("Please upload image", type=("jpg", "png"))
img_T=file

def import_and_predict(image_data):
  if file is None:
    st.text("Please upload an Image file")
  else:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(img_T,caption='Uploaded Image.', use_column_width=True)

  if Transformation == "Translation":
    if Direction == "Y":
      M = np.float32([[1, 0, 20], 
                [0, 1, Y_factor], 
                [0, 0, 1]])
      img1 = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
    
    
  else:
    M = np.float32([[1, 0, X_Factor], 
                [0, 1, 100], 
                [0, 0, 1]])
    img1 = cv.warpPerspective(image, M, (image.shape[1], image.shape[0]))
  plt.imshow(img1)
  plt.show() 
  
  if Transformation == "Shearing":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M1 = np.float32([[1, Y_factor, 0],
             	[0, 1  , 0],
            	[0, 0  , 1]])
      sheared_img = cv.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
     
     
  else:
    M1 = np.float32([[1, 0, 0],
             	[X_Factor, 1  , 0],
            	[0, 0  , 1]])
    sheared_img = cv.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
  plt.imshow(sheared_img)
  plt.show()
  
  
  if Transformation == "Scaling":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M = np.float32([[1, 0  , 0],
            	[0,   Y_factor, 0],
            	[0,   0,   1]])
      scaled_img = cv.warpPerspective(image,M,(cols,rows))
    
    
  else:
    M = np.float32([[X_Factor, 0  , 0],
            	[0,   1, 0],
            	[0,   0,   1]])
    scaled_img = cv.warpPerspective(image,M,(cols,rows))
  plt.imshow(scaled_img)
  plt.show() 
  
  if Transformation == "Reflection":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M = np.float32([[-1,  0, cols],
                [0, -1, rows],
                [0,  0, 1   ]])
      reflected_img = cv.warpPerspective(image,M,(int(cols),int(rows)))   
  else:
    M = np.float32([[-1,  0, cols],
                [0, -1, rows],
                [0,  0, 1   ]])
    reflected_img = cv.warpPerspective(image,M,(int(cols),int(rows)))   
  plt.imshow(reflected_img)
  plt.show()
  

if st.button("Perform Transformation"):
  result=import_and_predict(image)

if st.button("About"):
  st.header(" Shubham Toshniwal")
  st.subheader("Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">RTU Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
