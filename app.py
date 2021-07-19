import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
#import tensorflow as tf
#from keras.preprocessing import image
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
#from keras.models import load_model

html_temp = """
   <div class="" style="background-color:salmon;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Digital Image Processing End-Term Examination</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Ritik Banger - PIET18CS124</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
 Transformations on an image and Translation (-100 in X and -50 in Y)
         """
         )


img1= st.file_uploader("Please upload image 1", type=("jpg", "png"))
option = st.selectbox('Choose Appropriate option',('Reflection', 'Scaling', 'Shearing', 'Translation in x', 'Translation in y'))
Direction = st.selectbox('Y',('X', 'Y'))
Y_factor = st.number_input('Insert a Y Factor',0,200)
X_Factor = st.number_input('Insert a X Factor',0,200)

if img1 is None:
  st.text("Please upload an Image 1")
else:
  file_bytes = np.asarray(bytearray(img1.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(img1,caption='Uploaded Image 1', use_column_width=True)


st.write('You selected:', option)


from  PIL import Image, ImageOps
def import_and_predict():
  file_bytes1 = np.asarray(bytearray(img1.read()), dtype=np.uint8)
  opencv_image1 = cv2.imdecode(file_bytes1, 1)
  image = cv2.resize(opencv_image1,(300,300))
  if option == "Shearing":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M1 = np.float32([[1, Y_factor, 0],[0, 1  , 0],[0, 0  , 1]])
      sheared_img = cv2.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
      st.image(sheared_img,  use_column_width=True)
    else:
      M1 = np.float32([[1, 0, 0],[X_Factor, 1  , 0],[0, 0  , 1]])
      sheared_img = cv2.warpPerspective(image,M1,(int(cols*1.5),int(rows*1.5)))
      st.image(sheared_img,  use_column_width=True)

  if option == "Scaling":
    rows, cols, dim = image.shape
    if Direction == "Y":
       M = np.float32([[1, 0  , 0],[0,   Y_factor, 0],[0,   0,   1]])
       scaled_img = cv2.warpPerspective(image,M,(cols,rows))
       st.image(scaled_img,  use_column_width=True)
    else:
      M = np.float32([[X_Factor, 0  , 0],[0,   1, 0],[0,   0,   1]])
      scaled_img = cv2.warpPerspective(image,M,(cols,rows))
      st.image(scaled_img,  use_column_width=True)

  if option == "Reflection":
    rows, cols, dim = image.shape
    if Direction == "Y":
      M = np.float32([[-1,  0, cols],[0, -1, rows],[0,  0, 1   ]])
      reflected_img = cv2.warpPerspective(image,M,(int(cols),int(rows)))
      st.image(reflected_img,  use_column_width=True)
    else:
      M = np.float32([[-1,  0, cols],[0, -1, rows],[0,  0, 1   ]])
      reflected_img = cv2.warpPerspective(image,M,(int(cols),int(rows)))
      st.image(reflected_img,  use_column_width=True)

  if option == "Translation in x":
     M = np.float32([[1, 0, -100], [0, 1, 20], [0, 0, 1]])
     img3 = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
     st.image(img3,  use_column_width=True)
     
  else:
     M = np.float32([[1, 0, -50], [0, 1, 20], [0, 0, 1]])
     img4 = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
     
     st.image(img4,  use_column_width=True)
  return 0 
    
if st.button("Click To Perform Operation"):
  result=import_and_predict()
  
if st.button("About"):
  st.header("Ritik Banger")
  st.subheader("PIET18CS124, Department of Computer Engineering, PIET")
html_temp = """
   <div class="" style="background-color:white;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:black;margin-top:10px;">Digital Image processing EndTerm Lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
