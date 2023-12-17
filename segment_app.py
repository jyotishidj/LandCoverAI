import streamlit as st  
import random
import os
from PIL import Image, ImageOps
import numpy as np
import cv2
import warnings
from segmentation import SegmentationPipeline
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

warnings.filterwarnings('ignore')

labels_cmap = matplotlib.colors.ListedColormap(["#000000", "#A9A9A9",
        "#8B8680", "#D3D3D3", "#FFFFFF"])

#os.system("python main.py")
os.system("dvc repro")


# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Land Cover Segmentation",
    page_icon = ":earth_asia:",
    initial_sidebar_state = 'auto'
)

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) # hide the CSS code from the screen as they are embedded in markdown text. Also, allow streamlit to unsafely process as HTML

with st.sidebar:
    st.image('land_cover.png')
    st.title("Semantic Segmentation of Land Cover")


st.sidebar.header("Choose your Model: ")
model = st.sidebar.selectbox("Pick your Model", ['Unet','Hypernet','Valuenet'])

st.sidebar.header("Choose Model Version: ")
version = st.sidebar.selectbox("Pick the Model version", [1])

st.write("""
         # Semantic Segmentation of Land Cover Usage
         """
         )

def segment(image_data,model_name='Unet',model_version=1):
        
        model_segment = SegmentationPipeline(model_name=model_name,model_version=model_version)
        mask = model_segment.segment(image_data)

        return mask

col1, col2 = st.columns((2))

mask = None
with col1:
    st.title('Land Cover Image')
    file = st.file_uploader("", type=["jpg", "png","tif"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        image = np.array(image)
        mask = segment(image,model,version)

if mask is not None: 
    with col2:
        st.title('Segmented Land Cover')
        fig, ax = plt.subplots()
        ax.imshow(mask, cmap = labels_cmap, interpolation = None, vmin = -0.5, vmax = 4.5)
        ax.axis("off")
        st.pyplot(fig)









    