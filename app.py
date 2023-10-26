import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

from utils import get_toponym_tokens, prepare_bm25
import torch
import pdb 



# main title
st.title("Geologic Map Auto Georeferencing") 
# subtitle
st.markdown("### Upload Map")



file = st.file_uploader(label = "Upload your map", type=['png', 'jpg', 'jpeg', '.tif'])


if file:
    image = Image.open(file) # read image with PIL library
    st.image(image) #display


st.markdown("### Enter Title and Basemap Description")



title = st.text_input('Map title', 'Geologic Map of The Lake Helen Quadrangle, Big Horn and Johnson Counties, Wyoming')


basemap_descrip = st.text_input('Basemap Description', 'Base from U.S. Geological Survey, 1967')


query_sentence = title + ' ' + basemap_descrip 
# query_sentence = "Geologic Map of The Lake Helen Quadrangle, Big Horn and Johnson Counties, Wyoming. Base from U.S. Geological Survey, 1967"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

@st.cache_data
def load_data(topo_histo_meta_path, topo_current_meta_path):

    # topo_histo_meta_path = 'support_data/historicaltopo.csv'
    df_histo = pd.read_csv(topo_histo_meta_path) 
            
    # topo_current_meta_path = 'support_data/ustopo_current.csv'
    df_current = pd.read_csv(topo_current_meta_path) 


    common_columns = df_current.columns.intersection(df_histo.columns)

    df_merged = pd.concat([df_histo[common_columns], df_current[common_columns]], axis=0)
            
    bm25 = prepare_bm25(df_merged)

    return bm25, df_merged

bm25, df_merged = load_data(topo_histo_meta_path = 'support_data/historicaltopo.csv',
    topo_current_meta_path = 'support_data/ustopo_current.csv')


# def click_first_button():
#     st.session_state.first.clicked = True

# def click_second_button():
#     st.session_state.second.clicked = True

w1 = st.button("Start Processing")

if st.session_state.get('w1') != True:

    st.session_state['w1'] = w1


if st.session_state['w1'] == True:

    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence, device)
    # print(human_readable_tokens)

    st.write('Detected toponyms:', human_readable_tokens)

    query_sent = ' '.join(human_readable_tokens)

    # w2 = st.button("Find Base Topo Map") 

    # if st.session_state.get('w2') != True:

    #     st.session_state['w2'] = w2

    
    # if st.session_state['w2'] == True:
        

    tokenized_query = query_sent.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    sorted_bm25_list = np.argsort(doc_scores)[::-1]


    
    # st.table(df_merged.iloc[sorted_bm25_list[0:10]])
    st.dataframe(df_merged.iloc[sorted_bm25_list[0:10]])






        # # it will only detect the English and Turkish part of the image as text
        # reader = easyocr.Reader(['tr','en'], gpu=False) 
        # result = reader.readtext(np.array(image))  # turn image to numpy array

        # textdic_easyocr = {} 
        # for idx in range(len(result)): 
        #     pred_coor = result[idx][0] 
        #     pred_text = result[idx][1] 
        #     pred_confidence = result[idx][2] 
        #     textdic_easyocr[pred_text] = {} 
        #     textdic_easyocr[pred_text]['pred_confidence'] = pred_confidence

        # # create a dataframe which shows the predicted text and prediction confidence
        # df = pd.DataFrame.from_dict(textdic_easyocr).T
        # st.table(df)


        # def rectangle(image, result):
        #     # https://www.blog.pythonlibrary.org/2021/02/23/drawing-shapes-on-images-with-python-and-pillow/
        #     """ draw rectangles on image based on predicted coordinates"""
        #     draw = ImageDraw.Draw(image)
        #     for res in result:
        #         top_left = tuple(res[0][0]) # top left coordinates as tuple
        #         bottom_right = tuple(res[0][2]) # bottom right coordinates as tuple
        #         draw.rectangle((top_left, bottom_right), outline="blue", width=2)
        #     #display image on streamlit
        #     st.image(image)

        # dataframe = pd.DataFrame(
        #     np.random.randn(10, 20),
        #     columns=('col %d' % i for i in range(20)))
        # st.table(dataframe)


