import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import torch 
import transformers 
import pickle
import scipy.spatial as sp
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers import GeoLMModel, GeoLMTokenizer, GeoLMForTokenClassification
from utils import sort_ref_closest_match, get_toponym_tokens
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
# st.write('Map title is:', title)

basemap_descrip = st.text_input('Basemap Description', 'Base from U.S. Geological Survey, 1967')
# st.write('Basemap description:', basemap_descrip)

w1 = st.button("Find Base Topo Map") 



query_sentence = title + ' ' + basemap_descrip 
# query_sentence = "Geologic Map of The Lake Helen Quadrangle, Big Horn and Johnson Counties, Wyoming. Base from U.S. Geological Survey, 1967"




if w1:
    topo_meta_path = 'support_data/historicaltopo.csv'
    topo_feat_path = 'support_data/cand_feats_geolm_mean.pkl'
    df = pd.read_csv(topo_meta_path) 

    with open(topo_feat_path, 'rb') as f: 
        cand_feat_list = pickle.load(f)

    model = GeoLMModel.from_pretrained("zekun-li/geolm-base-cased")


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    query_tokens, human_readable_tokens = get_toponym_tokens(query_sentence)
    print(human_readable_tokens)


    query_outputs = model(torch.unsqueeze(query_tokens,0))
    # query_feat = query_outputs['pooler_output'][0].detach().cpu().numpy()
    query_feat = torch.mean(query_outputs.last_hidden_state[0], axis = 0).detach().cpu().numpy()


    sim_matrix = 1 - sp.distance.cdist(np.array(cand_feat_list), np.array([query_feat]), 'cosine')

    map_rowid_list = range(0, len(cand_feat_list))
    # closest_match_scanid = sort_ref_closest_match(sim_matrix, map_scanid_list)
    closest_match_rowid = sort_ref_closest_match(sim_matrix, map_rowid_list)
        
    sorted_sim_matrix = np.sort(sim_matrix, axis = 0)[::-1] # descending order

    ret_dict = dict()
    # ret_dict['query_sentence'] = input_sentence
    ret_dict['closest_match_rowid'] = [a[0] for a in closest_match_rowid]
    ret_dict['sorted_sim_matrix'] = [a[0] for a in sorted_sim_matrix]

    print(ret_dict['closest_match_rowid'][:5])

    st.table(df.iloc[ret_dict['closest_match_rowid'][:5]])






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


