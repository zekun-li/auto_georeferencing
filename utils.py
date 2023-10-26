import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pandas as pd
from rank_bm25 import BM25Okapi

def prepare_bm25(df):
    

    cand_text_list = []
    map_rowid_list = []
    for index, row in df.iterrows():
        # cand_text = row['map_name'] + row['primary_state'] + row['product_filename']
        cand_text = row['primary_state'] + ' ' + row['map_name']  

        if not pd.isnull(row['state_list']): 
            cand_text = cand_text + ' ' + row['state_list']

        if not pd.isnull(row['county_list']): 
            cand_text = cand_text + ' ' + row['county_list']
        
        cand_text_list.append(cand_text)
        map_rowid_list.append(index)



    corpus = cand_text_list

    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25

def sort_ref_closest_match(sim_matrix, word_list):
    '''
    sim_matrix should be (n_ref, n_query)
    word_list should be (n_ref,)
    '''
    n_ref, n_query = sim_matrix.shape[0], sim_matrix.shape[1]
    
    indices_list = np.argsort(sim_matrix, axis = 0)[::-1] # descending order
    
    #print(indices_list)
    ret_list = []
    for indices in indices_list:
        word_sorted = []
        for ind in indices:
            word_sorted.append(word_list[ind])
        ret_list.append(word_sorted)
    return ret_list


def generate_human_readable(tokens):
    ret = []
    for t in tokens:
        if t == '[SEP]':
            continue

        if t.startswith("##"):
            assert len(ret) > 0
            ret[-1] = ret[-1] + t.strip('##')
        else:
            ret.append(t)

    return ret

def get_toponym_tokens(query_sentence, device):

    model_name = "zekun-li/geolm-base-toponym-recognition"
    # Load tokenizer and topo_model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    topo_model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    # tokenizer = GeoLMTokenizer.from_pretrained(model_name)
    # topo_model = GeoLMForTokenClassification.from_pretrained(model_name)
    
    topo_model.to(device)
    topo_model.eval()

    
    tokens = tokenizer.encode(query_sentence, return_tensors="pt").to(device)
    # Pass tokens through the topo_model
    outputs = topo_model(tokens) 
    # Retrieve predicted labels for each token
    predicted_labels = torch.argmax(outputs.logits, dim=2)


    query_tokens = tokens[0][torch.where(predicted_labels[0] != 0)[0]]

    human_readable = generate_human_readable(tokenizer.convert_ids_to_tokens(query_tokens))
    return query_tokens, human_readable