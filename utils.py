import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

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

def get_toponym_tokens(query_sentence):

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


    return query_tokens, tokenizer.convert_ids_to_tokens(query_tokens)