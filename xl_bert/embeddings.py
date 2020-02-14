import torch
from transformers import *
from keras.preprocessing.sequence import pad_sequences


def get_sentence_embedding(sentence_list,model_name='xlnet',model_dir='xlnet-base-cased',token_model_dir='xlnet-base-cased',n_layers=1,strategy="avg",max_len=50):
    '''

        :param sentence_list: list of sentences to be embedded
        :param model_name=name of model, can be xlnet or bert for now
        :param model_dir: directory path to model
        :param token_model_dir: directory path to tokenizer
        :param n_layers: number of layers you want to use to get embedding
        :param strategy: avg|catavg|avgcat|cat
        :param max_len: how many length you want to concatenate to
    :return: tensor of shape (len(sentence_list),max_len)
    '''
    if model_name=='xlnet':
      tokenizer = XLNetTokenizer.from_pretrained(token_model_dir)
      model = XLNetModel.from_pretrained(model_dir,output_hidden_states=True)
    elif model_name=='bert':
      tokenizer = BertTokenizer.from_pretrained(token_model_dir)
      model =   BertModel.from_pretrained(model_dir,output_hidden_states=True)
    else:
      return "No support currently for {}".format(model_name)
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentence_list]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")
    if model_name=='xlnet':
      last_state,layers, = model(torch.tensor(input_ids))
    else:
      last_state,_,layers= model(torch.tensor(input_ids))
    total_layers = len(layers)
    if strategy=="avg":
        ###avg all layers individually and then then average them together
        doc_vector=layers[total_layers-1].mean(dim=1)
        layer=total_layers-2
        for i in range(n_layers-1):
            this_layer_vector = layers[layer].mean(dim=1)
            doc_vector+=this_layer_vector
            layer+=-1
        return doc_vector/n_layers
    elif strategy=="cat":
        ## concat all the vectors of individual layer and then concatenate individual layers
        doc_vector=torch.reshape(layers[total_layers-1],(len(sentence_list),-1))
        layer=total_layers-2
        for i in range(n_layers-1):
            this_layer_vector=torch.reshape(layers[layer],(len(sentence_list),-1))
            doc_vector=torch.cat((doc_vector,this_layer_vector),dim=1)
            layer+=-1
        return doc_vector
    elif strategy=="avgcat":
        ## avg all the words of individual layer and then concat each layer
        doc_vector=layers[total_layers-1].mean(dim=1)
        layer=total_layers-2
        for i in range(n_layers-1):
            this_layer_vector = layers[layer].mean(dim=1)
            doc_vector=torch.cat((doc_vector,this_layer_vector),dim=1)
            layer+=-1
        return doc_vector
    else:
        #catavg
        ### concat all the vectors of individual layer and then averages them
        doc_vector=torch.reshape(layers[total_layers-1],(len(sentence_list),-1))
        layer = total_layers-2
        for i in range(n_layers-1):
            this_layer_vector =  torch.reshape(layers[layer],(len(sentence_list),-1))
            doc_vector+=this_layer_vector
            layer+=-1
        return doc_vector/n_layers