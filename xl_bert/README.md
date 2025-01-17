*************
Documentation
*************
This is the repository of the xl_bert package. This library helps you find sentence embedding for your sentence using SOTA language models such as Bert and XLNET.
    
    This library takes in an input seven paramters :
    
    **sentence_list,model_name,model_dir,token_model_dir,n_layers,strategy,max_len**. Each parameter has been explained below.

    **sentence_list** : List of sentences you want to get embedding of. 
    
    **model_name** : Name of model which you want to use, currently can be 'bert' or 'xlnet'.
    
    **model_dir** : Directory path of pretrained/finetuned Bert/XLNet language model. Default is 'xlnet-base-cased'. Pretrained language models can be seen from here:
    

    <http://huggingface.co/transformers/pretrained_models.html>`_


    **token_model_dir** : Directory path of tokenizer. Default is 'xlnet-base-cased'


    **n_layers** :  Number of layers you want to use to get sentence embedding.Default is 1
    
    **Strategy** : This is where it gets interesting. Strategy is categorised in four choices.

        'avg': We average each layer individually and then average n_layers.
        'cat': We concatenate each layer individually, then we concatenate n_layers
        'avgcat': We average each layer individually and then concat n_layers
        'catavg': We concat each individual layer and then average n_layers. 
      
    **max_len** : Maximum length of sentence you want. Default 50

================
 Installation
================
    pip install xl_bert


**Usage with Bert as well as XLNet**

    get_sentence_embedding(['I am playing','let me dance'],model_name='xlnet',model_dir='xlnet- 
    large-cased',token_model_dir='xlnet-large-cased',n_layers=2,strategy='avg',max_len=50) 

**Contribution**

 Package author and current maintainer is Shivam Panwar (panwar.shivam199@gmail.com); You are more than welcome to approach him for help. Contributions are very welcomed, especially since this package is very much in its infancy.


Created by Shivam Panwar (panwar.shivam199@gmail.com)