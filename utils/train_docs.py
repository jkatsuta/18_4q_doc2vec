import pickle
from gensim.models.doc2vec import Doc2Vec

def train(fn_words_list, fn_model, **kwargs):
    hyper_params = {'vector_size': 100, 'window': 15, 'min_count': 1}
    hyper_params.update(kwargs)

    with open(fn_words_list, 'rb') as f:
        tagged_words_list = pickle.load(f)

    model = Doc2Vec(documents=tagged_words_list, dm=0, worker=4, 
                    **hyper_params)
    model.save(fn_model)
    return tagged_words_list
