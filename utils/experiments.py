import yaml
import pickle
import psutil
from gensim.models.doc2vec import Doc2Vec


class ExpPar(object):
    def __init__(self, tag_pp, tag_setup):
        fn_tagged_words_list = './preproc_data/words_list_mecab_%s.pickle' % tag_pp
        fn_category_map = './preproc_data/category_map_%s.pickle' % tag_pp
        parfile = './params/doc2vec_par_%s.yaml' % tag_pp
        
        self.fn_model = './model/doc2vec_%s_setup%s.model' % (tag_pp, tag_setup)
        self.tagged_words_list = pickle.load(open(fn_tagged_words_list, 'rb'))
        self.dic_category_map = pickle.load(open(fn_category_map, 'rb'))
        self.params = yaml.load(open(parfile))[tag_setup]

    def train_model(self, save=True, show=True):
        opt_parnames =\
            ['vector_size', 'window', 'min_count', 'sample', 'alpha', 'min_alpha', 'epochs']
        model = Doc2Vec(documents=self.tagged_words_list, worker=psutil.cpu_count(), **self.params)

        if show:
            print('\nDoc2Vec training parameters:')
            for parname in opt_parnames:
                print('  %s: %s' % (parname, self.params[parname]))
        if save:
            model.save(self.fn_model)
        return model