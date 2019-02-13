import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from collections import defaultdict, Counter
from sklearn.metrics import classification_report

import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.callbacks import EarlyStopping


class evaluation_unlabeled_docs(object):
    def __init__(self, model, tagged_words_list, n_clusters):
        self.model = model
        self.n_clusters = n_clusters
        self.dvecs = self.model.docvecs.vectors_docs
        self.tagged_words_list = tagged_words_list
        self.tag_list = self.get_tag_list(tagged_words_list)

        self.kmeans =\
            KMeans(n_clusters=self.n_clusters, random_state=0).fit(self.dvecs)
        self.labels = self.kmeans.labels_

    def print_similarities(self, tags, mode, n_show=5):
        print('\n===== %s similar words list =====\n' % mode)
        for tag in tags:
            print('-' * 50)
            print(tag, self.tagged_words_list[int(tag.lstrip('t'))])
            print('-' * 50, end='\n\n')
            
            if mode == 'most':
                for t, sim in self.model.docvecs.most_similar(tag, topn=n_show):
                    print(t, sim, self.tagged_words_list[int(t.lstrip('t'))], end='\n\n')
            elif mode == 'random':
                random_tags = self.get_random_tags(n_show)
                for t in random_tags:
                    sim = self.model.docvecs.similarity(tag, t)
                    print(t, sim, self.tagged_words_list[int(t.lstrip('t'))], end='\n\n')

    def print_each_cluster(self, n_each_show=5):
        for n in range(self.n_clusters):
            print('\n=== Cluster %d ===' % n)
            cluster_indices = np.arange(len(self.tagged_words_list))[self.labels==n]
            sampled_indices = np.random.choice(cluster_indices, n_each_show)
            for index in sampled_indices:
                print(self.tagged_words_list[index])

    def _get_plot_labels(self, label_kind):
        return self.labels

    def plot2d(self, mode, n_sample=None, label_kind='kmeans'):
        dvecs = self.dvecs
        plot_labels = self._get_plot_labels(label_kind)

        if mode == 'PCA':
            dvecs2d = PCA(n_components=2).fit_transform(dvecs)
            self._plot2d(dvecs2d, plot_labels, title=mode, n_sample=n_sample)
        elif mode == 't-SNE':
            if n_sample is not None:
                sampled_indices = np.random.choice(len(dvecs), n_sample)
                dvecs = dvecs[sampled_indices]
                plot_labels = plot_labels[sampled_indices]
            dvecs2d = TSNE(n_components=2, random_state=0).fit_transform(dvecs)
            self._plot2d(dvecs2d, plot_labels, title=mode)

    def _plot2d(self, dvecs2d, labels, title="", n_sample=None):
        df = pd.DataFrame(dvecs2d)
        df.columns = ['x', 'y']
        df['label'] = labels

        if n_sample is not None:
            df = df.sample(n_sample)
        plt.figure(figsize=(10, 10))
        cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)
        palette = sns.color_palette('husl', len(df['label'].unique()))
        ax = sns.scatterplot(x='x', y='y', hue='label', data=df, palette=palette, legend='full')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.01, 1.01))

    def get_tag_list(self, tagged_words_list):
        tag_list = []
        for tagged_words in tagged_words_list:
            tag_list.append(tagged_words.tags[0])
        return tag_list

    def get_random_tags(self, n_choice, random_state=None):
        if random_state is not None:
            np.random.seed(random_state)
        return np.random.choice(self.tag_list, n_choice)


class evaluation_labeled_docs(evaluation_unlabeled_docs):
    def __init__(self, model, tagged_words_list, n_clusters, dic_category_map):
        super().__init__(model, tagged_words_list, n_clusters)
        self.dic_category_map = dic_category_map
        self.category_ary = self.get_cagetory_array()

    def get_cagetory_array(self):
        return np.array([self.dic_category_map[tag] for tag in self.tag_list])

    def _get_plot_labels(self, label_kind):
        if label_kind == 'kmeans':
            return self.labels
        elif label_kind == 'category':
            return self.category_ary

    def _count_km_category(self):
        dic_km_cat_cnt = {}
        unique_km_grs = np.unique(self.labels)
        for km_gr in unique_km_grs:
            km_gr_cats = self.category_ary[self.labels == km_gr]
            dic_km_cat_cnt[km_gr] = Counter(km_gr_cats)
        return dic_km_cat_cnt

    def get_km_pred_category_map(self):
        dic_km_cat_cnt = self._count_km_category()
        dic_km_pred_cat_map = {}
        for km_gr in dic_km_cat_cnt.keys():
            vs = sorted(dic_km_cat_cnt[km_gr].items(), key=lambda x: x[1], reverse=True)
            dic_km_pred_cat_map[km_gr] = vs[0][0]
        return dic_km_pred_cat_map

    def eval_km_cat(self, show=True):
        dic_km_pred_cat_map = self.get_km_pred_category_map()
        pred_category_ary = np.array([dic_km_pred_cat_map[km_label]
                                      for km_label in self.labels])
        category_count = Counter(self.category_ary)
        dic_eval = {}
        for cat, real_n_cat in category_count.items():
            mask_cat = (self.category_ary == cat)
            pred_n_cat = Counter(pred_category_ary[mask_cat])[cat]
            accuracy = float(pred_n_cat) / real_n_cat
            dic_eval[cat] = accuracy

        if show:
            for cat, acc in sorted(dic_eval.items()):
                print(cat, acc)
        return dic_eval

    def eval_sv(self, epochs=1000, show=True, save_path=None, show_each_cat=False):
        batch_size = 128
        min_delta = 0
        patience = 50

        x_train, x_test, y_train, y_test, indices_train, indices_test =\
            self._make_data()
        model = self._make_model(x_train.shape[1], y_train.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                            verbose=0, validation_data=(x_test, y_test), callbacks=[early_stop])
        score = model.evaluate(x_test, y_test, verbose=0)

        if show:
            print('\nMLP score:')
            print('  Test loss:', score[0])
            print('  Test accuracy:', score[1])
            print('  Epoch:', history.epoch[-1])

            fig, ax1 = plt.subplots()
            c1 = 'tab:blue'; c2 = 'tab:red'
            ax1.plot(history.epoch, history.history['val_acc'], color=c1)
            ax1.set_xlabel('epoch'); ax1.set_ylabel('val_acc', color=c1)
            ax2 = ax1.twinx()
            ax2.plot(history.epoch, history.history['val_loss'], color=c2)
            ax2.set_ylabel('val_loss', color=c2)
            plt.show()

        if show_each_cat:
            y_test_cat = np.argmax(y_test, axis=1)
            y_pred = model.predict_classes(x_test)
            print('\n', classification_report(y_test_cat, y_pred, target_names=self.le.classes_))

        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            model.save(os.path.join(save_path, 'model.h5'))
            with open(os.path.join(save_path, 'history.dic'), 'w') as g:
                dic_vals = history.history.copy()
                dic_vals['epoch'] = history.epoch
                g.write(str(dic_vals))
        self.model = model
        return model, score, history

    def _make_data(self, test_ratio=0.2, show=True):
        xs = self.dvecs
        le = preprocessing.LabelEncoder()
        le.fit(self.category_ary)
        ys = le.transform(self.category_ary)
        num_classes = len(le.classes_)

        x_train, x_test, y_train, y_test, indices_train, indices_test =\
            train_test_split(xs, ys, np.arange(ys.size), test_size=test_ratio, random_state=0, stratify=ys)
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        if show:
            print('\nMLP data:')
            print('  ', x_train.shape[0], 'train samples')
            print('  ', x_test.shape[0], 'test samples')
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.indices_train = indices_train
        self.indices_test = indices_test
        self.le = le
        return x_train, x_test, y_train, y_test, indices_train, indices_test

    @staticmethod
    def _make_model(input_size, num_classes):
        l2_val = 0.01
        dropout_val = 0.2
        n_unit = 30

        model = Sequential()
        model.add(Dense(n_unit, activation='relu', input_shape=(input_size,),
                  kernel_regularizer=keras.regularizers.l2(l2_val)))
        model.add(Dropout(dropout_val))
        model.add(Dense(n_unit, activation='relu'))
        model.add(Dropout(dropout_val))
        model.add(Dense(num_classes, activation='softmax'))
        return model

    def make_correct_pred_dict(self):
        pred_cats = self.model.predict_classes(self.x_test)
        ys_int = [list(y).index(1) for y in self.y_test]

        dic_results = defaultdict(dict)
        for pred_cat, real_cat, index in zip(pred_cats, ys_int, self.indices_test):
            tagged_words = self.tagged_words_list[index]
            words = tagged_words[0]
            tag = tagged_words.tags[0]
            dic_results[tag]['words'] = words
            dic_results[tag]['pred_cat'] = self.le.inverse_transform(pred_cat)
            dic_results[tag]['real_cat'] = self.le.inverse_transform(real_cat)
            dic_results[tag]['correct'] = pred_cat == real_cat
        return dic_results
