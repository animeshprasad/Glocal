#!/usr/bin/python
# -*- coding: utf8 -*-
import os
from gensim.summarization import keywords
from .generate_graph import get_graph
from gensim.summarization.pagerank_weighted import build_adjacency_matrix
#from gensim.summarization.keywords import get_graph
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from .generate_graph  import preprocess_string, preprocess_string_nolower, _clean_text_by_word_no_stem
import sys, random
import scipy.sparse as sp

reload(sys)
sys.setdefaultencoding('utf8')

feat_dim = 50

class feature:

        def __init__(self, dir_name=None, extension='.txt.final'):
            all_text = ' '
            if dir_name:
                for item in os.listdir(dir_name):
                    if item.endswith(extension):
                        file_name = os.path.abspath(dir_name + '/' + item)
                        with open(file_name, 'r') as f:
                            try:
                                lines = f.readlines()
                                text = ' '.join([s.strip() for s in lines]) + ' '
                                all_text += text
                            except:
                                pass

            def get_vocab(start_index=1, min_count=2):
                all_words = preprocess_string(all_text)
                vocab = Counter(all_words).most_common()
                vocab_dict = {}
                for items in vocab:
                    if items[1] > min_count:
                        vocab_dict[items[0].decode('utf-8', 'replace')] = len(vocab_dict) + start_index

                print(len(vocab) - len(vocab_dict), ' words are discarded as OOV')
                print(len(vocab_dict), ' words are in vocab')
                return vocab_dict

            def get_subword_features(unique=True):
                #To capture textual features no lowerization
                all_words = preprocess_string_nolower(all_text)
                if unique:
                    all_words  =  list(set(all_words))
                vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=feat_dim)
                vectorizer.fit_transform(all_words)

                return vectorizer

            def get_wordemb_features(unique=True, glove='/Users/nus/Desktop/glove/glove.6B.50d.txt'):
                #To capture textual features no lowerization
                all_words = preprocess_string(all_text)
                if unique:
                    all_words  =  set(all_words)

                with open(glove) as f:
                    content = f.readlines()
                wordvec = {}
                for line in content:
                    splitLine = line.split()
                    word = splitLine[0]
                    embedding = np.array([float(val) for val in splitLine[1:]])
                    wordvec[word] = embedding

                print (len(all_words), len(all_words.intersection(set(wordvec.keys()))))

                return wordvec



            self.vocab_dict = get_vocab()
            self.word_embeddings = get_wordemb_features()
            self.subword_embeddings_lookup = get_subword_features()



class preprocessor:

        def __init__(self):
            pass

        def extract(self, dir_name=None,
                    label_file_name=None,
                    extension='.txt.final', is_train=True):

            ratio = 1.0
            nb_nodes = 1200
            current_dir = os.getcwd()
            dir_name = current_dir + '/' + dir_name

            feature_extractor = feature(dir_name)

            my_dict_label = {}
            my_dict_label_splitted = {}

            lgraphs = []

            lgraph_features_adj = []
            lgraph_features_feat = []
            lgraph_features_y_train = []
            lgraph_features_y_val = []
            lgraph_features_y_test = []
            lgraph_features_y_train_mask = []
            lgraph_features_y_val_mask = []
            lgraph_features_y_test_mask = []

            with open(current_dir + '/'+ label_file_name, 'r') as lf:
                all_labels = lf.readlines()
                for items in all_labels:
                    my_dict_label[items.split(':')[0].strip()] = items.split(':')[1].strip().split(',')
                    my_dict_label_splitted[items.split(':')[0].strip()] = list(set(' '.join(items.split(':')[1].strip().split(',')).split(' ')))
                    tk = _clean_text_by_word_no_stem(items.split(':')[1])
                    my_dict_label_splitted[items.split(':')[0].strip()] = [tk[units].token for units in tk]

                    #print (my_dict_label_splitted[items.split(':')[0].strip()])

            #print (my_dict_label_splitted)

            count = 0
            for item in os.listdir(dir_name):
                if item.endswith(extension):
                    count += 1
                    file_name = os.path.abspath(dir_name + '/' +item)
                    #print (file_name)
                    with open(file_name,'r') as f:
                            try:
                                lines = f.readlines()
                                text = ' '.join([s.strip() for s in lines])

                                gra = get_graph(text)
                                token_list = gra.nodes()

                                if token_list > nb_nodes:
                                    ratio = 0.7
                                else:
                                    ratio = 0.99 #sans unreachable nodes

                                keyword = keywords(text, scores=True, lemmatize=False, ratio=ratio)
                                keyword_list = [k[0] for k in keyword]
                                keyword_score = [k[1] for k in keyword]

                                keyword_list_full = []
                                keyword_score_full = []
                                for kindex, keyp in enumerate(keyword_list):
                                    if ' ' in keyp:
                                        a = keyp.split()
                                        for k in a:
                                            keyword_list_full.append(k.strip())
                                            keyword_score_full.append(keyword_score[kindex])
                                    else:
                                        keyword_list_full.append(keyp)
                                        keyword_score_full.append(keyword_score[kindex])



                                print ('before', len(token_list))
                                for token in token_list:
                                    if token not in keyword_list_full:
                                        gra.del_node(token)
                                token_list = gra.nodes()
                                print('after', len(token_list))
                                adj = build_adjacency_matrix(gra).todense()
                                np.fill_diagonal(adj, 1)
                                #print (adj[0])

                                rowsum = np.array(adj.sum(1))
                                d_inv_sqrt = np.power(rowsum, -0.5).flatten()
                                d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
                                d_mat_inv_sqrt = np.diag(d_inv_sqrt)
                                adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
                                #print(adj[0])

                                adj = np.dot(np.diag(keyword_score_full), adj)
                                #print(adj[0])



                                #feat = np.array(feature_extractor.subword_embeddings_lookup.transform(token_list).todense())
                                feat = []
                                for node_text in token_list:
                                    node_text = node_text.lower()
                                    try:
                                    #if node_text in feature_extractor.word_embeddings.keys():
                                        feat.append(np.array(feature_extractor.word_embeddings[node_text]))
                                    except:
                                    #else:
                                        feat.append(np.random.uniform(size=50))
                                feat=np.array(feat)

                                #feat = np.random.rand(adj.shape[0], feat_dim)

                                a=0
                                y_all = np.zeros((adj.shape[0], 2))
                                for i, tokens in enumerate(token_list):
                                    if tokens in my_dict_label_splitted[item[0:4]]:
                                        y_all[i][1] = 1
                                        a+=1
                                    else:
                                        y_all[i][0] = 1
                                        # if random.randint(1, 100) < 20:
                                        #     y_all[i][0] = 0
                                        #     y_all[i][1] = 1

                                print ('keywords found to all ratio: ', a, len(my_dict_label_splitted[item[0:4]]))

                                y_train, y_val, y_test  = np.zeros(y_all.shape), np.zeros(y_all.shape), np.zeros(y_all.shape)
                                y_train_mask, y_val_mask, y_test_mask = np.zeros(adj.shape[0]), np.zeros(adj.shape[0]), np.zeros(
                                    adj.shape[0])
                                if is_train:
                                    if count % 10 == 0:
                                        y_val = y_all
                                        y_val_mask = np.ones(adj.shape[0])
                                    else:
                                        y_train = y_all
                                        y_train_mask = np.ones(adj.shape[0])
                                else:
                                    #TODO
                                    y_test = y_all
                                    y_test_mask = np.ones(adj.shape[0])

                                print ('Graph ' + str(count) + ' parsed with shape ' +  str(feat.shape))
                                lgraphs.append(gra)


                                adj_padded  = np.zeros((nb_nodes, nb_nodes))
                                adj_padded[:adj.shape[0],:adj.shape[1]] = adj

                                feat_padded  = np.zeros((nb_nodes, feat_dim))
                                feat_padded[:feat.shape[0],:] = feat

                                def get_pad(y):
                                    a = np.zeros((nb_nodes, 2))
                                    a[:y.shape[0], :]= y
                                    return a

                                def get_pad_mask(y):
                                    a = np.zeros((nb_nodes))
                                    a[:y.shape[0]]= y
                                    return a

                                y_train_padded = get_pad(y_train)
                                y_val_padded = get_pad(y_val)
                                y_test_padded = get_pad(y_test)
                                y_train_mask_padded = get_pad_mask(y_train_mask)
                                y_val_mask_padded = get_pad_mask(y_val_mask)
                                y_test_mask_padded = get_pad_mask(y_test_mask)

                                lgraph_features_adj.append(adj_padded)
                                lgraph_features_feat.append(feat_padded)
                                lgraph_features_y_train.append(y_train_padded)
                                lgraph_features_y_val.append(y_val_padded)
                                lgraph_features_y_test.append(y_test_padded)
                                lgraph_features_y_train_mask.append(y_train_mask_padded)
                                lgraph_features_y_val_mask.append(y_val_mask_padded)
                                lgraph_features_y_test_mask.append(y_test_mask_padded)

                            except Exception as e:
                                print (e)

                    #debug version
                    # if count == 6:
                    #     a = (np.array(lgraph_features_adj),np.array(lgraph_features_feat)
                    #          , np.array(lgraph_features_y_train), np.array(lgraph_features_y_val),
                    #          np.array(lgraph_features_y_test), np.array(lgraph_features_y_train_mask),
                    #          np.array(lgraph_features_y_val_mask), np.array(lgraph_features_y_test_mask))
                    #     return (lgraphs, a)


            a = (np.array(lgraph_features_adj), np.array(lgraph_features_feat)
                 , np.array(lgraph_features_y_train), np.array(lgraph_features_y_val),
                 np.array(lgraph_features_y_test), np.array(lgraph_features_y_train_mask),
                 np.array(lgraph_features_y_val_mask), np.array(lgraph_features_y_test_mask))
            return (lgraphs, a)


if __name__ == "__main__":
    X = preprocessor()
    lgraphs, lgraph_features = X.extract('train/', 'train/train.combined.final')
    print (lgraphs)
