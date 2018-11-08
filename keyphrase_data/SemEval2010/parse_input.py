import os
from gensim.summarization import keywords
from gensim.summarization.keywords import get_graph
from gensim.summarization.pagerank_weighted import build_adjacency_matrix
import numpy as np


class preprocessor:

        def __init__(self):
            pass

        def extract(self, dir_name=None,
                    label_file_name=None,
                    extension='.txt.final', is_train=True):

            feat_dim = 10
            ratio = 0.9
            current_dir = os.getcwd()
            dir_name = current_dir + '/' + dir_name
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

            #print (my_dict_label)
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
                                keyword_list = keywords(text, scores=True, lemmatize=True, ratio=ratio)
                                gra = get_graph(text)
                                token_list = gra.nodes()
                                #TODO self link
                                adj = build_adjacency_matrix(gra).todense()
                                #TODO feature reader

                                print (adj.shape[0], feat_dim)
                                feat = np.random.rand(adj.shape[0], feat_dim)
                                y_all = np.zeros((adj.shape[0], 2))
                                for i, tokens in enumerate(token_list):

                                    if tokens in my_dict_label_splitted[item[0:4]]:
                                        y_all[i][1] = 1
                                    else:
                                        y_all[i][0] = 1
                                y_train, y_val, y_test  = np.zeros(y_all.shape), np.zeros(y_all.shape), np.zeros(y_all.shape)
                                y_train_mask, y_val_mask, y_test_mask = np.zeros(adj.shape[0]), np.zeros(adj.shape[0]), np.zeros(
                                    adj.shape[0])
                                if is_train:
                                    if count%5 == 0:
                                        y_val = y_all
                                        y_val_mask = np.ones(adj.shape[0])
                                    else:
                                        y_train = y_all
                                        y_train_mask = np.ones(adj.shape[0])
                                else:
                                    y_test = y_all
                                    y_test_mask = np.ones(adj.shape[0])

                                lgraphs.append(gra)

                                lgraph_features_adj.append(adj)
                                lgraph_features_feat.append(feat)
                                lgraph_features_y_train.append(y_train)
                                lgraph_features_y_val.append(y_val)
                                lgraph_features_y_test.append(y_test)
                                lgraph_features_y_train_mask.append(y_train_mask)
                                lgraph_features_y_val_mask.append(y_val_mask)
                                lgraph_features_y_test_mask.append(y_test_mask)


                            except Exception as e:
                                print (e)


                    #debug version
                    if count == 5:
                        a = (np.array(lgraph_features_adj),np.array(lgraph_features_feat)
                             , np.array(lgraph_features_y_train), np.array(lgraph_features_y_val),
                             np.array(lgraph_features_y_test), np.array(lgraph_features_y_train_mask),
                             np.array(lgraph_features_y_val_mask), np.array(lgraph_features_y_test_mask))
                        return (lgraphs, a)


            a = (np.array(lgraph_features_adj), np.array(lgraph_features_feat)
                 , np.array(lgraph_features_y_train), np.array(lgraph_features_y_val),
                 np.array(lgraph_features_y_test), np.array(lgraph_features_y_train_mask),
                 np.array(lgraph_features_y_val_mask), np.array(lgraph_features_y_test_mask))
            return (lgraphs, a)


if __name__ == "__main__":
    X = preprocessor()
    lgraphs, lgraph_features = X.extract('train/', 'train/train.combined.stem.final')
    print (lgraphs)
