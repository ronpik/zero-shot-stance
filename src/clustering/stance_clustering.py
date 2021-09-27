import sys
import os
from tqdm.auto import tqdm
from typing import List

import torch
import numpy as np
import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib.pyplot as plt

# from IPython import embed

import pickle

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(SRC_DIR)

from modeling import datasets, data_utils
import modeling.input_models as im

# import datasets, data_utils

# type aliases
Corpus = List[str]

use_cuda = torch.cuda.is_available()
SEED = 4783
DEFAULT_TOPIC_DIR = "../resources/topicreps"


def load_data(train_path: str, id_col="ori_id", text_col="text_s") -> Corpus:
    """
    Reads from train_path the text of each record, and return a list of the corresponding texts.
    :param train_path: path to train data
    :param id_col: the column containing the record ID
    :param text_col: the column containing the text
    :return: list of records' texts
    """
    df = pd.read_csv(train_path)

    seen_ids = set()
    corpus = []
    for i in df.index:
        row = df.iloc[i]
        if row[id_col] not in seen_ids:
            corpus.append(row[text_col])
            seen_ids.add(row[id_col])

    return corpus


def get_features(corpus: Corpus):
    """
    calculates the idf value of words from corpus.
    :param corpus: sequence of texts.
    :return: A mapping between words to their idf value according to corpus.
    """
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    word2idf = dict()
    for w, i in vectorizer.vocabulary_.items():
        word2idf[w] = vectorizer.idf_[i]
    return word2idf


def combine_word_piece_tokens(word_tokens, word2tfidf):
    """

    :param word_tokens:
    :param word2tfidf:
    :return:
    """
    # join the BERT word-piece tokens
    new_word_tokens = []
    for token_list in word_tokens:
        word2pieces = dict()
        i = 0
        new_token_list = []
        while i < len(token_list):
            w = token_list[i]
            if not w.startswith('##'):
                new_token_list.append(w)
            else:
                old_word = new_token_list.pop(-1)
                new_w = old_word + w.strip("##")
                new_token_list.append(new_w)

                word2pieces[new_w] = [old_word, w]
            i += 1
        new_word_tokens.append(new_token_list)

        for w, p_lst in word2pieces.items():
            if w not in word2tfidf:
                continue

            all_pieces = [p_lst[1]]
            wp = p_lst[0]
            while wp in word2pieces:
                all_pieces.append(word2pieces[wp][1])
                wp = word2pieces[wp][0]
            all_pieces.append(wp)

            for wp in all_pieces:
                if wp not in word2tfidf:
                    word2tfidf[wp] = word2tfidf[w]

    return new_word_tokens


def get_tfidf_weights(new_word_tokens, vecs, word2tfidf):
    tfidf_lst = []
    for toklst in new_word_tokens:  # word_tokens:
        temp = []
        for w in toklst:
            temp.append(word2tfidf.get(w, 0.))
        # temp = [word2tfidf[w] for w in toklst if w in word2tfidf]
        while len(temp) < vecs.shape[1]:  # padding to maxlen
            temp.append(0)
        tfidf_lst.append(temp)
    return tfidf_lst


def save_bert_vectors(embed_model, dataloader, batching_fn, batching_kwargs, word2tfidf, topicdir, dataname):
    doc_matrix = []
    topic_matrix = []
    doc2i = dict()
    topic2i = dict()
    didx = 0
    tidx = 0
    for sample_batched in tqdm(dataloader, total=dataloader.n_batches):
        args = batching_fn(sample_batched, **batching_kwargs)
        with torch.no_grad():
            embed_args = embed_model(**args)
            args.update(embed_args)

            vecs = args['txt_E']  # (B, L, 768)
            word_tokens = [dataloader.data.tokenizer.convert_ids_to_tokens(args['text'][i],
                                                                           skip_special_tokens=True)
                           for i in range(args['text'].shape[0])]

            # join the BERT word-piece tokens
            new_word_tokens = combine_word_piece_tokens(word_tokens, word2tfidf)

            tfidf_lst = get_tfidf_weights(new_word_tokens, vecs, word2tfidf)

            tfidf_weights = torch.tensor(tfidf_lst, device=('cuda' if use_cuda else 'cpu'))  # (B, L)
            tfidf_weights = tfidf_weights.unsqueeze(2).repeat(1, 1, vecs.shape[2])
            weighted_vecs = torch.einsum('blh,blh->blh', vecs, tfidf_weights)

            avg_vecs = weighted_vecs.sum(1) / args['txt_l'].unsqueeze(1)

            doc_vecs = avg_vecs.detach().cpu().numpy()
            topic_vecs = args['avg_top_E'].detach().cpu().numpy()

        for bi, b in enumerate(sample_batched):
            if b['ori_text'] not in doc2i:
                doc2i[b['ori_text']] = didx
                didx += 1
                doc_matrix.append(doc_vecs[bi])
            if b['ori_topic'] not in topic2i:
                topic2i[b['ori_topic']] = tidx
                tidx += 1
                topic_matrix.append(topic_vecs[bi])

    docm = np.array(doc_matrix)
    docm_path = os.path.join(topic_dir, f"bert_tfidfW_doc-{dataname}.vecs.npy")
    np.save(docm_path, docm)
    print(f"docm [{dataname}] saved to {docm_path}")
    del docm

    topicm = np.array(topic_matrix)
    topicm_path = os.path.join(topic_dir, f"bert_topic-{dataname}.vecs.npy")
    np.save(topicm_path, topicm)
    print(f"topicm [{dataname}] saved to {topicm_path}")
    del topicm

    doc2i_path = os.path.join(topic_dir, f"bert_tfidfW_doc-{dataname}.vocab.pkl")
    with open(doc2i_path, 'wb') as out_f:
        pickle.dump(doc2i, out_f)
        print(f"doc2i [{dataname}] saved to {doc2i_path}")

    topic2i_path = os.path.join(topic_dir, f"bert_topic-{dataname}.vocab.pkl")
    with open(topic2i_path, 'wb') as out_f:
        pickle.dump(topic2i, out_f)
        print(f"topic2i [{dataname}] saved to {topic2i_path}")


def load_vector_data(p, docname, topicname, dataname, dataloader, mode='concat'):
    docm = np.load(os.path.join(p, f"{docname}-{dataname}.vecs.npy"))
    topicm = np.load(os.path.join(p, f"{topicname}-{dataname}.vecs.npy"))
    doc2i = pickle.load(open(os.path.join(p, f"{docname}-{dataname}.vocab.pkl"), "rb"))
    topic2i = pickle.load(open(os.path.join(p, f"{topicname}-{dataname}.vocab.pkl"), "rb"))

    doc2topics = dict()
    unique_topics = set()

    dataY = dict()
    dataX = []
    idx = -1
    for sample_batched in dataloader:
        for bi, b in enumerate(sample_batched):
            x = b['ori_text']
            t = b['ori_topic']

            if x not in doc2topics:
                doc2topics[x] = set()
            if t not in doc2topics[x]:
                doc2topics[x].add(t)
                docv = docm[doc2i[x]]
                topicv = topicm[topic2i[t]]

                if mode == 'concat':
                    dataX.append(np.concatenate((docv, topicv)))
                elif mode == 'avg':
                    dataX.append(np.mean((docv, topicv), 0))
                else:
                    print("ERROR")
                    sys.exit(1)
                idx += 1
            dataY[b['id']] = idx
    assert len(dataX) - 1 == idx
    return np.array(dataX), dataY


def cluster(train_X, train_Y, dev_X, dev_Y, k, trial_num, topic_dir, dataname, link_type='ward', m='euclidean'):
    print("[{}] clustering with: linkage={}, m={}, n_clusters={}...".format(trial_num, link_type, m, k))
    clustering = AgglomerativeClustering(n_clusters=k, linkage=link_type, affinity=m)
    # clustering = KMeans(n_clusters=k)
    clustering.fit(train_X)
    labels = clustering.labels_
    print('[{}] finished clustering.'.format(trial_num))

    ## labels: new_id -> cluster_number
    train_id2i = dict()
    for rid, eid in train_Y.items():
        train_id2i[rid] = labels[eid]
    train_outpath = os.path.join(topic_dir, f"{dataname}_{link_type}_{m}_{k}-train.labels.pkl")
    with open(train_outpath, 'wb') as out_f:
        pickle.dump(train_id2i, out_f)
        print(f"[{trial_num}] saved to {train_outpath}")

    print(f"[{trial_num}] fitting centroid classifier ...")
    clf = NearestCentroid()
    clf.fit(train_X, labels)
    print(f"[{trial_num}] finished fitting classifier.")

    centroids_outpath = os.path.join(topic_dir, f"{dataname}_{link_type}_{m}_{k}.centroids.npy")
    np.save(centroids_outpath, clf.centroids_)
    print(f"[{trial_num}] saved to {centroids_outpath}")

    dev_labels = clf.predict(dev_X)
    sse = calculate_sse(clf.centroids_, dev_X, dev_labels)
    print(f"[{trial_num}] Sum Squared Error: {sse}")

    dev_id2i = dict()
    for rid, eid in dev_Y.items():
        dev_id2i[rid] = dev_labels[eid]

    dev_outpath = os.path.join(topic_dir, f"{dataname}_{link_type}_{m}_{k}-dev.labels.pkl")
    with open(dev_outpath, 'wb') as out_f:
        pickle.dump(dev_id2i, out_f)
        print(f"[{trial_num}] saved to {dev_outpath}")

    print()
    return sse


def calculate_sse(centroids, dev_X, dev_labels):
    temp = euclidean_distances(dev_X, centroids)
    sse = 0
    for i, l in enumerate(dev_labels):
        sse += temp[i, l]
    return sse


def get_cluster_labels(k, X, Y, topic_dir, embedding_name, data_name):
    train_centroids = np.load(os.path.join(topic_dir, f"{embedding_name}_ward_euclidean_{k}.centroids.npy"))
    classes = np.array([i for i in range(len(train_centroids))])

    clf = NearestCentroid()
    clf.centroids_ = train_centroids
    clf.classes_ = classes

    labels = clf.predict(X)
    id2i = dict()
    for rid, eid in Y.items():
        id2i[rid] = labels[eid]

    outpath = os.path.join(topic_dir, f"{embedding_name}_ward_euclidean_{k}-{data_name}.labels.pkl")
    with open(outpath, "wb") as out_f:
        pickle.dump(id2i, out_f)
        print(f"saved to {outpath}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='What to do', required=True)
    parser.add_argument('-i', '--train_data', help='Name of the training data file', required=False)
    parser.add_argument('-d', '--dev_data', help='Name of the dev data file', required=False)
    parser.add_argument('-e', '--test_data', help='path to the test data file', required=False,
                        default=None)
    parser.add_argument('-a', '--test_name', help='Name of the test data file', required=False,
                        default="test")
    parser.add_argument('-p', '--data_path', default=DEFAULT_TOPIC_DIR, help='Data path to directory for topic reps')
    parser.add_argument('-t', '--topic_name', required=False, default='bert_topic')
    parser.add_argument('-c', '--doc_name', required=False, default='bert_tfidfW_doc')
    parser.add_argument('-k', '--k', type=int, default=197, help="Number of clusters")
    parser.add_argument('-v', '--value_range', required=False, help='Range of values for search')
    parser.add_argument('-n', '--n', help='Num neighbors', required=False)
    parser.add_argument('-f', '--file_name', help='Name for files', required=False,
                        default='bert_tfidfW')
    parser.add_argument('-r', '--num_trials', type=int, help='Number of trials for search')

    args = vars(parser.parse_args())

    topic_dir = args["data_path"]
    data_name = args['file_name']
    test_name = args["test_name"]

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    data = datasets.StanceData(args['train_data'], None, max_token_len=200, max_top_len=5, is_bert=True,
                               add_special_tokens=True)
    dataloader = data_utils.DataSampler(data, batch_size=64, shuffle=False)

    dev_data = datasets.StanceData(args['dev_data'], None, max_token_len=200,
                                   max_top_len=5, is_bert=True, add_special_tokens=True)
    dev_dataloader = data_utils.DataSampler(dev_data, batch_size=64, shuffle=False)

    test_dataloader = None
    if args['test_data'] is not None:
        test_data = datasets.StanceData(args['test_data'], None, max_token_len=200,
                                        max_top_len=5, is_bert=True, add_special_tokens=True)
        test_dataloader = data_utils.DataSampler(test_data, batch_size=64, shuffle=False)

    if args['mode'] == '1':
        print("Saving vectors")

        input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)
        setup_fn = data_utils.setup_helper_bert_ffnn
        batching_fn = data_utils.prepare_batch
        batch_args = {'keep_sen': False}

        print("Load training data")
        corpus = load_data(args['train_data'])

        print("Create tfidf features")
        word2tfidf = get_features(corpus)

        save_bert_vectors(input_layer, dataloader, batching_fn, batch_args, word2tfidf, topic_dir, 'train')
        save_bert_vectors(input_layer, dev_dataloader, batching_fn, batch_args, word2tfidf, topic_dir, 'dev')

        if args['test_data'] is not None:
            save_bert_vectors(input_layer, test_dataloader, batching_fn, batch_args, word2tfidf, topic_dir, test_name)

    elif args['mode'] == '2':
        print("Clustering")
        train_X, train_Y = load_vector_data(topic_dir, docname=args['doc_name'], topicname=args['topic_name'],
                                            dataname='train', dataloader=dataloader, mode='concat')
        dev_X, dev_Y = load_vector_data(topic_dir, docname=args['doc_name'], topicname=args['topic_name'],
                                        dataname='dev', dataloader=dev_dataloader, mode='concat')
        if args['k'] == 0:
            min_v, max_v = args['value_range'].split('-')
            tried_v = set()
            sse_lst = []
            k_lst = []


            def choose_random_k():
                return np.random.randint(int(min_v), int(max_v) + 1)


            for trial_num in range(args['num_trials']):
                while (k := choose_random_k()) in trial_num: pass

                sse = cluster(train_X, train_Y, dev_X, dev_Y, k, trial_num, topic_dir, data_name)
                sse_lst.append(sse)
                k_lst.append(k)

                tried_v.add(k)

            sort_k_indices = np.argsort(k_lst)
            sorted_k = [k_lst[i] for i in sort_k_indices]
            sorted_sse = [sse_lst[i] for i in sort_k_indices]
            plt.plot(sorted_k, sorted_sse, 'go--')
            plt.savefig(os.path.join(topic_dir, f"SSE_clusters_{data_name}.png"))
        else:
            cluster(train_X, train_Y, dev_X, dev_Y, args['k'], 0, topic_dir, data_name)

    elif args['mode'] == '3':
        print("Getting cluster assignments")
        embedding_name = "bert_tfidfW"
        X, Y = load_vector_data(topic_dir, docname=args['doc_name'], topicname=args['topic_name'],
                                dataname='train', dataloader=dataloader, mode='concat')
        get_cluster_labels(args['k'], X, Y, topic_dir, embedding_name, 'train')

        X, Y = load_vector_data(topic_dir, docname=args['doc_name'], topicname=args['topic_name'],
                                dataname='dev', dataloader=dev_dataloader, mode='concat')
        get_cluster_labels(args['k'], X, Y, topic_dir, embedding_name, 'dev')

        X, Y = load_vector_data(topic_dir, docname=args['doc_name'], topicname=args['topic_name'],
                                dataname=test_name, dataloader=test_dataloader, mode='concat')
        get_cluster_labels(args['k'], X, Y, topic_dir, embedding_name, test_name)

    else:
        print("doing nothing.")
