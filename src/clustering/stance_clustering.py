import csv

from functools import partial

import sys
import os
import traceback
from tqdm.auto import tqdm
from typing import List, Callable, Dict, Any, Tuple, Sequence, Protocol, Union
from multiprocessing import Pool

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

from modeling import data_utils
from modeling.data_utils import DataSampler
from modeling.datasets import StanceData
import modeling.input_models as im

SEED = 4783
DEFAULT_TOPIC_DIR = "../resources/topicreps"
EMBEDDING_NAME = "bert_tfidfW"
DOC_REPR_PREFIX = "bert_tfidfW_doc"
TOPIC_REPR_PREFIX = "bert_topic"

# type aliases
Corpus = List[str]
NamedDataset = Tuple[str, str]
NamedDataLoader = Tuple[str, DataSampler]

# from typing_extensions import Protocol  # for Python <3.8

class ClusterPredictor(Protocol):
    def predict(self, X: Sequence) -> Sequence[int]: ...

use_cuda = torch.cuda.is_available()
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True


def get_data_sampler(path: str, vocab_name: str = None) -> DataSampler:
    data = StanceData(path, vocab_name, max_token_len=200,
                      max_top_len=5, is_bert=True, add_special_tokens=True)

    return DataSampler(data, batch_size=64, shuffle=False)


def parse_datasets(datasets_str: str) -> List[Tuple[str, DataSampler]]:
    """
    Get the datasets argument and parse it to pairs of name and path to the given datasets.
    The 'datasets_str' argument consists of pairs of name and path to datasets separated with comman between pairs,
    where each pair is separated with a colon between the name and the path to the specific dataset.

    'datasets_str' is of the form: name1:path1,name2:path2,...
    :param datasets_str:
    :return: a listy of tuple pairs with name as the first argument and the path is the second
    """
    datasets = []
    for dataset in datasets_str.split(","):
        parts = dataset.split(":")
        if len(parts) != 2:
            print(f"Could not parse the pair \"{dataset}\"")
            print("Skipping")
            continue

        name, path = parts
        print(f"Dataset - {name}: {path}")
        try:
            sampler = get_data_sampler(path)
            datasets.append((name, sampler))
        except Exception as e:
            print(traceback.format_exc())
            print(f"Skip dataset \"{name}\" in path: \"{path}\"")

    print(f"Parsed total of {len(datasets)} datasets")
    return datasets


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


def get_features(corpus: Corpus) -> Dict[str, float]:
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


def store_idf_values(word2idf: Dict[str, float], path: str):
    with open(path, 'w') as idf_f:
        writer = csv.writer(idf_f, lineterminator='\n')
        writer.writerow("word,idf".split(','))
        writer.writerows(word2idf.items())


def load_idf_values(path: str) -> Dict[str, float]:
    with open(path, 'r') as idf_f:
        reader = csv.reader(idf_f)
        next(reader)  # skip header
        return dict(((word, float(idf)) for word, idf in reader))


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


def get_tfidf_weights(
        new_word_tokens: Sequence[Sequence[str]],
        embeddings: np.ndarray,
        word2idf: Dict[str, float]
) -> List[List[float]]:
    tfidf_list = []
    for tokens in new_word_tokens:  # word_tokens:
        idf_values = []
        for w in tokens:
            idf_values.append(word2idf.get(w, 0.))

        # padding to maxlen
        while len(idf_values) < embeddings.shape[1]:
            idf_values.append(0.)
        tfidf_list.append(idf_values)
    return tfidf_list


def save_bert_vectors(
        embed_model: im.BERTLayer,
        dataloader: DataSampler,
        batching_fn: Callable[[List[dict]], Dict[str, Any]],
        word2tfidf: Dict[str, float],
        topic_dir: str,
        dataname: str
):
    doc_matrix = []
    topic_matrix = []
    doc2i = dict()
    topic2i = dict()
    didx = 0
    tidx = 0
    for sample_batched in tqdm(dataloader, total=dataloader.n_batches):
        args = batching_fn(sample_batched)
        with torch.no_grad():
            embed_args = embed_model(**args)
            args.update(embed_args)

            embeddings = args["txt_E"]
            word_tokens = [dataloader.data.tokenizer.convert_ids_to_tokens(args["text"][i],
                                                                           skip_special_tokens=True)
                           for i in range(args["text"].shape[0])]

            # join the BERT word-piece tokens
            new_word_tokens = combine_word_piece_tokens(word_tokens, word2tfidf)

            tfidf_list = get_tfidf_weights(new_word_tokens, embeddings, word2tfidf)

            tfidf_weights = torch.tensor(tfidf_list, device=('cuda' if use_cuda else 'cpu'))  # (B, L)
            tfidf_weights = tfidf_weights.unsqueeze(2).repeat(1, 1, embeddings.shape[2])
            weighted_vecs = torch.einsum('blh,blh->blh', embeddings, tfidf_weights)

            avg_vecs = weighted_vecs.sum(1) / args["txt_l"].unsqueeze(1)

            doc_vecs = avg_vecs.detach().cpu().numpy()
            topic_vecs = args["avg_top_E"].detach().cpu().numpy()

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


def load_vector_data(topic_dir: str, dataname: str, dataloader: DataSampler, mode: str = 'concat'):
    docm = np.load(os.path.join(topic_dir, f"{DOC_REPR_PREFIX}-{dataname}.vecs.npy"))
    topicm = np.load(os.path.join(topic_dir, f"{TOPIC_REPR_PREFIX}-{dataname}.vecs.npy"))
    doc2i = pickle.load(open(os.path.join(topic_dir, f"{DOC_REPR_PREFIX}-{dataname}.vocab.pkl"), "rb"))
    topic2i = pickle.load(open(os.path.join(topic_dir, f"{TOPIC_REPR_PREFIX}-{dataname}.vocab.pkl"), "rb"))

    doc2topics = dict()
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


def create_and_save_topic_vectors(
        datasets: List[Tuple[str, DataSampler]],
        word2idf: Dict[str, float],
        topic_dir: str,
        use_cuda: bool
):
    print("Saving vectors")
    input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)
    batch_args = {'keep_sen': False}
    batching_fn = partial(data_utils.prepare_batch, **batch_args)

    for dataset_name, dataloader in datasets:
        print(f"Get embeddings for dataset {dataset_name} and save results")
        save_bert_vectors(input_layer, dataloader, batching_fn, word2idf, topic_dir, dataset_name)


def cluster(train_X, k, trial_num, link_type="ward", m="euclidean") -> NearestCentroid:
    print(f"[{trial_num}] clustering with: linkage={link_type}, m={m}, n_clusters={k}...")
    clustering = AgglomerativeClustering(n_clusters=k, linkage=link_type, affinity=m)
    clustering.fit(train_X)
    labels = clustering.labels_
    print(f"[{trial_num}] finished clustering.")

    print(f"[{trial_num}] fitting centroid classifier ...")
    clf = NearestCentroid()
    clf.fit(train_X, labels)
    print(f"[{trial_num}] finished fitting classifier.")
    return clf


def store_cluster_predictor(clf: NearestCentroid, topic_dir: str, k: int, link_type="ward", m="euclidean"):
    centroids_outpath = os.path.join(topic_dir, f"{EMBEDDING_NAME}_ward_euclidean_{k}.centroids.npy")
    np.save(centroids_outpath, clf.centroids_)
    print(f"saved centroids to {centroids_outpath}")


def calculate_sse(centroids, dev_X, dev_labels):
    distances = euclidean_distances(dev_X, centroids)
    sse = 0
    for i, l in enumerate(dev_labels):
        sse += distances[i, l]

    return sse


def measure_sse(clf: NearestCentroid, X):
    labels = clf.predict(X)
    sse = calculate_sse(clf.centroids_, X, labels)
    return sse


def get_cluster_labels(clf: ClusterPredictor, X: np.ndarray, Y: dict) -> Dict[Any, int]:
    labels = clf.predict(X)
    id2cluster = dict()
    for rid, eid in Y.items():
        id2cluster[rid] = labels[eid]

    return id2cluster


def store_clustered(cluster_mapping: Dict[Any, int], topic_dir: str, k:int, data_name):
    outpath = os.path.join(topic_dir, f"{EMBEDDING_NAME}_ward_euclidean_{k}-{data_name}.labels.pkl")
    with open(outpath, "wb") as out_f:
        pickle.dump(cluster_mapping, out_f)
        print(f"saved to {outpath}")


def cluster_and_measure(k: int, train_X: np.ndarray, dev_X: np.ndarray) -> float:
    clf = cluster(train_X, k, k)
    return measure_sse(clf, dev_X)


def find_best_k(train_X: np.ndarray, dev_X: np.ndarray, num_trials: int, min_value: int, max_value: int, n_jobs: int = None) -> Tuple[int, float]:
    k_values = np.random.randint(min_value, max_value + 1, num_trials)

    scorer = partial(cluster_and_measure, train_X=train_X, dev_X=dev_X)
    pool = Pool(n_jobs)
    sse_scores = pool.map(scorer, k_values, chunksize=1)

    # plot results
    sort_k_indices = np.argsort(k_values)
    sorted_k = [k_values[i] for i in sort_k_indices]
    sorted_sse = [sse_scores[i] for i in sort_k_indices]
    plt.plot(sorted_k, sorted_sse, 'go--')
    plt.savefig(os.path.join(topic_dir, f"SSE_clusters.png"))

    min_index = np.argmin(sse_scores)
    return k_values[min_index], sse_scores[min_index]


def detect_k(topic_dir: str) -> Union[int, None]:
    prefix = f"{EMBEDDING_NAME}_ward_euclidean_"
    for filename in os.listdir(topic_dir):
        if filename.startswith(prefix):
            k = filename[len(prefix):].split(".")[0].split("-")[0]
            return int(k)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=int, default=1, help='What to do', required=True)
    parser.add_argument('-d', '--datasets', type=str, help='datasets to process their topics', required=True)
    parser.add_argument('-p', '--data_path', default=DEFAULT_TOPIC_DIR, help='Data path to directory for topic reps')

    # training arguments
    parser.add_argument("--train", action="store_const", const=True, default=False,
                        help='Flag to use the first dataset for training, that is - creating')
    parser.add_argument('-k', '--k', type=int, default=None, help="Number of clusters to create")
    parser.add_argument('-v', '--value_range', default="0-500", help='Range of values for search')
    parser.add_argument('-r', '--num_trials', type=int, default=10, help='Number of trials for search')
    parser.add_argument('-j', '--num_jobs', type=int, default=os.cpu_count(), help='Number of cores to parallel where possible')

    args = parser.parse_args()

    print("Processing all datasets")
    named_samplers = parse_datasets(args.datasets)
    topic_dir = args.data_path
    if not os.path.isdir(topic_dir):
        os.makedirs(topic_dir)

    num_clusters = args.k
    if num_clusters is None:
        num_clusters = detect_k(topic_dir)
    if num_clusters is None:
        num_clusters = 0

    train_name, train_data_sampler, dev_name, dev_data_sampler = None, None, None, None

    if args.train:
        train_name, train_data_sampler = named_samplers[0]
        dev_name, dev_data_sampler = named_samplers[1]

    if args.mode <= 1:
        print("Compute BERT-based representation of topics and documents")
        idf_path = os.path.join(topic_dir, "idf.csv")
        if train_data_sampler is not None:
            print("Load training data")
            train_path = train_data_sampler.data.data_name
            corpus = load_data(train_path)
            print("Create tfidf features")
            word2idf = get_features(corpus)
            print(f"Store words' idf values in {idf_path}")
            store_idf_values(word2idf, idf_path)

        word2idf = load_idf_values(idf_path)
        create_and_save_topic_vectors(named_samplers, word2idf, topic_dir, use_cuda=use_cuda)

    if args.mode <= 2:
        if num_clusters == 0:
            print("Perform clustering")
            print("Searching for the best 'k' value")
            train_X, _ = load_vector_data(topic_dir, dataname=train_name, dataloader=train_data_sampler, mode='concat')
            dev_X, _ = load_vector_data(topic_dir, dataname=dev_name, dataloader=dev_data_sampler, mode='concat')
            min_value, max_value = tuple(map(int, args.value_range.split('-')))
            num_clusters, score = find_best_k(train_X, dev_X, args.num_trials, min_value, max_value, n_jobs=args.num_jobs)
            print(f"Done! Best number of clusters: {num_clusters} with SSE score of {score}")
            clf = cluster(train_X, num_clusters, 0)
            store_cluster_predictor(clf, topic_dir, num_clusters)
        else:
            print(f"Number of clustered was already determined: {num_clusters}")

    if args.mode <= 3:
        print("Getting cluster assignments")
        clf = NearestCentroid()
        clf.centroids_ = np.load(os.path.join(topic_dir, f"{EMBEDDING_NAME}_ward_euclidean_{num_clusters}.centroids.npy"))
        clf.classes_ = np.arange(len(clf.centroids_))

        for data_name, data_sampler in named_samplers:
            X, Y = load_vector_data(topic_dir, dataname=data_name, dataloader=data_sampler, mode='concat')
            clusters = get_cluster_labels(clf, X, Y)
            store_clustered(clusters, topic_dir, num_clusters, data_name)

    else:
        print(f"doing nothing: mode={args.mode}")
