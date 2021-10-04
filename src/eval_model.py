import traceback

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import List, Tuple, Iterable, Sequence, Dict, Callable, Union

import numpy as np
import torch, os, sys, argparse

from modeling.data_utils import DataSampler
from modeling.datasets import StanceData

sys.path.append('./modeling')
from modeling.model_utils import TorchModelHandler
from modeling import data_utils, datasets, input_models as im, models as bm
import torch.nn as nn
import torch.optim as optim
import pandas as pd

DEFAULT_TOPIC_DIR = "../resources/topicreps"

VECTOR_NAME = 'glove.6B.100d'
SEED = 0
NUM_GPUS = None
use_cuda = torch.cuda.is_available()

# alias
Prediction = Tuple[Sequence[int], Sequence[int], Dict[str, int], np.ndarray]
PredictionsPerTopic = Dict[str, Tuple[Sequence[int], Sequence[int]]]


def analyze_predictions(
        topics: Sequence[str], y_true: Sequence[int], y_pred: Sequence[int]
) -> PredictionsPerTopic:

    predictions_per_topic = {}
    for topic, label, pred in zip(topics, y_true, y_pred):
        preds, labels = predictions_per_topic.setdefault(topic, ([], []))
        preds.append(pred)
        labels.append(label)

    return predictions_per_topic


def compute_scores(score_fn: Callable, true_labels: Sequence[int], pred_labels: Sequence[int], class_wise: bool,
                   prefix: str, label_names: Sequence[str]) -> Dict[str, float]:
    """
    Computes scores using the given scoring function of the given name. The scores
    are stored in the internal score dictionary.
    :param score_fn: the scoring function to use.
    :param true_labels: the true labels.
    :param pred_labels: the predicted labels.
    :param class_wise: flag to determine whether to compute class-wise scores in
                        addition to macro-averaged scores.
    :param prefix: the name of this score function, to be used in storing the scores.
    :param num_labels:
    """
    num_labels = len(label_names)
    labels = [i for i in range(num_labels)]
    label_scores = score_fn(true_labels, pred_labels, labels=labels, average=None)
    scores = {f"{prefix}_macro": label_scores.mean()}
    if class_wise:
        for name, score in zip(label_names, label_scores):
            scores[f"{prefix}_{name}"] = score

    return scores


def score(pred_labels: Sequence[int], true_labels: Sequence[int], t2pred: PredictionsPerTopic, marks: Sequence[int],
          class_wise: bool = True, topic_wise: bool = False) -> Dict[str, float]:
    """
    Helper Function to compute scores. Stores updated scores in
    the field "score_dict".
    :param pred_labels: the predicted labels
    :param true_labels: the correct labels
    :param t2pred:
    :param marks:
    :param class_wise: flag to determine whether to compute class-wise scores in
                        addition to macro-averaged scores.
    :param topic_wise:
    """
    labels = ["anti", "pro", "none"]
    scores = {"accuracy": accuracy_score(true_labels, pred_labels)}
    scores.update(compute_scores(f1_score, true_labels, pred_labels, class_wise, 'f', labels))
    scores.update(compute_scores(precision_score, true_labels, pred_labels, class_wise, 'p', labels))
    scores.update(compute_scores(recall_score, true_labels, pred_labels, class_wise, 'r', labels))

    for v in [1, 0]:
        tl_lst = []
        pl_lst = []
        for m, tl, pl in zip(marks, true_labels, pred_labels):
            if m != v: continue
            tl_lst.append(tl)
            pl_lst.append(pl)

        scores.update(compute_scores(f1_score, tl_lst, pl_lst, class_wise, f"f-{v}", labels))
        scores.update(compute_scores(precision_score, tl_lst, pl_lst, class_wise, f"p-{v}", labels))
        scores.update(compute_scores(recall_score, tl_lst, pl_lst, class_wise, f"r-{v}", labels))

    if topic_wise:
        for t in t2pred:
            preds, trues = t2pred[t]
            scores.update(compute_scores(f1_score, trues, preds, class_wise, f"{t}-f", labels))

    return scores


def eval_predicted_dataset(data: pd.DataFrame, predictions: np.ndarray,
                           class_wise: bool = False, topic_wise: bool = False) -> Dict[str, float]:
    """
    Evaluates this model on the given data. Stores computed
    scores in the field "score_dict". Currently computes macro-averaged
    F1 scores, precision and recall. Can also compute scores on a class-wise basis.
    :param data: the data to use for evaluation. By default uses the internally stored data
                (should be a DataSampler if passed as a parameter).
    :param predictions:
    :param class_wise: flag to determine whether to compute class-wise scores in
                        addition to macro-averaged scores.
    :param topic_wise:
    :return: a map from score names to values
    """
    topics = data["ori_topic"]
    true_labels = data["label"]
    marks = data["seen?"]
    predictions_per_topic = analyze_predictions(topics, true_labels, predictions)
    scores = score(predictions, true_labels, predictions_per_topic, marks, class_wise, topic_wise)
    return scores


def eval_and_print(data_name: str, data: pd.DataFrame, predictions: np.ndarray,
                   class_wise: bool = False, topic_wise: bool = False) -> Dict[str, float]:
    '''
    Evaluates this model on the given data. Stores computed
    scores in the field "score_dict". Currently computes macro-averaged.
    Prints the results to the console.
    F1 scores, precision and recall. Can also compute scores on a class-wise basis.
    :param data_name: the name of the data evaluating.
    :param data: the data to use for evaluation. By default uses the internally stored data
                (should be a DataSampler if passed as a parameter).
    :param predictions:
    :param class_wise: flag to determine whether to compute class-wise scores in
                        addition to macro-averaged scores.
    :param topic_wise:
    :return: a map from score names to values
    '''
    scores = eval_predicted_dataset(data, predictions, class_wise, topic_wise)
    print("Evaluation on \"{}\" data".format(data_name))
    for s_name, s_val in scores.items():
        print("{}: {}".format(s_name, s_val))
    return scores


def eval(predicted_datasets: Iterable[Tuple[str, pd.DataFrame, np.ndarray]],
         class_wise: bool = False, topic_wise: bool = False):
    """
    Evaluates the given model on the given data, by computing
    macro-averaged F1, precision, and recall scores. Can also
    compute class-wise scores. Prints the resulting scores
    :param class_wise: whether to return class-wise scores. Default(False):
                        does not return class-wise scores.
    :return: a dictionary from score names to the score values.
    """
    for data_name, data, predictions in predicted_datasets:
        eval_and_print(data_name, data, predictions, class_wise=class_wise, topic_wise=topic_wise)


def merge_predictions_to_dataset(predictions: Sequence[int], dataset: StanceData):
    out_data = []
    cols = list(dataset.data_file.columns)
    for i in dataset.data_file.index:
        row = dataset.data_file.iloc[i]
        temp = [row[c] for c in cols]
        temp.append(predictions[i])
        out_data.append(temp)
    cols += ["pred"]
    return pd.DataFrame(out_data, columns=cols)


def save_predicted(data_sampler: DataSampler, predictions: Sequence[int], outpath: str):
    merge_predictions_to_dataset(predictions, data_sampler.data).to_csv(outpath, index=False)
    print(f"Predicted dataset saved to {outpath}")


def parse_dataset_arguments(datasets_str: Union[str, None]) -> Iterable[Tuple[str, str]]:
    if datasets_str is None:
        return []

    for dataset in datasets_str.split(","):
        parts = dataset.split(":")
        if len(parts) != 2:
            print(f"Could not parse the pair \"{dataset}\"")
            print("Skipping")
            continue

        name, path = parts
        yield name, path


def load_word_embeddings(config: dict) -> Tuple[str, np.ndarray]:
    """
    load the word embeddings specified in the 'config' dict, and returns the embeddings' name with the loaded vectors
    :param config:
    :return:
    """
    embeddings_name = config["vec_name"]
    embedding_dim = int(config["vec_dim"])
    embeddings_path = f"../resources/{embeddings_name}.vectors.npy"
    return embeddings_name, data_utils.load_vectors(embeddings_path, dim=embedding_dim, seed=SEED)


def get_samplers(named_datasets: Iterable[Tuple[str, str, dict]], config: dict) -> List[Tuple[str, DataSampler]]:
    """
    Get the datasets argument and parse it to pairs of name and path to the given datasets.
    The 'datasets_str' argument consists of pairs of name and path to datasets separated with comman between pairs,
    where each pair is separated with a colon between the name and the path to the specific dataset.

    'datasets_str' is of the form: name1:path1,name2:path2,...
    :param named_datasets: pairs of name and path of datasets to load.
    :param config: model configuration
    :return: a list of tuple pairs with name as the first argument and the path is the second
    """
    # initialize data parameters by using bert
    max_token_length = int(config.get("max_tok_len", 200))
    max_sentence_length = 10
    max_top_length = config.get("max_top_len")
    keep_sentence = False
    add_special_tokens = not bool(int(config.get("together_in", "0")))
    padding = 0
    vocab_path = None
    use_bert = ("bert" in config) or ("bert" in config["name"])
    if not use_bert:
        max_top_length = 5
        max_sentence_length = config.get("max_sen_len") or max_sentence_length
        add_special_tokens = True
        keep_sentence = "keep_sen" in config

        embeddings_name, embeddings = load_word_embeddings(config)
        padding = embeddings.shape[0] - 1
        vocab_path = f"../resources/{embeddings_name}.vocab.pkl"

    samplers = []
    for name, path, dataset_kwargs in named_datasets:
        print(f"Dataset - {name}: {path}")
        try:
            data = StanceData(path, vocab_path, pad_val=padding, max_tok_len=max_token_length,
                              max_sen_len=max_sentence_length, max_top_len=max_top_length,
                              add_special_tokens=add_special_tokens, keep_sen=keep_sentence, is_bert=use_bert,
                              **dataset_kwargs)

            sampler = DataSampler(data, batch_size=batch_size, shuffle=False)
            samplers.append((name, sampler))

        except Exception as e:
            print(traceback.format_exc())
            print(f"Skip dataset \"{name}\" in path: \"{path}\"")

    print(f"Parsed total of {len(samplers)} datasets")
    return samplers


def load_predicted(path: str) -> Tuple[pd.DataFrame, Sequence[int]]:
    data = pd.read_csv(path)
    return data, data["pred"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--config_file', help='Name of the cofig data file', required=False)
    parser.add_argument('-p', '--topic_dir', type=str, help='Data path to directory for topic reps')
    parser.add_argument('-k', '--ckp_name', default="BEST", help='Checkpoint suffix name')
    parser.add_argument('-d', '--datasets', default=None, help='Path to the dev data file')
    parser.add_argument('-r', '--preds', default=None, help='predictions to evaluate')
    parser.add_argument('-b', '--batch-size', type=int, default=64,
                        help="batch size - use larger values for faster processing, or smaller values if memory is insufficient")
    parser.add_argument('-o', '--outdir', help='Ouput file name', default='')

    args = parser.parse_args()
    batch_size = args.batch_size
    named_datapaths = parse_dataset_arguments(args.datasets)
    named_predicted = parse_dataset_arguments(args.preds)
    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    ####################
    # load config file #
    ####################
    with open(args.config_file, 'r') as f:
        config = dict()
        for l in f.readlines():
            config[l.strip().split(":")[0]] = l.strip().split(":")[1]

    topic_dir = args.topic_dir or config["topic_path"]
    topic_name = config.get("topic_name")
    topic_embeddings = None
    if "topic_name" in config:
        reps = config.get("rep_v", "centroids")
        topic_embeddings = np.load(f"{topic_dir}/{topic_name}.{reps}.npy")

    named_datasets = []
    for data_name, data_path in named_datapaths:
        data_kwargs = {}
        if "topic_name" in config:
            data_kwargs["topic_rep_dict"] = f"{topic_dir}/{topic_name}-{data_name}.labels.pkl"

        named_datasets.append((data_name, data_path, data_kwargs))

    #############
    # LOAD DATA #
    #############
    # load training data
    data_samplers = get_samplers(named_datasets, config)

    lr = float(config.get('lr', '0.001'))
    batch_args = {}
    model_kwargs = {
        'dataloader': None,
        'batching_fn': data_utils.prepare_batch,
        'batching_kwargs': batch_args,
        'name': config['name'],
        "loss_function": nn.CrossEntropyLoss(),
    }

    if 'tganet' in config['name']:
        batch_args["keep_sen"] = False
        input_layer = im.JointBERTLayerWithExtra(vecs=topic_embeddings, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

        model = bm.TGANet(in_dropout_prob=float(config['in_dropout']),
                          hidden_size=int(config['hidden_size']),
                          text_dim=int(config['text_dim']),
                          add_topic=(config.get('add_resid', '0') == '1'),
                          att_mode=config.get('att_mode', 'text_only'),
                          topic_dim=int(config['topic_dim']),
                          learned=(config.get('learned', '0') == '1'),
                          use_cuda=use_cuda)

        model_kwargs["embed_model"] = input_layer
        model_kwargs["model"] = model
        model_kwargs["optimizer"] = optim.Adam(model.parameters())
        model_kwargs["setup_fn"] = data_utils.setup_helper_bert_attffnn

    elif 'ffnn-bert' in config['name']:
        batch_args["keep_sen"] = False
        if config.get("together_in", '0') == '1':

            if 'topic_name' in config:
                input_layer = im.JointBERTLayerWithExtra(vecs=topic_embeddings, use_cuda=use_cuda,
                                                         use_both=(config.get('use_ori_topic', '1') == '1'),
                                                         static_vecs=(config.get('static_topics', '1') == '1'))
            else:
                input_layer = im.JointBERTLayer(use_cuda=use_cuda)

        else:
            input_layer = im.BERTLayer(mode='text-level', use_cuda=use_cuda)


        model = bm.FFNN(input_dim=input_layer.dim, in_dropout_prob=float(config['in_dropout']),
                        hidden_size=int(config['hidden_size']), bias=False,
                        add_topic=(config.get('add_resid', '1') == '1'), use_cuda=use_cuda)

        model_kwargs["embed_model"] = input_layer
        model_kwargs["model"] = model
        model_kwargs["optimizer"] = optim.Adam(model.parameters())
        model_kwargs["setup_fn"] = data_utils.setup_helper_bert_ffnn
        model_kwargs["fine_tune"] = config.get("fine-tune", "no") == "yes"

    elif 'BiCond' in config['name']:
        input_layer = im.BasicWordEmbedLayer(vecs=data_samplers[0], use_cuda=use_cuda,
                                             static_embeds=(config.get('tune_embeds', '0') == '0'))

        model = bm.BiCondLSTMModel(hidden_dim=int(config['h']), embed_dim=input_layer.dim,
                                   input_dim=(int(config['in_dim']) if 'in_dim' in config['name'] else input_layer.dim),
                                   drop_prob=float(config['dropout']), use_cuda=use_cuda,
                                   num_labels=3, keep_sentences=('keep_sen' in config),
                                   doc_method=config.get('doc_m', 'maxpool'))

        model_kwargs["embed_model"] = input_layer
        model_kwargs["model"] = model
        model_kwargs["optimizer"] = optim.Adam(model.parameters(), lr=lr)
        model_kwargs["setup_fn"] = data_utils.setup_helper_bicond

    elif 'CTSAN' in config['name']:
        _, embeddings = load_word_embeddings(config)
        input_layer = im.BasicWordEmbedLayer(vecs=embeddings, use_cuda=use_cuda)

        model = bm.CTSAN(hidden_dim=int(config['h']), embed_dim=input_layer.dim, att_dim=int(config['a']),
                         lin_size=int(config['lh']), drop_prob=float(config['dropout']),
                         use_cuda=use_cuda, out_dim=3, keep_sentences=('keep_sen' in config),
                         sentence_version=config.get('sen_v', 'default'),
                         doc_method=config.get('doc_m', 'maxpool'),
                         premade_topic=('topic_name' in config),
                         topic_trans=('topic_name' in config),
                         topic_dim=(int(config.get('topic_dim')) if 'topic_dim' in config else None))

        model_kwargs["embed_model"] = input_layer
        model_kwargs["model"] = model
        model_kwargs["optimizer"] = optim.Adam(model.parameters(), lr=lr)
        model_kwargs["setup_fn"] = data_utils.setup_helper_bicond

    elif 'repffnn' in config['name']:
        batch_args["keep_sen"] = False
        input_layer = im.JointBERTLayerWithExtra(vecs=topic_embeddings, use_cuda=use_cuda,
                                                 use_both=(config.get('use_ori_topic', '1') == '1'),
                                                 static_vecs=(config.get('static_topics', '1') == '1'))

        model = bm.RepFFNN(in_dropout_prob=float(config['in_dropout']),
                           hidden_size=int(config['hidden_size']),
                           input_dim=int(config['topic_dim']),
                           use_cuda=use_cuda)

        optimizer = optim.Adam(model.parameters())

        model_kwargs["embed_model"] = input_layer
        model_kwargs["model"] = model
        model_kwargs["optimizer"] = optim.Adam(model.parameters(), lr=lr)
        model_kwargs["setup_fn"] = data_utils.setup_helper_bert_attffnn

# END CONFIGURE THE SPECIFIED MODEL AND HYPER-PARAMETERS

    model_handler = TorchModelHandler(use_cuda=use_cuda, num_gpus=NUM_GPUS,
                                                  checkpoint_path=config.get('ckp_path', 'data/checkpoints/'),
                                                  result_path=config.get('res_path', 'data/gen-stance/'),
                                                  **model_kwargs)

    checkpoints_dir_path = config.get('ckp_path', 'data/checkpoints/')
    model_path = os.path.join(checkpoints_dir_path, f"ckp-[NAME]-{args.ckp_name}.tar")
    model_handler.load(filename=model_path)

    predicted_datasets = []
    for name, data in data_samplers:
        predictions = model_handler.predict_dataset(data)
        predicted_datasets.append((name, data.data.data_file, predictions))
        outpath = os.path.join(args.outdir, f"{name}-preds.csv")
        save_predicted(data, predictions, outpath)

    for name, path in named_predicted:
        data, preds = loaded_predicted = load_predicted(path)
        predicted_datasets.append((name, data, preds))

    eval(predicted_datasets, class_wise=True)
