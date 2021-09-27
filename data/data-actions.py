from typing import Tuple, Sequence
import os

import argparse

import pandas as pd


def to_absolute_size(frac_or_size: float, total: int) -> int:
    if frac_or_size < 0 or frac_or_size > total:
        return int(total)
    if frac_or_size >= 1:
        return int(frac_or_size)

    return int(frac_or_size * total)


def sample(data: pd.DataFrame, size: float) -> pd.DataFrame:
    absolute_size = to_absolute_size(size, total=len(data))
    return data.sample(n=absolute_size, random_state=1919)


def merge_datasets(data1: pd.DataFrame, data2: pd.DataFrame, size1: float, size2: float, shuffle: bool) -> pd.DataFrame:
    sample1 = sample(data1, size1)
    sample2 = sample(data2, size2)

    merged_data = sample1.append(sample2)
    if shuffle:
        merged_data = merged_data.sample(frac=1, random_state=1919)

    return merged_data


def split_data(dataset: pd.DataFrame, train_size: float, dev_size: float, shuffle: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if shuffle:
        dataset = data.sample(frac=1, random_state=1919)

    train_size = to_absolute_size(train_size, total=len(dataset))
    dev_size = to_absolute_size(dev_size, total=len(dataset))

    train_data = dataset[:train_size]
    dev_data = dataset[train_size: train_size + dev_size]
    test_data = dataset[train_size + dev_size:]
    return train_data, dev_data, test_data


def store_data(outdir:str, datasets: Sequence[pd.DataFrame], names: Sequence[str], prefix: str = None):
    prefix = "" if prefix is None else f"{prefix}-"
    paths = [os.path.join(outdir, f"{prefix}{name}.csv") for name in names]
    for path, dataset in zip(paths, datasets):
        dataset.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--no-shuffle", action='store_const', const=True, default=False,
                        help="prevent datasets from shuffling")

    subparsers = parser.add_subparsers(title="Actions on Data", dest="action_name")

    # concat parser
    concat_parser = subparsers.add_parser("concat", help="concatenating two datasets vertically")
    concat_parser.add_argument("data1", type=str,
                        help="Path to the first dataset to merge")
    concat_parser.add_argument("data2", type=str,
                        help="Path to the second dataset to merge")
    concat_parser.add_argument("--size1", type=float, default=-1,
                        help="Size (or fraction) of the first dataset to sample")
    concat_parser.add_argument("--size2", type=float, default=-1,
                        help="Size (or fraction) of the second dataset to sample")
    concat_parser.add_argument("--outpath", "-o", type=str, default="merged.csv",
                        help="Path of the output file")

    # split parser
    split_parser = subparsers.add_parser("split", help="split a dataset into train, dev and test datasets")
    split_parser.add_argument("data", type=str,
                               help="Path to the first dataset to merge")
    split_parser.add_argument("--train-size", "-t", type=float, default=0.7,
                               help="Size (or fraction) of the first dataset to sample")
    split_parser.add_argument("--dev_size", "-d", type=float, default=-0.1,
                               help="Size (or fraction) of the second dataset to sample")
    split_parser.add_argument("--prefix", "-p", type=str, default=None,
                               help="prefix to add to the output name of the split datasets")
    split_parser.add_argument("--outdir", "-o", type=str, default="merged.csv",
                              help="Path of the output directory where to store the split datasets")

    args = parser.parse_args()

    if args.action_name == "concat":
        data1 = pd.read_csv(args.data1)
        data2 = pd.read_csv(args.data2)
        data = merge_datasets(data1, data2, args.size1, args.size2, shuffle=(not args.no_shuffle))
        data.to_csv(args.outpath, index=False)

    elif args.action_name == "split":
        data = pd.read_csv(args.data)
        datasets = split_data(data, args.train_size, args.dev_size, shuffle=(not args.no_shuffle))
        store_data(args.outdir, datasets, names=["train", "dev", "test"], prefix=args.prefix)


