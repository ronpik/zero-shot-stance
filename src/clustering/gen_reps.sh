#!/usr/bin/env bash

prefix="test"
topic_path="../../resources/topicreps/"
num_clusters="197"

while getopts ":a:p:k" opt; do
  case $opt in
    a) prefix="$OPTARG"
    ;;
    p) topic_path="$OPTARG"
    ;;
    k) num_clusters="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# shift options
shift $((OPTIND - 1))

train_data=$1
dev_data=$2
test_data=$3

# create topic directory if doesn't exist
[ ! -d "${topic_path}" ]  && mkdir -p "${topic_path}"

echo "Saving document and topic vectors from BERT"
echo "python stance_clustering.py -m 1 -i \"${train_data}\" -d \"${dev_data}\" -e \"${test_data}\" -a \"${prefix}\" -p \"${topic_path}\""
python stance_clustering.py -m 1 -i "${train_data}" -d "${dev_data}" -e "${test_data}" -a "${prefix}" -p "${topic_path}"

echo "Generating generalized topic representations through clustering"
echo "python stance_clustering.py -m 2 -i ${train_data} -d ${dev_data} -p \"${topic_path}\" -k \"${num_clusters}\" -p \"${topic_path}\""
python stance_clustering.py -m 2 -i "${train_data}" -d "${dev_data}" -p "${topic_path}" -k "${num_clusters}" -p "${topic_path}"

echo "Getting cluster assignments"
echo "python stance_clustering.py -m 3 -k \"${num_clusters}\" -i \"${train_data}\" -d \"${dev_data}\" -e \"${test_data}\" -a \"${prefix}\" -p \"${topic_path}\""
python stance_clustering.py -m 3 -k "${num_clusters}" -i "${train_data}" -d "${dev_data}" -e "${test_data}" -a "${prefix}" -p "${topic_path}"