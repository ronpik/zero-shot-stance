#!/usr/bin/env bash

config_path="../config/config-tganet.txt"
num_warmups="0"
scorer="f_macro"
topic_dir="../resources/topicreps"

while getopts "c:p:k:t:" opt; do
  case $opt in
    c) config_path="$OPTARG"
    ;;
    p) num_warmups="$OPTARG"
    ;;
    k) scorer="$OPTARG"
    ;;
    t) topic_dir="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# shift options
shift $((OPTIND - 1))

train_data=$1
dev_data=$2

CURRENT_DATE=$(date '+%Y-%m-%dT%H:%M:%S')

echo "Training model with early stopping and ${num_warmups} warm-up epochs"
echo "python -u train_model.py -s \"${config_path}\" -t \"${topic_dir}\" -i \"${train_data}\" -d \"${dev_data}\" -e 1 -p \"${num_warmups}\" -k \"${scorer}\""
nohup python -u train_model.py -s "${config_path}" -t "${topic_dir}" -i "${train_data}" -d "${dev_data}" -e 1 -p "${num_warmups}" -k "${scorer}" > train_${CURRENT_DATE}.log 2>&1 &
