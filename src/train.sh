#!/usr/bin/env bash

train_data=../data/VAST/vast_train.csv
dev_data=../data/VAST/vast_dev.csv

CURRENT_DATE=$(date '+%Y-%m-%dT%H:%M:%S')

echo "training model with early stopping and $3 warm-up epochs"
#python -u train_model.py -s $1 -i ${train_data} -d ${dev_data} -e 1 -p $2 -k $3
nohup python -u train_model.py -s $1 -i ${train_data} -d ${dev_data} -e 1 -p $2 -k $3 > train_${CURRENT_DATE}.log 2>&1 &
