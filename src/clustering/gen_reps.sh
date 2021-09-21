#!/usr/bin/env bash

train_data=../../data/VAST/vast_train.csv
dev_data=../../data/VAST/vast_dev.csv
#test_data=../../data/VAST/vast_test.csv
#test_data=../../data/fourforums/4forums.zs.csv

dataname=$1
datapath=$2

echo "Saving document and topic vectors from BERT"
echo "python stance_clustering.py -m 1 -i ${train_data} -d ${dev_data} -e ${datapath} -a ${dataname}"
python stance_clustering.py -m 1 -i ${train_data} -d ${dev_data} -e ${datapath} -a ${dataname}

echo "Generating generalized topic representations through clustering"
echo "python stance_clustering.py -m 2 -i ${train_data} -d ${dev_data} -p ../../resources/topicreps/ -k 197"
python stance_clustering.py -m 2 -i ${train_data} -d ${dev_data} -p ../../resources/topicreps/ -k 197

echo "Getting cluster assignments"
echo "python stance_clustering.py -m 3 -k 197 -i ${train_data} -d ${dev_data} -e ${datapath} -a ${dataname} -p ../../resources/topicreps/"
python stance_clustering.py -m 3 -k 197 -i ${train_data} -d ${dev_data} -e "${datapath}" -a "${dataname}" -p ../../resources/topicreps/