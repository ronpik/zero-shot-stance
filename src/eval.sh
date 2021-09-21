#!/usr/bin/env bash


train_data=../data/VAST/vast_train.csv
dev_data=../data/VAST/vast_dev.csv
fourforums_data=../data/fourforums/4forum.zs.csv
#dev_data="/Users/ronpick/workspace/zero-shot-stance/data/createdebate/cdvast.csv"

dataname=$3
datapath=$4

if [ $1 == 'eval' ]
then
    echo "Evaluating a model"
    python eval_model.py -m "eval" -k BEST -s $2 -i ${train_data} -d ${dev_data}

elif [ $1 == 'predict' ]
then
    echo "Saving predictions from a model to $3"
    echo "eval_model.py -m \"predict\" -k BEST -s $2 -i ${train_data} -d \"${datapath}\" -a \"${dataname}\" -o \"${dataname}\""
    python eval_model.py -m "predict" -k BEST -s $2 -i ${train_data} -d "${datapath}" -a "${dataname}" -o "${dataname}"


else
    echo "Doing nothing"
fi
