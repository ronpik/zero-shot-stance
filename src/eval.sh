#!/usr/bin/env bash


prefix="test"
topic_path="../resources/topicreps/"
config_path="../config/config-tganet.txt"
outdir="./"
action=$1
shift

while getopts "a:p:c:o:k:" opt; do
  case $opt in
    a) prefix="$OPTARG"
    ;;
    p) topic_path="$OPTARG"
    ;;
    c) config_path="$OPTARG"
    ;;
    o) outdir="$OPTARG"
    ;;
    k) model_suffix="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

# shift options
shift $((OPTIND - 1))

train_data=$1
test_data=$2
outpath="${outdir}/${prefix}-preds.csv"


if [ "${action}" == 'eval' ]; then
    echo "Evaluating a model"
elif [ "$action" == 'predict' ]; then
    echo "Saving predictions from a model to ${outpath}"
else
    echo "Unknown action: ${action}"
    exit
fi

echo "python eval_model.py -m \"${action}\" -k \"${model_suffix}\" -s \"${config_path}\" -i \"${train_data}\" -d \"${test_data}\" -a \"${prefix}\" -p \"${topic_path}\" -o \"${outpath}\""
python eval_model.py -m "${action}" -k "${model_suffix}" -s "${config_path}" -i "${train_data}" -d "${test_data}" -a "${prefix}" -p "${topic_path}" -o "${outpath}"


