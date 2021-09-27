
data_name=$1
data_path=$2
outdir=$3

prefix="combined"
frac="0.1"
# -p : prefix=mixed
# -f: frac=$5

while getopts ":f:p:" opt; do
  case $opt in
    f) prefix="$OPTARG"
    ;;
    p) frac="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


# create topic directory if doesn't exist
[ ! -d "${outdir}" ]  && mkdir -p "${outdir}"

# split dataset
echo "Split dataset into train, dev and test"
python data-actions.py split "${data_path}" -t 0.7 -d 0.1 -p "${data_name}" -o "${outdir}"

# combine datasets
echo "Combine train, dev and test datasets with VASTs' datasets"
for dataset in train dev test
do
  vast_path="./VAST/vast_${dataset}.csv"
  new_dataset_path="${outdir}/${data_name}-${dataset}.csv"
  out_path="./${prefix}-${dataset}.csv"
  python data-actions.py concat "${vast_path}" "${new_dataset_path}" --size2 "${frac}" -o "${out_path}"
done