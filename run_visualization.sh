data_dir="/home1/TestDataFromWen/arranged/steady_condition/pkl/"
model_path="checkpoint/dvgg19/2018-09-19_204935/model.ckpt"
debug=$1
if [ $debug ]
then
    python -m pdb ./visualization/test_cnn_local.py --data_dir=$data_dir\
    --model_path=$model_path
else
    python ./visualization/test_cnn_local.py --data_dir=$data_dir --model_path=$model_path
fi
