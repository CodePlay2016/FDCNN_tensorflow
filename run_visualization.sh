data_dir="/home1/TestDataFromWen/arranged/steady_condition/pkl/"
model_path="checkpoint/cvgg19/2018-09-10_163046/model.ckpt"
debug=$1
if [ $debug = true ]
then
    python -m pdb ./visualization/test_cnn_local.py --data_dir=$data_dir\
    --model_path=$model_path
else
    python ./visualization/test_cnn_local.py --data_dir=$data_dir --model_path=$model_path
fi
