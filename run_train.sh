python train_net_pai.py --checkpointDir=/home/ad/code/FDCNN_tensorflow/checkpoint/resnet/ \
--buckets=/home1/data/TestDataFromWen/arranged/steady_condition/pkl/ \
--input_size=8192 \
--train_split=0.8 \
--train_speed_list=50,30,20,10 \
--test_speed_list=40 \
--divide_step=5 \
--do_fft=false \
--do_norm=false \
--use_speed=true \
--network=sphere20 \
--num_epoch=20000 \
--batch_size=64 \
--learning_rate=0.0001 \
--early_stop=false \
--stop_standard=adatest \
--accuracy_threshold=1 \
--accuracy_delta=0.01 \
--speed_loss_factor=0