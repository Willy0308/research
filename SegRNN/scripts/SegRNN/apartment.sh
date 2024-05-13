if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi


root_path_name=./dataset/ausdata
data_path_name=ausdata.csv
model_id_name=Electricity
data_name=custom

seq_len=720 #720 (1 month) to predict 5 days, 4320 (~6 months) to predict 1 month, and 8640 (~1 year) to predict 1 month
train_epochs=100
batch_size=128
learning_rate=0.001

# for model_name in SegRNN Informer Autoformer Transformer DLinear Linear NLinear PatchTST VanillaRNN
for model_name in Transformer SegRNN Informer DLinear PatchTST VanillaRNN
do
    # for pred_len in 96 192 336 720
    for pred_len in 120
    do
        python -u run_longExp.py \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name'_'$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len $seq_len \
          --pred_len $pred_len \
          --seg_len 24 \
          --enc_in 6 \
          --d_model 512 \
          --dropout 0.1 \
          --train_epochs $train_epochs \
          --patience 30 \
          --rnn_type gru \
          --dec_way pmf \
          --channel_id 1 \
          --itr 1 --batch_size $batch_size --learning_rate $learning_rate > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$data_path_name'_'$train_epochs'_'$batch_size'_'$learning_rate.log
    done
done




# if [ ! -d "./logs" ]; then
#     mkdir ./logs
# fi

# if [ ! -d "./logs/LongForecasting" ]; then
#     mkdir ./logs/LongForecasting
# fi
# model_name=SegRNN

# root_path_name=./dataset/electricity
# # 定义一个数组包含所有的数据文件名
# # data_files=("seasonal.csv" "resid.csv" "trend.csv")
# model_id_name=Electricity
# data_name=custom

# seq_len=720
# for data_file in seasonal.csv resid.csv trend.csv
# do
#     # for pred_len in 96 192 336 720
#     for pred_len in 96
#     do
#         python -u run_longExp.py \
#           --is_training 1 \
#           --root_path $root_path_name \
#           --data_path $data_file \
#           --model_id $model_id_name'_'$seq_len'_'$pred_len \
#           --model $model_name \
#           --data $data_name \
#           --features M \
#           --seq_len $seq_len \
#           --pred_len $pred_len \
#           --seg_len 48 \
#           --enc_in 321 \
#           --d_model 512 \
#           --dropout 0.1 \
#           --train_epochs 30 \
#           --patience 20 \
#           --rnn_type gru \
#           --dec_way pmf \
#           --channel_id 1 \
#           --itr 1 --batch_size 16 --learning_rate 0.0005 > logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$data_file.log
#     done
# done

