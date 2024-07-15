# here label_len is not used by model but only provided to data loader, keep it allign to the pre_len

seq_len=336
model_name=PatchMixer

root_path_name=./dataset/btc/
data_path_name=btc_t_v_withf.csv
model_id_name=mbtc
data_name=mbtc

pred_len=96

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name_$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --features MS \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 138 \
  --e_layers 1\
  --d_model 256 \
  --dropout 0.2\
  --seg_len 16\
  --stride 8\
  --des 'Exp' \
  --train_epochs 100\
  --patience 3\
  --itr 1 \
  --batch_size 256 \
  --learning_rate 0.0001 \
  --head_dropout 0 \
  --target range5 \
  --num_workers 0 
  