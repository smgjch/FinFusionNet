  # --enc_in 138 \ Note this should be the input demesion of features
export CUDA_VISIBLE_DEVICES=0

model_name=SegRNN

seq_len=30



python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_8tc_noi.csv \
  --model_id BTC_$seq_len'_'$pred_len \
  --model $model_name \
  --data mbtc_block \
  --features MS \
  --seq_len $seq_len \
  --pred_len 0 \
  --label_len 1 \
  --seg_len 2 \
  --enc_in 4 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 


