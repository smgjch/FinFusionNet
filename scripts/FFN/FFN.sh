export CUDA_VISIBLE_DEVICES=1

model_name=FFN

seq_len=30

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_t_v_withf.csv \
  --model_id BTC_$seq_len'_'$pred_len \
  --model $model_name \
  --data mbtc \
  --features MS \
  --seq_len $seq_len \
  --pred_len 0 \
  --label_len 1 \
  --kernel_size 2\
  --num_kernels 16 \
  --enc_in 138 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --batch_size 10 \
  --patience 10 \
  --train_epochs 1000
