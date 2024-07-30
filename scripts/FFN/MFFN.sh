export CUDA_VISIBLE_DEVICES=0

model_name=MFFN

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
  --num_kernels 2 \
  --enc_in 138 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --batch_size 128 \
  --patience 3 \
  --is_training 1 \
  --train_epochs 1000

  # --gradient_checkpoint\
