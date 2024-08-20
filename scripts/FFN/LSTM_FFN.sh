export CUDA_VISIBLE_DEVICES=0


model_name=LSTM_FFN_ablation
seq_len=2

  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path visual_inspect_set.csv \
  --model_id BTC_$seq_len'_'$pred_len \
  --model $model_name \
  --data mbtc_block \
  --features MS \
  --seq_len $seq_len \
  --pred_len 0 \
  --label_len 1 \
  --kernel_size 2 \
  --num_kernels 2 \
  --enc_in 3 \
  --des 'Exp' \
  --itr 10 \
  --num_workers 0 \
  --batch_size 3 \
  --patience 3 \
  --is_training 1 \
  --use_amp\
  --train_epochs 1000\
  --write_graph\
  --verbose 1

