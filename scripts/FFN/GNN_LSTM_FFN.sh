export CUDA_VISIBLE_DEVICES=0


model_name=GNN_LSTM_FFN
seq_len=30

  python -u run.py \
  --task_name long_term_forecast_CGNN \
  --is_training 1 \
  --root_path ./dataset/btc_raw/ \
  --data_path btc_lowerhalf_ponly.csv \
  --model_id BTC_$seq_len'_'$pred_len \
  --model $model_name \
  --data mDataset_btc_GNN \
  --features MS \
  --seq_len $seq_len \
  --pred_len 0 \
  --label_len 1 \
  --kernel_size 2 \
  --num_kernels 2 \
  --enc_in 5 \
  --des 'Exp' \
  --itr 5 \
  --num_workers 0 \
  --batch_size 128 \
  --patience 3 \
  --is_training 1 \
  --use_amp\
  --train_epochs 1000\
  --GNN_type 0