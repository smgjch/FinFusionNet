export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST
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
  --label_len 1 \
  --pred_len 0 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 5 \
  --target range5 \
  --num_workers 0 \
  --batch_size 100
