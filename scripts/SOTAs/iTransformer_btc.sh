# Note even in the manner of seq to seq, keep pred_len as 0 since it always aligns with label length

export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
seq_len=30

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_8tc_noi.csv \
  --model_id BTC_$seq_len'_'$pred_len \
  --model $model_name \
  --data mbtc \
  --features MS \
  --seq_len $seq_len \
  --label_len 1 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 7 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 5 \
  --target range5 \
  --num_workers 0  \
  --freq t \
  --batch_size 64

