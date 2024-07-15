export CUDA_VISIBLE_DEVICES=0

model_name=Baseline

seq_len=1

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
  --seg_len 24 \
  --enc_in 138 \
  --d_model 512 \
  --dropout 0.5 \
  --learning_rate 0.0001 \
  --des 'Exp' \
  --itr 1 \
  --num_workers 0 \
  --batch_size 2048
