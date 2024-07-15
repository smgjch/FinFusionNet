export CUDA_VISIBLE_DEVICES=0

model_name=PatchTST

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_t_v_withf.csv \
  --model_id btc_96_96 \
  --model $model_name \
  --data mbtc \
  --features MS \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --itr 1 \
  --target range5 \
  --num_workers 0 \
  --batch_size 100
