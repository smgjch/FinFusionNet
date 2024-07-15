# Note even in the manner of seq to seq, keep pred_len as 0 since it always aligns with label length
# Since miTransformer use mbtc that label the same sequence of input,enc_in is used and should be set to the number of features (exclude the label)
export CUDA_VISIBLE_DEVICES=0

model_name=miTransformer


    python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_t_v_withf.csv \
  --model_id btc_96_96 \
  --model $model_name \
  --data btc \
  --features MS \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 139 \
  --dec_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --target range5 \
  --num_workers 0  \
  --freq t \
  --verbose 0 \
  --batch_size 64  



    python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_t_v_withf.csv \
  --model_id btc_96_96 \
  --model $model_name \
  --data btc \
  --features MS \
  --seq_len 96 \
  --label_len 10 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 139 \
  --dec_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --target range5 \
  --num_workers 0  \
  --freq t \
  --verbose 0 \
  --batch_size 64  

      python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/btc/ \
  --data_path btc_t_v_withf.csv \
  --model_id btc_96_96 \
  --model $model_name \
  --data btc \
  --features MS \
  --seq_len 96 \
  --label_len 1 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 139 \
  --dec_in 7 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \
  --target range5 \
  --num_workers 0  \
  --freq t \
  --verbose 0 \
  --batch_size 64  

