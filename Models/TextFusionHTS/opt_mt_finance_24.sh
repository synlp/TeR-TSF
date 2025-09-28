# export CUDA_VISIBLE_DEVICES=0

data_dir=/data2/user2/ter_tsf/qwen3-1.7b/tfhts/MTBench_finance/reinforced_data/
save_dir=/data2/user2/ter_tsf/TextFusionHTS/
exp_time=001
data_name=MTBench_finance
hist_len=96
pred_len=24
iter_idx=3

python ./Models/TextFusionHTS/optimize.py \
    --study_name ${data_name}_reinforced_text_${hist_len}_${pred_len} \
    --data_dir $data_dir \
    --save_dir $save_dir \
    --data_name $data_name \
    --hist_len $hist_len \
    --pred_len $pred_len \
    --text_type reinforced_text \
    --exp_time $exp_time \
    --iter_idx $iter_idx \
    --epochs 1
