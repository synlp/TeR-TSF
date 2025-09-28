#!/bin/bash
set -e

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --llm_type) llm_type="$2"; shift ;;
            --tsf_type) tsf_type="$2"; shift ;;
            --data_name) data_name="$2"; shift ;;
            --hist_len) hist_len="$2"; shift ;;
            --pred_len) pred_len="$2"; shift ;;
            --exp_time) exp_time="$2"; shift ;;
            --batch_size) batch_size="$2"; shift ;;
            --gen_num) gen_num="$2"; shift ;;
            --iter_num) iter_num="$2"; shift ;;
            --lora_rank) lora_rank="$2"; shift ;;
            --lr) lr="$2"; shift ;;
            --num_train_epochs) num_train_epochs="$2"; shift ;;
            --per_device_train_batch_size) per_device_train_batch_size="$2"; shift ;;
            --llama_factory_dir) llama_factory_dir="$2"; shift ;;
            --down_sample) down_sample="$2"; shift ;;
            --disable_text_quality_reward) disable_text_quality_reward="--disable_text_quality_reward" ;;
            *) echo "未知参数: $1"; exit 1 ;;
        esac
        shift
    done
}

get_llm_config() {
    local llm_type=$1
    
    case "$llm_type" in
        "qwen3-1.7b")
            echo "qwen3:/data2/user2/Qwen3-1.7B"
            ;;
        "qwen3-8b")
            echo "qwen3:/data2/user2/Qwen3-8B"
            ;;
        "llama3.1-8b")
            echo "llama3:/data2/user2/Llama-3.1-8B"
            ;;
        "llama3.2-1b")
            echo "llama3:/data2/user2/Llama-3.2-1B"
            ;;
        "llama3.2-3b")
            echo "llama3:/data2/user2/Llama-3.2-3B"
            ;;
        "qwen3-4b")
            echo "qwen3:/data2/user2/Qwen3-4B"
            ;;
        *)
            echo "error '$llm_type'" >&2
            exit 1
            ;;
    esac
}

setup_model_paths() {
    local iter_idx=$1
    local llm_template=$2
    local initial_path=$3
    
    
    local path_identifier="${hist_len}_${pred_len}_${exp_time}"
    
    if (( iter_idx == 0 )); then
        llm_path="$initial_path"
    else
        local found_model=0
        local search_iter=$((iter_idx - 1))
        while (( search_iter >= 0 )); do
            local prev_merge_path="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/merge/iter${search_iter}"
            if [ -d "$prev_merge_path" ]; then
                llm_path="$prev_merge_path"
                found_model=1
                break
            fi
            search_iter=$((search_iter - 1))
        done
        if (( found_model == 0 )); then
            llm_path="$initial_path"
        fi
    fi
    
    local dataset_name="${data_name}_h${hist_len}_p${pred_len}_${llm_type}_${tsf_type}_genNum${gen_num}_iter${iter_idx}_${exp_time}"
    adapter_output_dir="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/lora/iter${iter_idx}"
    merge_output_dir="/data2/user2/ter_tsf/models/${llm_type}/${tsf_type}/${data_name}/${path_identifier}/merge/iter${iter_idx}"
    
    echo "llm_path = ${llm_path}"
    echo "adapter_output_dir = ${adapter_output_dir}"
    echo "merge_output_dir = ${merge_output_dir}"
}

run_data_preparation() {
    local iter_idx=$1
    local llm_path=$2
    
    python prepare_stage.py \
        --data_dir /data2/user2/ter_tsf \
        --data_name $data_name \
        --llm_type $llm_type \
        --tsf_type $tsf_type \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size $batch_size \
        --gen_num $gen_num \
        --iter_idx $iter_idx \
        --llm_path $llm_path \
        --exp_time $exp_time \
        --llama_factory_dir $llama_factory_dir \
        --down_sample $down_sample \
        $disable_text_quality_reward
    
    local exit_code=$?
    
    if (( exit_code == 0 )); then
        return 0
    elif (( exit_code == 1 )); then
        return 1
    else
        exit $exit_code
    fi
}

run_tfhts_training() {
    local iter_idx=$1
    
    local text_type="original_text"
    if (( iter_idx > 0 )); then
        text_type="reinforced_text"
    fi
    
    local data_dir="/data2/user2/ter_tsf/${llm_type}/${tsf_type}/${data_name}/reinforced_data"
    local save_dir="/data2/user2/ter_tsf/${llm_type}/tfhts/${data_name}/saved_models"

    local patch_len=16
    local stride=8

    if [[ "$data_name" == "weather"* ]] || [[ "$data_name" == "Heart_Rate"* ]]; then
        patch_len=4
        stride=2
    fi
    
    
    python ./Models/TextFusionHTS/train_tfhts.py \
        --data_dir $data_dir \
        --save_dir $save_dir \
        --data_name $data_name \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size 32 \
        --text_type $text_type \
        --patch_len $patch_len \
        --stride $stride \
        --epochs 200 \
        --lr 1e-3 \
        --exp_time $exp_time \
        --iter_idx $iter_idx
    
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "TextFusionHTS training completed."
    else
        echo "TextFusionHTS training failed: $exit_code"
        exit 1
    fi
}

run_dpo_training() {
    local iter_idx=$1
    local llm_path=$2
    local llm_template=$3
    local dataset_name=$4
    
    
    if [ ! -d "$adapter_output_dir" ]; then
        pushd "$llama_factory_dir" > /dev/null
        
        local log_file="${adapter_output_dir}/training.log"
        mkdir -p "$adapter_output_dir"
        
        
        llamafactory-cli train \
            --model_name_or_path $llm_path \
            --stage dpo \
            --do_train \
            --finetuning_type lora \
            --lora_rank $lora_rank \
            --lora_target all \
            --pref_beta 0.1 \
            --pref_loss sigmoid \
            --dataset $dataset_name \
            --dataset_dir "/home/user2/projects/TeR_TSF/llama-factory-main/data" \
            --template $llm_template \
            --cutoff_len 5120 \
            --max_samples 3000 \
            --overwrite_cache \
            --preprocessing_num_workers 128 \
            --dataloader_num_workers 128 \
            --output_dir $adapter_output_dir \
            --logging_steps 10 \
            --save_steps 500 \
            --per_device_train_batch_size $per_device_train_batch_size \
            --gradient_accumulation_steps 8 \
            --learning_rate $lr \
            --num_train_epochs $num_train_epochs \
            --lr_scheduler_type cosine \
            --warmup_ratio 0.1 \
            --bf16 \
            --ddp_timeout 180000000 \
            2>&1 | tee "$log_file"
        
        local exit_code=${PIPESTATUS[0]}
        
        if [ $exit_code -eq 0 ]; then
            echo "DPO training completed"
        else
            echo "DPO training failed: $exit_code"
            exit 1
        fi
        
        popd > /dev/null
    else
        echo "Skipping..."
    fi
}

export_model() {
    local iter_idx=$1
    local llm_path=$2
    local llm_template=$3
    
    
    if [ ! -d "$merge_output_dir" ]; then
        pushd "$llama_factory_dir" > /dev/null
        llamafactory-cli export \
            --model_name_or_path $llm_path \
            --adapter_name_or_path $adapter_output_dir \
            --template $llm_template \
            --export_dir $merge_output_dir \
            --export_size 5 \
            --export_device cpu \
        || { echo "failed"; exit 1; }
        popd > /dev/null
    else
        echo "Skipping..."
    fi
}

run_evaluation() {
    local iter_idx=$1
    local merge_output_dir=$2

    
    python evaluate.py \
        --data_dir /data2/user2/ter_tsf \
        --data_name $data_name \
        --llm_type $llm_type \
        --tsf_type $tsf_type \
        --hist_len $hist_len \
        --pred_len $pred_len \
        --batch_size $batch_size \
        --iter_idx $iter_idx \
        --llm_path $merge_output_dir \
        --exp_time $exp_time \
        --gen_num $gen_num \
        --dpo_lr $lr \
        --dpo_epoch $num_train_epochs \
        --down_sample $down_sample
}

main() {
    
    local llm_config=$(get_llm_config "$llm_type")
    local llm_template=$(echo "$llm_config" | cut -d: -f1)
    local initial_path=$(echo "$llm_config" | cut -d: -f2)
    
    for ((iter_idx=0; iter_idx<$iter_num; iter_idx++)); do
        
        setup_model_paths $iter_idx $llm_template $initial_path
        
        if ! run_data_preparation $iter_idx $llm_path; then
            continue
        fi
        
        run_tfhts_training $iter_idx
        
        local dataset_name="${data_name}_h${hist_len}_p${pred_len}_${llm_type}_${tsf_type}_genNum${gen_num}_iter${iter_idx}_${exp_time}"
        run_dpo_training $iter_idx $llm_path $llm_template $dataset_name
        
        export_model $iter_idx $llm_path $llm_template

        run_evaluation $iter_idx $merge_output_dir
        
    done
    
}

parse_arguments "$@"

main