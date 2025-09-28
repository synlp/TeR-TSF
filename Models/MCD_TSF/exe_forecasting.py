import argparse
import torch
import datetime
import json
import yaml
import os
import numpy as np
import random
import sys
# current_dir = os.path.abspath(__file__)
# project_root = os.path.dirname(os.path.dirname(current_dir))
# if project_root not in sys.path:
sys.path.insert(0, "/home/suchen/RL-TSF")
from Models.MCD_TSF.main_model import CSDI_Forecasting
from Models.MCD_TSF.dataset_forecasting import get_dataloader
from Models.MCD_TSF.utils.utils import train, evaluate

def main():
    parser = argparse.ArgumentParser(description="MCD-TSF")
    parser.add_argument("--config", type=str, default="exchange_rate_96_96.yaml")
    parser.add_argument("--datatype", type=str, default="multimodal")
    parser.add_argument('--device', default='cuda:0', help='Device for Attack')
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--unconditional", action="store_true")
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument("--nsample", type=int, default=3)
    parser.add_argument("--data", type=str, default="pretrain")
    parser.add_argument("--data_name", type=str, default="exchange_rate")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument('--root_path', type=str, default='/media/ubuntu/data/collaborations/tsf/TeR-TSF/my_datasets/others', help='')
    parser.add_argument('--data_path', type=str, default='Economy/Economy.csv', help='data file')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--text_len', type=int, default=36, help='context length in time series freq')
    parser.add_argument('--features', type=str, default='S', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--freq', type=str, default='d', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--attn_drop', type=float, default=0.)
    parser.add_argument('--init', type=str, default='None')
    parser.add_argument('--valid_interval', type=int, default=1)
    parser.add_argument('--time_weight', type=float, default=0.1)
    parser.add_argument('--c_mask_prob', type=float, default=-1)
    parser.add_argument('--beta_end', type=float, default=-1)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--save_attn', type=bool, default=False)
    parser.add_argument('--save_token', type=bool, default=False)
    parser.add_argument('--num_steps', type=int, default=-1)
    parser.add_argument('--sample_steps', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=-1)


    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.text_len == 0:
        args.text_len = args.seq_len

    timestep_dim_dict = {
        'd': 3,
        'w': 2,
        'm': 1,
        'h': 4,
    }
    context_dim_dict = {
        'bert': 768,
        'llama': 4096,
        'gpt2': 768
    }
    path = "Models/MCD_TSF/config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    if args.embed == 'timeF':
        if config["model"]["timestep_branch"] or config["model"]["timestep_emb_cat"]:
            config["model"]["timestep_dim"] = timestep_dim_dict[args.freq]
        else:
            config["model"]["timestep_dim"] = 0
    else:
        config["model"]["timestep_dim"] = 4
    config["model"]["context_dim"] = context_dim_dict[config["model"]["llm"]] if config["model"]["with_texts"] else 0

    if args.datatype == 'electricity':
        target_dim = 370
        args.seq_len = 168
        args.pred_len = 24
    else:
        target_dim = 1
    if args.epochs > 0:
        config["train"]["epochs"] = args.epochs
    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["lookback_len"] = args.seq_len
    config["model"]["pred_len"] = args.pred_len
    config["model"]["domain"] = args.data_path.split('/')[0]
    config["model"]["text_len"] = args.text_len
    config["model"]["save_attn"] = args.save_attn
    config["model"]["save_token"] = args.save_token
    config["diffusion"]["dropout"] = args.dropout
    config["diffusion"]["attn_drop"] = args.attn_drop
    config["diffusion"]["time_weight"] = args.time_weight

    if args.c_mask_prob > 0:
        config["diffusion"]["c_mask_prob"] = args.c_mask_prob

    if args.beta_end > 0:
        config["diffusion"]["beta_end"] = args.beta_end

    if args.lr > 0:
        config["train"]["lr"] = args.lr

    if args.num_steps > 0:
        config["diffusion"]["num_steps"] = args.num_steps
    if args.sample_steps > 0:
        config["diffusion"]["sample_steps"] = args.sample_steps

    args.batch_size = config["train"]["batch_size"]

    print(json.dumps(config, indent=4))

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = "./Models/MCD_TSF/save/forecasting_" + args.datatype + '_' + current_time + "/"
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config_results.json", "w") as f:
        json.dump(config, f, indent=4)

    train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
        datatype=args.datatype,
        device= args.device,
        batch_size=config["train"]["batch_size"],
        args=args
    )

    model = CSDI_Forecasting(config, args.device, target_dim, window_lens=[args.seq_len, args.pred_len], freq=args.freq).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            valid_epoch_interval=args.valid_interval
        )
    else:
        model.load_state_dict(torch.load("./Models/MCD-TSF/save/" + args.modelfolder + "/model.pth"))
    model.target_dim = target_dim
    if config["diffusion"]["cfg"]:
        best_mse = 10e10
        for guide_w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0]:
            mse = evaluate(
                model,
                test_loader,
                nsample=args.nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername,
                window_lens=[args.seq_len, args.pred_len],
                guide_w=guide_w,
                save_attn=args.save_attn,
                save_token=args.save_token
            )
            if mse < best_mse:
                best_mse = mse
                # print(best_mse)
            else:
                break
    else:
        evaluate(
                model,
                test_loader,
                nsample=args.nsample,
                scaler=scaler,
                mean_scaler=mean_scaler,
                foldername=foldername,
                window_lens=[args.seq_len, args.pred_len],
                save_attn=args.save_attn,
                save_token=args.save_token
            )

if __name__ == '__main__':
    main()