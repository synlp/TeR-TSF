import re
from statistics import mode

import numpy as np
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from Models.ChatTime.utils.prompt import getPrompt
from Models.ChatTime.utils.tools import Discretizer, Serializer


class ChatTime:
    def __init__(self, model_path, hist_len=None, pred_len=None,
                 max_pred_len=16, num_samples=8, top_k=100, top_p=1.0, temperature=1.0):
        self.model_path = model_path
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                max_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                # max_length=3072
            )
            
            
            samples = pipe(serialized_series)
            # except torch.OutOfMemoryError:
            #     # import pdb; pdb.set_trace()
            #     print(samples)

            pred_list = []
            for sample in samples:
                serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                if len(dispersed_prediction) < 1:
                    pred = np.array([np.NaN])
                else:
                    pred = self.discretizer.inverse_discretize(dispersed_prediction)
                # pred = self.discretizer.inverse_discretize(dispersed_prediction)

                if len(pred) < min(remaining, self.max_pred_len):
                    pred = np.concatenate([pred, np.full(min(remaining, self.max_pred_len) - len(pred), np.NaN)])

                pred_list.append(pred[:min(remaining, self.max_pred_len)])

            prediction = np.nanmedian(pred_list, axis=0)
            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)

        return prediction

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )
        samples = pipe(serialized_series)

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response
