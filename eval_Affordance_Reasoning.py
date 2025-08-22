from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import AutoModelForCausalLM, AutoTokenizer
from llava.model.language_model.llava_llama import LlavaConfig
from llava.model import *
from peft import PeftModel, PeftConfig
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model_was
from llava.utils import disable_torch_init
from PIL import Image
import requests
import os
import json
import time
from tqdm import tqdm

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def cal_score(y_true, y_pred, pos_label):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    return {"accuracy":accuracy, "precision":precision, "recall":recall, "f1":f1}

def save_output(score, result_dict, exception_dict, path):
    output_data = {
        "scores":score, 
        "results": result_dict,
        "exceptions": exception_dict
    }

    output_path = path
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2, separators=(",", ":"))

    print(f"Results saved to {output_path}")

class llava_bot:
    def __init__(self, model_path):
        self.model_path = model_path
        """self.args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": get_model_name_from_path(self.model_path),
            "query": None,
            "conv_mode": None,
            "image_file": None,
            "sep": ",",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()
        """
        disable_torch_init()
        self.model_name = get_model_name_from_path(model_path)
        print("model name / model path:", self.model_name, model_path)
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
        model_path, None, self.model_name
        ) # modify: load 4 bit

    def ask_llava(self, prompt, image_file):

        response = eval_model_was(prompt, image_file, self.model, self.model_name, self.image_processor, self.tokenizer)

        #response = eval_model(args)
        print("prompt: ", prompt)
        print("response: ", response)
        return response
    
    def test_data(self, json_file):
        exception_dict = {}
        result_dict = {}

        with open(json_file) as f:
            data = json.load(f)
            cnt = 0
            for d in tqdm(data):
                cnt += 1
                
                cat = (d['image'].split('/')[1]).split('_')[0]
                if cat not in result_dict.keys():
                    result_dict[cat] = {"score":{"TP":0, "TN":0, "FP":0, "FN":0}, 
                                        "gt":[], "pred":[]}
                    exception_dict[cat] = {}

                

                img = load_image(d['image'])
                prompt = d['conversations'][0]['value'][9:]
                gt = d['conversations'][1]['value'].lower()
                res = self.ask_llava(prompt=prompt, image_file=img)
                res = res.lower()

                result_dict[cat]["gt"].append(gt)
                result_dict[cat]["pred"].append(res)

                if cat == "microwave":
                    val = ["available", "occupied"]
                    t = "available"
                    f = "occupied"
                else:
                    val = ["yes", "no"]
                    t = "yes"
                    f = "no"

                if res not in val:
                    if res not in exception_dict[cat]:
                        exception_dict[cat][res] = 1
                    else:
                        exception_dict[cat][res] += 1

                if gt == t and res == t: result_dict[cat]["score"]["TP"] += 1
                elif gt == f and res == f: result_dict[cat]["score"]["TN"] += 1
                elif gt == f and res == t: result_dict[cat]["score"]["FP"] += 1
                elif gt == t and res == f: result_dict[cat]["score"]["FN"] += 1


                #if cnt == 20:
                #    break

        score = {}
        for cat, val in result_dict.items():
            if cat == "microwave": 
                pos = "available"
            else:
                pos = "yes"
            score[cat] = cal_score(y_true=result_dict[cat]["gt"], y_pred=result_dict[cat]["pred"], pos_label=pos)
        return score, result_dict, exception_dict
                
start = time.time()
llava_was = llava_bot('liuhaotian/llava-v1.5-7b')
score, result_dict, exception_dict = llava_was.test_data("0615_finetune_val_templated_pair_correct_woGD_random.json")
save_output(score, result_dict, exception_dict, "eval_results/eval_affordance_reasoning_llava-pretrained_output_correct_woGD_random.json")
end = time.time()
print("Time : ", end - start)


## the checkpoint of LoRA fine-tuning llava-v1.5-7b will be released upon acceptance and can be evaluated as follows:
'''
start = time.time()
llava_was = llava_bot('fine_tune_llava_lora_0615-pair')
score, result_dict, exception_dict = llava_was.test_data("finetune_llava/0615_finetune_val_templated_pair_correct_woGD_random.json")
save_output(score, result_dict, exception_dict, "eval_results/eval_affordance_reasoning_llava-0615-pair_output_correct_woGD_random.json")
end = time.time()
print("Time : ", end - start)
'''