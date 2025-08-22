# MICL: 

This repository provides scripts and code for finetuning LLaVA with LoRA and performing affordance reasoning inference on our collected dataset.

Our approach combines two key techniques:
1. Multimodal In-Context Learning (MICL) — leveraging visual-language demonstrations inside the prompt to guide affordance understanding.

2. LoRA Finetuning — efficient low-rank adaptation of LLaVA to specialize on object affordance tasks without full model retraining.

## Contributions

1. Dataset Collection: We construct an object affordance dataset, where objects may be in different states (e.g., clean vs dirty, available vs unavailable). This dataset enables training embodied AI models to reason about affordances in realistic, dynamic environments.

2. Method Development: We release scripts for LoRA-based finetuning of LLaVA (adapted from LLaVA v1.5 scripts) as well as evaluation scripts for affordance reasoning.

3. Efficient Adaptation: By combining Multimodal In-Context Learning with LoRA Finetuning, our method achieves strong performance on affordance recognition while remaining computationally efficient and scalable.

## Setup Environment
```bash
conda create -n llava python==3.10
conda activate llava
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y 

git clone https://github.com/haotian-liu/LLaVA.git && cd LLaVA
pip install --no-deps -e .
cd ..

# Copy scripts for finetuning and inference
cp run_llava.py LLaVA/llava/eval/run_llava.py
cp llava_was.py LLaVA/llava_was.py
cp finetune_task_lora.sh LLaVA/scripts/v1_5/finetune_task_lora.sh
cp finetune_merge_lora.sh LLaVA/finetune_merge_lora.sh
```

## Dataset Format
Training/validation datasets follow a JSON list format. Each entry contains:

1. `id`: Unique identifier for the sample

2. `image`: Path to the corresponding image

3. `conversations`: Multimodal instruction–response pairs
```

Examples:
```json
[
  {
    "id": "trial_T20190909_040738_543042_3_Pot|-00.52|+00.83|+01.68_000000119",
    "image": "all_finetune_dataset_templated_pair_new/pot_dirty/trial_T20190909_040738_543042_3_Pot|-00.52|+00.83|+01.68_000000119.png",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\n The image shows three pots: clean (left, single color), dirty (middle, with stains), and the query (right). Is the query pot used? Answer only 'yes' or 'no'."
      },
      {
        "from": "gpt",
        "value": "yes"
      }
    ]
  }
]
```

## Quick Start — LoRA Finetuning
We provide a ready-to-use script adapted from [finetune_task.sh](https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/finetune_task.sh)
```bash 
cd LLaVA
bash scripts/v1_5/finetune_task_lora.sh
```
This will finetune LLaVA with LoRA adapters on the provided dataset. Adjust hyperparameters in the script as needed for your hardware and data size.

## Merge weights after finetuning
After training, merge the LoRA weights into the base LLaVA model for standalone deployment:
```bash 
bash finetune_merge_lora.sh
```

## Training and validation dataset 
- Full dataset: Will be released soon.

- Small dataset: Available at `all_finetune_dataset_all_templated_pair_random/` for quick experiments. 

## Evaluation of Affordance Reasoning Capability
We provide evaluation code to test the model’s affordance reasoning ability on held-out validation sets.
```bash
unzip all_finetune_dataset_all_templated_pair_random.zip
python eval_Affordance_Reasoning.py
```
