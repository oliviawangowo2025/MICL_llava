import json, math

def truncate_percent(value, digits=2):
    value_percent = value * 100
    factor = 10 ** digits
    truncated = math.floor(value_percent * factor) / factor
    return f"{truncated:.2f}%"

with open('eval_results/eval_affordance_reasoning_llava-pretrained_output.json', 'r') as f:
    data=json.load(f)

obj_sum_test = {}
print("=== Sample Count per Object ===")
for k, v in data['results'].items():
    if 'score' in v:
        total_samples = sum(v['score'].values())
        obj_sum_test[k] = total_samples
        print(f"{k:<12}: {total_samples}")
    else:
        obj_sum_test[k] = 0
print(f"\nTotal samples: {sum(obj_sum_test.values())}\n")

target_results = ["eval_results/eval_affordance_reasoning_llava-pretrained_output.json"]

for r in target_results:
    with open(r, 'r') as f:
        raw_result = json.load(f)
    
    total_correct = 0
    total = sum(obj_sum_test.values())
    
    print("=== Accuracy per Object ===")
    for k, v in raw_result['scores'].items():
        object_total = obj_sum_test.get(k, 0)
        correct = object_total * v['accuracy']
        total_correct += correct
        print(f"{k:<12}: accuracy = {v['accuracy']:.2%}, samples = {object_total}, correct = {correct:.2f}")
    
    total_acc = total_correct / total if total > 0 else 0
    print(f"\n=== Overall Accuracy ===\nWeighted Total Correct: {total_correct:.2f}\nTotal Samples: {total}\nOverall Accuracy: {truncate_percent(total_acc)}")
