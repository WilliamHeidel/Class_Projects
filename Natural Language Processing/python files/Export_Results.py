
import evaluate
import os
import json
from tqdm import tqdm

squad_metric = evaluate.load("squad")

path = "Results_Fixed_Change"

def file_performance_replace(file_path, squad_metric=squad_metric):
    # Reading the .jsonl file
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    # Path to your output .jsonl file
    output_file_path = file_path.replace('.jsonl', '_mistakes.jsonl')

    # Delete the file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Writing to .jsonl file
    with open(output_file_path, 'w') as file:
        for item in tqdm(data, desc="Writing data to JSONL"):
            
            pred = [{'prediction_text':item['predicted_answer'], 'id':item['id']}]
            refs = [{'answers':item['answers'], 'id':item['id']}]
            result = squad_metric.compute(predictions=pred, references=refs)
            
            write = item.copy()
            write['performance'] = result
            if ((result['f1'] < 100.0) and (not ('ablations' in file_path))) or ((result['f1'] > 0.0) and ('ablations' in file_path)):        
                json.dump(write, file)
                file.write('\n')


for file_path in os.listdir(path):
    name, extension = os.path.splitext(file_path)
    if (extension == '.jsonl') and ('mistake' not in name):
        with_path = os.path.join(path, file_path)
        file_performance_replace(with_path)


def file_performance_replace(file_path, squad_metric=squad_metric):
    # Reading the .jsonl file
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    
    # Path to your output .jsonl file
    output_file_path = file_path.replace('.jsonl', '_mistakes_bad.jsonl')

    # Delete the file if it exists
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    # Writing to .jsonl file
    with open(output_file_path, 'w') as file:
        for item in tqdm(data, desc="Writing data to JSONL"):
            
            pred = [{'prediction_text':item['predicted_answer'], 'id':item['id']}]
            refs = [{'answers':item['answers'], 'id':item['id']}]
            result = squad_metric.compute(predictions=pred, references=refs)
            
            write = item.copy()
            write['performance'] = result
            if ((result['f1'] == 0.0) and (not ('ablations' in file_path))) or ((result['f1'] > 0.0) and ('ablations' in file_path)):        
                json.dump(write, file)
                file.write('\n')

path = "Results"
for file_path in os.listdir(path):
    name, extension = os.path.splitext(file_path)
    if (extension == '.jsonl') and ('mistake' not in name) and (name=='squad_base_eval_predictions'):
        with_path = os.path.join(path, file_path)
        file_performance_replace(with_path)

path = "Results_Fixed_Change"
for file_path in os.listdir(path):
    name, extension = os.path.splitext(file_path)
    if (extension == '.jsonl') and ('mistake' not in name) and (name=='eval_predictions_sq1.0_as1.0_adv0_hp1.0_'):
        with_path = os.path.join(path, file_path)
        file_performance_replace(with_path)