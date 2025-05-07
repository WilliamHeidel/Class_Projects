import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
import evaluate
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy
import os
import json
from Ablations import *
#from Fixes import *
import torch
from Adversarial_Datasets import *

NUM_PREPROCESSING_WORKERS = 2


def main():
    argp = HfArgumentParser(TrainingArguments)
    # The HfArgumentParser object collects command-line arguments into an object (and provides default values for unspecified arguments).
    # In particular, TrainingArguments has several keys that you'll need/want to specify (when you call run.py from the command line):
    # --do_train
    #     When included, this argument tells the script to train a model.
    #     See docstrings for "--task" and "--dataset" for how the training dataset is selected.
    # --do_eval
    #     When included, this argument tells the script to evaluate the trained/loaded model on the validation split of the selected dataset.
    # --per_device_train_batch_size <int, default=8>
    #     This is the training batch size.
    #     If you're running on GPU, you should try to make this as large as you can without getting CUDA out-of-memory errors.
    #     For reference, with --max_length=128 and the default ELECTRA-small model, a batch size of 32 should fit in 4gb of GPU memory.
    # --num_train_epochs <float, default=3.0>
    #     How many passes to do through the training data.
    # --output_dir <path>
    #     Where to put the trained model checkpoint(s) and any eval predictions.
    #     *This argument is required*.

    argp.add_argument('--model', type=str,
                      default='google/electra-small-discriminator',
                      help="""This argument specifies the base model to fine-tune.
        This should either be a HuggingFace model ID (see https://huggingface.co/models)
        or a path to a saved model checkpoint (a folder containing config.json and pytorch_model.bin).""")
    argp.add_argument('--task', type=str, choices=['nli', 'qa'], required=True,
                      help="""This argument specifies which task to train/evaluate on.
        Pass "nli" for natural language inference or "qa" for question answering.
        By default, "nli" will use the SNLI dataset, and "qa" will use the SQuAD dataset.""")
    argp.add_argument('--dataset', type=str, default=None,
                      help="""This argument overrides the default dataset used for the specified task.""")
    argp.add_argument('--max_length', type=int, default=128,
                      help="""This argument limits the maximum sequence length used during training/evaluation.
        Shorter sequence lengths need less memory and computation time, but some examples may end up getting truncated.""")
    argp.add_argument('--max_train_samples', type=int, default=None,
                      help='Limit the number of examples to train on.')
    argp.add_argument('--max_eval_samples', type=int, default=None,
                      help='Limit the number of examples to evaluate on.')
    
    argp.add_argument('--ablation', type=str, default="",
                      help='Determine which Ablation function to use.')
    argp.add_argument('--adversarial', type=float, default=None,
                      help='Determine which Ablation function to use.')
    argp.add_argument('--context_retrieval', action='store_true',
                        help='Incorporates a context retrieval model.')
    argp.add_argument('--entity_ranking', action='store_true',
                        help='Incorporates an entity ranking model.')

    argp.add_argument('--perc_squad', type=float, default=1.0,
                      help='Percentage of the SQUAD dataset used in Training.')
    argp.add_argument('--perc_adversarial_squad', type=float, default=0,
                      help='Percentage of the SQUAD Adversarial dataset used in Training.')
    argp.add_argument('--adversarial_qa_subset', type=str, default='adversarialQA',
                      help="Subset of the Adversarial QA dataset used in Training ('adversarialQA', 'dbert', 'dbidaf', 'droberta').")
    argp.add_argument('--perc_adversarial_qa', type=float, default=0,
                      help='Percentage of the Adversarial QA dataset used in Training.')
    argp.add_argument('--hotpot_distractor', action='store_true',
                      help="If set to True, it will use the HotPotQA 'Distractor' dataset instead of the 'FullWiki' dataset.")
    argp.add_argument('--perc_hotpotqa', type=float, default=0,
                      help='Percentage of the HotPot QA dataset used in Training.')

    argp.add_argument('--eval_tough_examples', action='store_true',
                    help='Runs an additional evaluation round on the toughest examples.')


    training_args, args = argp.parse_args_into_dataclasses()
    pipeline = False
    if args.context_retrieval or args.entity_ranking:
        pipeline = True
    adversarial_datasets = False
    if (args.perc_squad > 0) or (args.perc_adversarial_squad > 0) or (args.perc_adversarial_qa) or (args.perc_hotpotqa):
        adversarial_datasets = True
        filename = f"_sq{args.perc_squad }_as{args.perc_adversarial_squad}_adv{args.perc_adversarial_qa}_hp{args.perc_hotpotqa}"
        if args.adversarial_qa_subset != 'adversarialQA':
            filename += f"_{args.adversarial_qa_subset}"
        if args.hotpot_distractor:
            filename += "_distractor"
    else:
        filename = ''

    # Dataset selection
    # IMPORTANT: this code path allows you to load custom datasets different from the standard SQuAD or SNLI ones.
    # You need to format the dataset appropriately. For SNLI, you can prepare a file with each line containing one
    # example as follows:
    # {"premise": "Two women are embracing.", "hypothesis": "The sisters are hugging.", "label": 1}
    if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
        dataset_id = None
        # Load from local json/jsonl file
        dataset = datasets.load_dataset('json', data_files=args.dataset)
        # By default, the "json" dataset loader places all examples in the train split,
        # so if we want to use a jsonl file for evaluation we need to get the "train" split
        # from the loaded dataset
        eval_split = 'train'
    else:
        default_datasets = {'qa': ('squad',), 'nli': ('snli',)}
        dataset_id = tuple(args.dataset.split(':')) if args.dataset is not None else \
            default_datasets[args.task]
        # MNLI has two validation splits (one with matched domains and one with mismatched domains). Most datasets just have one "validation" split
        eval_split = 'validation_matched' if dataset_id == ('glue', 'mnli') else 'validation'
        # Load the raw data
        dataset = datasets.load_dataset(*dataset_id)
    
    # NLI models need to have the output label count specified (label 0 is "entailed", 1 is "neutral", and 2 is "contradiction")
    task_kwargs = {'num_labels': 3} if args.task == 'nli' else {}

    # Here we select the right model fine-tuning head
    model_classes = {'qa': AutoModelForQuestionAnswering,
                     'nli': AutoModelForSequenceClassification}
    model_class = model_classes[args.task]
    # Initialize the model and tokenizer from the specified pretrained model/checkpoint
    model = model_class.from_pretrained(args.model, **task_kwargs)
    # Make tensor contiguous if needed https://github.com/huggingface/transformers/issues/28293
    if hasattr(model, 'electra'):
        for param in model.electra.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Select the dataset preprocessing function (these functions are defined in helpers.py)
    if args.task == 'qa':
        prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer)
        prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer)
    elif args.task == 'nli':
        prepare_train_dataset = prepare_eval_dataset = \
            lambda exs: prepare_dataset_nli(exs, tokenizer, args.max_length)
        # prepare_eval_dataset = prepare_dataset_nli
    else:
        raise ValueError('Unrecognized task name: {}'.format(args.task))

    print("Preprocessing data... (this takes a little bit, should only happen once per dataset)")
    if dataset_id == ('snli',):
        # remove SNLI examples with no label
        dataset = dataset.filter(lambda ex: ex['label'] != -1)
    
    train_dataset = None
    eval_dataset = None
    train_dataset_featurized = None
    eval_dataset_featurized = None
    if training_args.do_train:
        train_dataset = dataset['train']
        if args.adversarial:
            if (args.adversarial >= 0.0) and (args.adversarial <= 1.0):
                dynamic = False
            else:
                dynamic = True
            train_dataset = generate_adversarial_dataset(train_dataset, args.adversarial, dynamic=dynamic)
        if adversarial_datasets:
            train_dataset = get_combined_datasets(train_dataset, args.perc_squad, args.perc_adversarial_squad, args.adversarial_qa_subset, args.perc_adversarial_qa, args.hotpot_distractor, args.perc_hotpotqa)
        if args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        train_dataset_featurized = train_dataset.map(
            prepare_train_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=train_dataset.column_names
        )
    if training_args.do_eval:
        eval_dataset = dataset[eval_split]
        if args.ablation != "":
            ablation_function = ablation_functions[args.ablation]
            if args.ablation == 'randomized_context_ablation':
                contexts = eval_dataset['context']
                eval_dataset = eval_dataset.map(lambda x: ablation_function(x, contexts))
            else:
                eval_dataset = eval_dataset.map(ablation_function)
                eval_dataset = eval_dataset.filter(lambda x: x is not None)
        # Assuming `validation_data` contains your SQuAD validation dataset
        if args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        eval_dataset_featurized = eval_dataset.map(
            prepare_eval_dataset,
            batched=True,
            num_proc=NUM_PREPROCESSING_WORKERS,
            remove_columns=eval_dataset.column_names
        )

    # Select the training configuration
    trainer_class = Trainer
    eval_kwargs = {}
    # If you want to use custom metrics, you should define your own "compute_metrics" function.
    # For an example of a valid compute_metrics function, see compute_accuracy in helpers.py.
    compute_metrics = None
    if args.task == 'qa':
        # For QA, we need to use a tweaked version of the Trainer (defined in helpers.py)
        # to enable the question-answering specific evaluation metrics
        trainer_class = QuestionAnsweringTrainer
        eval_kwargs['eval_examples'] = eval_dataset
        metric = evaluate.load('squad')   # datasets.load_metric() deprecated
        compute_metrics = lambda eval_preds: metric.compute(
            predictions=eval_preds.predictions, references=eval_preds.label_ids)
    elif args.task == 'nli':
        compute_metrics = compute_accuracy
    if pipeline:
        trainer_class = BatchPipelineQuestionAnsweringTrainer
        eval_examples = [ex for ex in eval_dataset]
        

    # This function wraps the compute_metrics function, storing the model's predictions
    # so that they can be dumped along with the computed metrics
    eval_predictions = None
    def compute_metrics_and_store_predictions(eval_preds):
        nonlocal eval_predictions
        eval_predictions = eval_preds
        return compute_metrics(eval_preds)

    # Initialize the Trainer object with the specified arguments and the model and dataset we loaded above
    if not pipeline:
        print("\nBeginning Pipeline!\n")
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset_featurized,
            eval_dataset=eval_dataset_featurized,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics_and_store_predictions
        )
        print("\nBeginning Evaluation!\n")
    else:
        trainer = trainer_class(
            model=model.to(device),
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_examples=eval_examples,
            pipeline=qa_pipeline_batch,
            use_retrieval=args.context_retrieval,
            use_ranking=args.entity_ranking,
            compute_metrics=compute_metrics_and_store_predictions
        )
    # Train and/or evaluate
    if training_args.do_train:
        trainer.train()
        trainer.save_model()
        # If you want to customize the way the loss is computed, you should subclass Trainer and override the "compute_loss"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.compute_loss).
        #
        # You can also add training hooks using Trainer.add_callback:
        #   See https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer.add_callback
        #   and https://huggingface.co/transformers/main_classes/callback.html#transformers.TrainerCallback

    if training_args.do_eval:
        results = trainer.evaluate(**eval_kwargs)

        # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
        #
        # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
        # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
        # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
        # values that it returns.

        print('Evaluation results:')
        print(results)
        
        if training_args.output_dir != "None":

            os.makedirs(training_args.output_dir, exist_ok=True)
    
            with open(os.path.join(training_args.output_dir, f'eval_metrics{filename}.json'.replace('.json', f'_{args.ablation}.json')), encoding='utf-8', mode='w') as f:
                json.dump(results, f)

            with open(os.path.join(training_args.output_dir, f'eval_predictions{filename}.jsonl'.replace('.jsonl', f'_{args.ablation}.jsonl')), encoding='utf-8', mode='w') as f:
                if args.task == 'qa':
                    predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                    for example in eval_dataset:
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')
                else:
                    for i, example in enumerate(eval_dataset):
                        example_with_prediction = dict(example)
                        example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                        example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                        f.write(json.dumps(example_with_prediction))
                        f.write('\n')

    if (training_args.do_eval) and (args.eval_tough_examples):

        for txtfile, txtfilename in zip(["hard_eval_examples_ids.txt","hard_eval_examples_ids_entities.txt"],["tough","tough_entities"]):
            with open(txtfile, "r") as f:
                ids = [line.strip() for line in f]

            filtered_dataset = eval_dataset.filter(lambda example: example["id"] in ids)
            eval_dataset_featurized_filtered = filtered_dataset.map(
                prepare_eval_dataset,
                batched=True,
                num_proc=NUM_PREPROCESSING_WORKERS,
                remove_columns=filtered_dataset.column_names
            )

            eval_kwargs['eval_dataset'] = eval_dataset_featurized_filtered
            eval_kwargs['eval_examples'] = filtered_dataset

            results = trainer.evaluate(**eval_kwargs)

            # To add custom metrics, you should replace the "compute_metrics" function (see comments above).
            #
            # If you want to change how predictions are computed, you should subclass Trainer and override the "prediction_step"
            # method (see https://huggingface.co/transformers/_modules/transformers/trainer.html#Trainer.prediction_step).
            # If you do this your custom prediction_step should probably start by calling super().prediction_step and modifying the
            # values that it returns.

            print('Evaluation results:')
            print(results)
            
            if training_args.output_dir != "None":
                output_dir = "Results_Fixed_Test"
                os.makedirs(output_dir, exist_ok=True)

                with open(os.path.join(output_dir, f'eval_metrics{filename}.json'.replace('.json', f'_{args.ablation}_{txtfilename}.json')), encoding='utf-8', mode='w') as f:
                    json.dump(results, f)

                with open(os.path.join(output_dir, f'eval_predictions{filename}.jsonl'.replace('.jsonl', f'_{args.ablation}_{txtfilename}.jsonl')), encoding='utf-8', mode='w') as f:
                    if args.task == 'qa':
                        predictions_by_id = {pred['id']: pred['prediction_text'] for pred in eval_predictions.predictions}
                        for example in filtered_dataset:
                            example_with_prediction = dict(example)
                            example_with_prediction['predicted_answer'] = predictions_by_id[example['id']]
                            f.write(json.dumps(example_with_prediction))
                            f.write('\n')
                    else:
                        for i, example in enumerate(filtered_dataset):
                            example_with_prediction = dict(example)
                            example_with_prediction['predicted_scores'] = eval_predictions.predictions[i].tolist()
                            example_with_prediction['predicted_label'] = int(eval_predictions.predictions[i].argmax())
                            f.write(json.dumps(example_with_prediction))
                            f.write('\n')

        eval_metrics_path = f'eval_metrics{filename}.json'.replace('.json', f'_{args.ablation}.json')
        eval_predictions_path = f'eval_predictions{filename}.jsonl'.replace('.jsonl', f'_{args.ablation}.jsonl')
        import shutil
        for file in [eval_metrics_path, eval_predictions_path]:
            shutil.move(os.path.join(training_args.output_dir, file), os.path.join(output_dir, file))

if __name__ == "__main__":
    main()
