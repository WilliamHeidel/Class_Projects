
import datasets
import math
import random

selected_columns = ['id', 'title', 'context', 'question', 'answers']


### HotPotQA

def hotpotqa_to_squad(example):
    context_title = example['context']['title']
    title = ", ".join(context_title)
    example['title'] = title

    context_sentences = example['context']['sentences']
    sentences = " ".join(["".join(sentence) for sentence in context_sentences])
    example['context'] = sentences

    answer_text = example['answer']
    answer_start = sentences.find(answer_text)
    example['answers'] = {'text':[answer_text], 'answer_start':[answer_start]}

    return example

def get_hotpotqa(distractor=False):
    if distractor:
        mode = 'distractor'
    else:
        mode = 'fullwiki'
    hpqa = datasets.load_dataset('hotpot_qa',mode,trust_remote_code=True)
    hpqa_train = hpqa['train']

    hpqa_mapped = hpqa_train.map(hotpotqa_to_squad)
    hpqa_mapped = hpqa_mapped.cast_column(
        "answers",
        datasets.Sequence(
            feature={
                "text": datasets.Value(dtype="string", id=None),
                "answer_start": datasets.Value(dtype="int32", id=None),
            }
        ),
    )
    hpqa_mapped = hpqa_mapped.filter(lambda example: example['answers']['answer_start'][0] != -1)
    hpqa_mapped = hpqa_mapped.select_columns(selected_columns)

    return hpqa_mapped


def upsample_dataset(dataset, factor):
    """
    Upsample a dataset by duplicating its examples.

    :param dataset: A Hugging Face Dataset object.
    :param factor: The factor by which to upsample (e.g., 2 = double the size).
    :return: An upsampled dataset.
    """
    # Create a list of indices to duplicate examples
    indices = list(range(len(dataset))) * math.ceil(factor)

    random.seed(42)
    random.shuffle(indices)

    selected_indices = indices[:math.ceil(factor*len(dataset))]

    # Use select() to duplicate examples
    return dataset.select(selected_indices)


### Combined Datasets

def get_combined_datasets(train_dataset, perc_squad, perc_adversarial_squad, adversarial_qa_subset, perc_adversarial_qa, hotpot_distractor, perc_hotpotqa):
    datasets_list = []

    if perc_squad <= 0:
        pass
    else:
        train_dataset = upsample_dataset(train_dataset, perc_squad)
        datasets_list.append(train_dataset)

    if perc_adversarial_squad <= 0:
        pass
    else:
        adversarial_dataset_squad = datasets.load_dataset('stanfordnlp/squad_adversarial', 'AddSent', trust_remote_code=True)
        adversarial_dataset_squad_eval = adversarial_dataset_squad['validation']
        adversarial_dataset_squad_eval = upsample_dataset(adversarial_dataset_squad_eval, perc_adversarial_squad)
        datasets_list.append(adversarial_dataset_squad_eval)

    if perc_adversarial_qa <= 0:
        pass
    else:
        adversarial_qa_train = datasets.load_dataset("adversarial_qa", adversarial_qa_subset, split="train")
        adversarial_qa_train = adversarial_qa_train.select_columns(selected_columns)
        adversarial_qa_train = upsample_dataset(adversarial_qa_train, perc_adversarial_qa)
        datasets_list.append(adversarial_qa_train)

    if perc_hotpotqa <= 0:
        pass
    else:
        hpqa = get_hotpotqa(distractor=hotpot_distractor)
        hpqa = upsample_dataset(hpqa, perc_hotpotqa)
        datasets_list.append(hpqa)

    combined_dataset = datasets.concatenate_datasets(datasets_list)
    return combined_dataset
