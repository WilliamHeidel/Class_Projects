
import spacy
import re
import random


nlp = spacy.load("en_core_web_sm")


# 1. Ablation: Mask Question Keywords
"""
Purpose: Test if the model relies on superficial cues from the question (e.g., 'who' or 'when') rather than deeper reasoning.

Experiment
    1. Replace question keywords like 'who,' 'when,' or "where" with neutral tokens (e.g., '[KEYWORD]').
    2. Re-evaluate predictions.

Expected Result
    A significant drop in accuracy would indicate over-reliance on shallow keyword-based patterns rather than deeper entity matching.
"""
def mask_question_keywords(question):
    keywords = ["who", "when", "where", "what", "why", "how"]
    return re.sub(r'\b(?:' + '|'.join(keywords) + r')\b', '[KEYWORD]', question, flags=re.IGNORECASE)

def ablation_mask_question_keywords(example):
    masked_question = mask_question_keywords(example['question'])
    example['question'] = masked_question
    return example


# 2. Answer-Only Ablation
"""
    Goal: Determine if the model’s predictions are overly influenced by the presence of certain answer types or patterns.
    Method: Provide the model with only the answer text (in place of the question) and check whether it predicts the answer correctly.
    Expected Outcome: If the model frequently predicts the answer correctly, it suggests it might be picking up biases or shortcuts in the dataset (e.g., specific answer spans are more likely for certain questions).
"""
# for example in squad_data:
#     answer = example['answers']['text']  # Use answer text only
    # Pass (answer, context) into the model for prediction
def ablation_answer_only(example):
    example['question'] = example['answers']['text'][0] #" ".join(list(set(example['answers']['text'])))
    return example


# 3. Ablation: Limit Context Length
"""
Purpose: Determine if the model relies on context proximity for prediction and struggles with longer contexts.

Experiment
    1. Truncate the context to include only the first N tokens or sentences.
    2. Evaluate if errors increase, especially in cases requiring reasoning over broader context.

Expected Result
    If accuracy drops sharply, it suggests the model relies on specific parts of the context for prediction.
"""
def limit_context_length(example, window=25):
    """
    Limits context length to a specified number of tokens around the correct answer.
    
    Parameters:
        example (dict): A dictionary containing context, answer_start, and answer_text.
        window (int): Number of tokens to include before and after the answer span.
    
    Returns:
        dict: Updated example with truncated context.
    """
    context = example["context"]
    answer_text = example["answers"]["text"][0]  # Assume first answer is correct
    answer_start = example["answers"]["answer_start"][0]  # Start index of the answer

    # Calculate answer end index
    answer_end = answer_start + len(answer_text)

    # Ensure the context includes the answer with a window around it
    start_idx = max(0, answer_start - window)
    end_idx = min(len(context), answer_end + window)

    # Truncate context
    truncated_context = context[start_idx:end_idx]
    return truncated_context

def ablation_limit_context_length(example):
    truncated_context = limit_context_length(example)
    example['context'] = truncated_context
    return example


# 4. Ablation: Compare with Randomized Context Sentence Order.
"""
Purpose: Test if the model relies more on context structure rather than semantic alignment.

Experiment
    1. Randomize the order of sentences in the context.
    2. Measure the model's performance to see if it is resilient to such perturbations.

Expected Result
    A drop in performance indicates that the model heavily relies on context sequence rather than content relevance.
"""
def randomize_context_sentences(context):
    sentences = context.split('. ')
    random.shuffle(sentences)
    return '. '.join(sentences)

def ablation_randomize_context_sentences(example):
    randomized_context = randomize_context_sentences(example['context'])
    example['context'] = randomized_context
    return example


# 5. Token Shuffling Ablation (in Context)
"""
    Goal: Check if the model relies on understanding the syntactic structure of the passage.
    Method: Shuffle the words within the context while maintaining the original vocabulary.
    Expected Outcome: If the model still performs well, it may indicate that the model is not truly relying on syntactic and grammatical cues, which are often essential for true comprehension.
"""
def ablation_context_token_shuffling(example):
    context_tokens = example['context'].replace(". "," ").split()
    answer_words = list(set(" ".join(example["answers"]["text"]).split()))
    tokens_not_answer = [word for word in context_tokens if word not in answer_words]
    tokens_full = tokens_not_answer+answer_words
    random.shuffle(tokens_full)
    context = ' '.join(tokens_full)
    example['context'] = context
    return example


# 6. Ablation: Generate Nonsense Contexts
"""
Purpose: Test if the model relies more on context structure rather than semantic alignment.

Experiment
    1. Create a context of nonsensical words and randomly add the correct answer into it.
    2. Measure the model's performance to see if it is resilient to such perturbations.

Expected Result
    A relatively high performance indicates that the model heavily relies on entity type rather than content relevance.
"""
# Define a list of random words
words = [
    "flibber", "gobble", "zork", "bloop", "snarf", "quibble", "zoink",
    "splunge", "glorp", "trundle", "wiggle", "bork", "zibble", "florp",
    "grumble", "snizzle", "frood", "grizzle", "wobble", "plonk"
]

# Function to generate a random sentence
def generate_sentence(word_list, min_words=3, max_words=10):
    num_words = random.randint(min_words, max_words)
    sentence = " ".join(random.choice(word_list) for _ in range(num_words))
    return sentence.capitalize() + "."

# Function to generate multiple nonsense sentences
def generate_nonsense_text(num_sentences=5, word_list=None, answers=None):
    if word_list is None:
        word_list = words
    sentences = [generate_sentence(word_list) for _ in range(num_sentences)]
    if answers:
        sentences += [answer+'.' for answer in set(answers)]
        random.shuffle(sentences)
    return " ".join(sentences)

def ablation_nonsense_context(example):
    nonsense_text = generate_nonsense_text(num_sentences=10, answers=example["answers"]["text"])
    example['context'] = nonsense_text
    return example


# 7. Ablation: Mask Entity Names in Context
"""
Purpose: Evaluate if the model is overly reliant on entity names without proper contextual understanding.

Experiment
    1. Mask all entities in the context (e.g., replace names like 'Bill Gates' with '[MASK]').
    2. Re-evaluate model predictions to see if accuracy drops significantly.

Expected Result
    If the hypothesis is correct, masking entities will cause performance to degrade significantly, as the model will struggle to distinguish entities.
"""
def substitute_same_type_entities(context, answers, mask=False):
    # Process the context and answer with NER
    context_doc = nlp(context)
    if type(answers) == str:
        answer_docs = [nlp(answers)]
    else:
        answer_docs = [nlp(answer) for answer in answers]

    # Determine the entity type of the answer
    answer_entity_type = None
    for answer_doc in answer_docs:
        for ent in answer_doc.ents:
            answer_entity_type = ent.label_
            answer_text = answer_doc.text
            substitute = '[MASK]' if mask else answer_text
            break  # Use the first recognized entity type
        else:
            continue  # If the inner loop didn't break, continue outer loop
        break  # Exit the outer loop if the inner loop breaks

    # Substitute entities in the context matching the answer's type
    substituted_context = context
    if answer_entity_type:
        for ent in context_doc.ents:
            if (ent.label_ == answer_entity_type) and (ent.text not in answer_text):
                substituted_context = substituted_context.replace(ent.text, substitute)
    return substituted_context

def ablation_mask_entity_names(example):
    substituted_context = substitute_same_type_entities(example['context'], example['answers']['text'], mask=True)
    example['context'] = substituted_context
    return example


# 8. Ablation: Substitute Entity Names in Context
"""
Purpose: Evaluate if the model is overly reliant on entity names without proper contextual understanding.

Experiment
    1. Substitute all entities in the context with the correct answers (e.g., replace names like 'Bill Gates' with 'Correct Answer').
    2. Re-evaluate model predictions to see if accuracy increases significantly.

Expected Result
    If the hypothesis is correct, substituting entities will cause performance to increase significantly, as the model will select the correct entity.
"""
def ablation_substitute_entity_names(example):
    substituted_context = substitute_same_type_entities(example['context'], example['answers']['text'])
    example['context'] = substituted_context
    return example


# 9. Ablation: Add Answers and Question to Context.
"""
Purpose: Evaluate if the model is overly reliant on similarities in words and structure between the context and the question in making its predictions.

Experiment
    1. Add the answer after the question and randomly put it somewhere in the context.
    2. Re-evaluate model predictions to see if accuracy increases significantly.

Expected Result
    If model performance improves, the model is over-emphasizing similarities between the context and the question.
"""
def ablation_add_qa_to_context(example):
    context = example['context']
    sentences = context.split('. ')
    answers_string = example['answers']['text'][0]# " ".join(list(set(example['answers']['text'])))
    question_answer = example['question'] + " " + answers_string
    
    random_index = random.randint(0, len(sentences))
    sentences.insert(random_index, question_answer)
    new_context = ". ".join(sentences)
    example['context'] = new_context
    return example


# Dictionary of Ablation Functions:
ablation_functions = {
    "ablation_mask_question_keywords":ablation_mask_question_keywords,
    "ablation_answer_only":ablation_answer_only,
    "ablation_limit_context_length":ablation_limit_context_length,
    "ablation_randomize_context_sentences":ablation_randomize_context_sentences,
    "ablation_context_token_shuffling":ablation_context_token_shuffling,
    "ablation_nonsense_context":ablation_nonsense_context,
    "ablation_mask_entity_names":ablation_mask_entity_names,
    "ablation_substitute_entity_names":ablation_substitute_entity_names,
    "ablation_add_qa_to_context":ablation_add_qa_to_context
}
