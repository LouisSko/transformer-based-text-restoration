import torch
import streamlit as st
from model import Model, load_weights
from solver import define_parser
from prepare_data import preprocess_text
import os
import numpy as np


def restore_text(model, text, classes):
    with torch.no_grad():
        # clean text
        input_text = preprocess_text(text)

        # remove punctuation signs and lowercase text
        words = [word[:-1].lower() if word[-1] in classes else word.lower() for word in input_text.split()]

        # tokenize
        encoding = model.tokenizer(' '.join([word for word in words]),
                                   add_special_tokens=True,
                                   return_overflowing_tokens=True,
                                   padding='max_length',
                                   truncation=True,
                                   return_tensors='pt')

        # save word ids
        word_ids = [w_id for i in range(len(encoding['input_ids'])) for w_id in encoding[i].word_ids]
        word_ids = [w_id if w_id is not None else np.nan for w_id in word_ids]
        word_ids = torch.tensor(word_ids).reshape(-1, model.tokenizer.model_max_length)

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].bool().to(device)

        # make predictions
        logits = model(input_ids, attention_mask)
        attention_mask = ~torch.isnan(
            word_ids)  # this is an attention mask which also sets special tokens to False
        word_ids = word_ids[attention_mask].detach().to('cpu')
        logits = logits[attention_mask].detach().to('cpu')
        preds = torch.stack((logits[:, :2].argmax(dim=1), logits[:, 2:].argmax(dim=1))).T.detach().cpu()

        # map back to word_level_labels
        unique_words, counts = torch.unique_consecutive(word_ids,
                                                        return_counts=True)  # Find unique words and their counts in the word_ids
        word_starts = torch.cat((torch.tensor([0]), counts.cumsum(dim=0)[
                                                    :-1]))  # Calculate the start index of each word in the sorted array

        word_level_preds = torch.tensor(
            [(preds[start, 0].item(), preds[start + count - 1, 1].item()) for start, count in zip(word_starts, counts)])

        # capitalize
        restored_text = [word.capitalize() if (word_level_preds[i, 0] == 1) else word for (i, word) in enumerate(words)]
        # punctuate
        restored_text = [(word + classes[word_level_preds[i, 1] - 1]) if word_level_preds[i, 1] > 0 else word for
                         (i, word) in enumerate(restored_text)]

    return ' '.join([word for word in restored_text])


@st.cache_resource
def load_model(directory, _args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(_args).to(device)
    model = load_weights(model, os.path.join(directory, _args.load_model))
    return model


def main():
    script_path = os.path.abspath(__file__)
    # Get the directory of the current file
    script_directory = os.path.dirname(script_path)

    parser = define_parser()
    args = parser.parse_args([
        "--classes", ".,:!?",
        "--pretrained_model", 'microsoft/deberta-v3-base',
        "--load_model", "deberta-v3-base/model_deberta-v3-tokens_2_last.pth"
    ])

    # Set page config to wide mode
    st.set_page_config(layout="wide")

    # Load the model using the cached function
    model = load_model(script_directory, args)

    # Streamlit UI
    st.title("Punctuation and Capitalization Restoration")

    # Display the original and restored text side by side
    col1, col2 = st.columns(2)
    with col1:
        input_string = st.text_area("Enter Text:", "bla bla bla", height=400)

    with col2:
        area = st.empty()
        area.text_area("Improved text:", '', height=400, key='empty_text')

    # Restore button
    if st.button("Improve"):
        # Call the punctuation restoration function
        restored_text_value = restore_text(model, input_string, args.classes)

        with col2:
            area.text_area("Improved text:", restored_text_value, height=400, key='text')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
