import torch
from torch.utils.data import Dataset
from transformers import BatchEncoding
import os
import re
import numpy as np


def clean_whitespace(text):
    """
    Cleans up whitespace in a given string by removing leading and trailing spaces 
    and condensing multiple spaces into a single space.

    This function first trims any whitespace at the beginning and end of the input text. 
    It then identifies any occurrences of multiple consecutive whitespace characters 
    within the text and replaces them with a single space. This process ensures that 
    the text is neatly formatted without unnecessary or excessive spacing.

    Parameters:
    text (str): The input string to be cleaned.

    Returns:
    str: The cleaned text with normalized whitespace.
    """

    cleaned_text = text.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    return cleaned_text


def adjust_capitalization(text):
    """
    Adjusts the capitalization of each word in the text.

    This function capitalizes the first letter of each word if the word is in 
    uppercase. e.g.: "HELLO". Otherwise, it leaves the word as it is. 
    
    Parameters:
    text (str): The input text whose words' capitalization needs adjustment.

    Returns:
    str: The text with adjusted capitalization.

    Note:
    - This function does not affect the case of words that are not fully in uppercase.
    """

    def capitalize_first_letter(word):
        return word.capitalize() if word.isupper() else word

    return ' '.join([capitalize_first_letter(word) for word in text.split()])


def remove_punctuation(text, classes):
    """
    Removes punctuation marks at the end of each word in the text.

    This function iterates through each word in the input text and removes any 
    punctuation mark at the end of the word if it belongs to the specified classes.

    Parameters:
    text (str): The input text from which punctuation is to be removed.
    classes (list): A list of punctuation marks to be removed.

    Returns:
    str: The text with specified punctuation marks removed from the end of words.

    Note:
    - This function only removes punctuation marks that are at the end of words.
    """

    return ' '.join([word[:-1] if word[-1] in classes else word for word in text.split()])


def remove_non_ascii(text):
    """
    Removes all non-ASCII characters from the text.

    This function discards any characters in the text that are not part of the 
    standard ASCII character set.

    Parameters:
    text (str): The input text from which non-ASCII characters are to be removed.

    Returns:
    str: The text with all non-ASCII characters removed.

    Note:
    - This is useful for standardizing texts that contain characters from various 
      encodings or special symbols.
    """

    return re.sub(r'[^\x00-\x7F]+', '', text)


def remove_quotations(text):
    """
    Removes all quotation marks from the text.

    Parameters:
    text (str): The input text from which quotation marks are to be removed.

    Returns:
    str: The text with all quotation marks removed.

    Note:
    - The function only targets double quotation marks.
    """
    return text.replace('"', '')  # remove all occurences when one of the punctuation sign in classses preceeds a word


def preprocess_text(text):
    """
    Performs a series of preprocessing steps on the text.

    This function sequentially applies multiple text preprocessing functions including 
    removing non-ASCII characters, normalizing whitespaces, adjusting capitalization, 
    and removing quotations from the input text.

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text after applying all specified modifications.
    """

    return remove_quotations(adjust_capitalization(clean_whitespace(remove_non_ascii(text))))


class DatasetTextProcessor(Dataset):

    def __init__(self, data_source, classes, tokenizer, mapping=None, is_directory=False, mapping_tokens=True,
                 return_overflowing_tokens=False):
        """
        Initializes the dataset processor.

        :param data_source: Directory path or list of text data.
        :param classes: List of classes for labeling.
        :param tokenizer: Tokenizer to be used for processing text.
        :param is_directory: Flag to indicate if data_source is a directory.
        :param mapping_tokens: If set to false, each subword token derived from a word is assigned the same label as the original word. 
                               For example, if 'Unbelievable' is labeled as capitalized and followed by a Period, all its subword tokens receive the same label. If true, the word-label pairs are mapped to token-label pairs with specific rules:
          - For a word split into multiple subword tokens:
            For capitalization:
              - Only the first subword token gets the capitalized label
            - For punctuation
              - Only the last subword token gets the punctuation label
            - Labels for all other subword tokens are set to zero.
          Example:
          Word: "Unbelievable"              ----   Word Label: [Capitalized, Period]
          
          Tokens: ["Un", "believ", "able"]  ----   Token Label:
                                                    if mapping_tokens is false: [Capitalized, Period]
                                                                                [Capitalized, Period]
                                                                                [Capitalized, Period]
                                                    if  mapping_tokens is true: [Capitalized, 0]
                                                                                [0, 0]
                                                                                [0, Period]
            
        """

        self.tokenizer = tokenizer
        self.classes = classes
        self.is_directory = is_directory
        self.mapping_tokens = mapping_tokens
        self.wordids_labels = None
        self.return_overflowing_tokens = return_overflowing_tokens
        if mapping is None:
            self.mapping = {c: (i + 1) for i, c in enumerate(classes)}
        else:
            self.mapping = mapping

        if is_directory:
            self.files_list = [os.path.join(data_source, f) for f in os.listdir(data_source)]

        else:
            self.data = data_source

        self.word_ids = None
        self.words_labels = None

    def __len__(self):
        return len(self.files_list if self.is_directory else self.data)

    def create_encoding_wordids_labels(self, words_labels):
        """
        Creates token encodings and labels for capitalization and punctuation.

        :param words_labels: List of words with their labels.
        """
        encoding = self.tokenizer(' '.join([word for (word, _, _) in words_labels]),
                                  add_special_tokens=True,
                                  return_overflowing_tokens=self.return_overflowing_tokens,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        if self.tokenizer.name_or_path == 'microsoft/deberta-v3-base':
            word_ids = self.get_word_ids_deberta(encoding)
        elif self.tokenizer.name_or_path == 'roberta-base':
            word_ids = self.get_word_ids_roberta(encoding)

        wordids_labels = [(word_id, *words_labels[word_id][1:]) if word_id is not None else (np.nan, 0, 0) for word_id
                          in word_ids]
        wordids_labels = torch.tensor(wordids_labels).reshape(-1, self.tokenizer.model_max_length, 3)

        if self.mapping_tokens:
            wordids_labels = self._mask_labels(wordids_labels)

        # encoding, word_ids, labels 
        return encoding, wordids_labels[:, :, 0], wordids_labels[:, :, 1:].long()
        # return encoding, wordids_labels

    def _mask_labels(self, wordids_labels):
        # create a mask for all the token labels which should be set to zero 
        mask_cap = torch.cat((torch.zeros(wordids_labels.shape[0], 1, dtype=torch.bool),
                              wordids_labels[:, 1:, 0] == wordids_labels[:, :-1, 0]), dim=1)
        mask_punc = torch.cat((wordids_labels[:, 1:, 0] == wordids_labels[:, :-1, 0],
                               torch.ones(wordids_labels.shape[0], 1, dtype=torch.bool)), dim=1)

        wordids_labels[:, :, 1][mask_cap] = 0
        wordids_labels[:, :, 2][mask_punc] = 0

        return wordids_labels

    def __getitem__(self, idx):
        """
        Retrieves and processes an item from the dataset.

        :param idx: Index of the item in the dataset.
        """
        if self.is_directory:  # if data source is a directory
            file_path = self.files_list[idx]
            with open(file_path, 'r') as f:
                data = f.read()
            lines = data.split('\n')
            words_labels = [(line.split('\t')[0], 0, self.mapping[line.split('\t')[1]]) for line in lines if line]

        else:  # if data is a list of raw text
            text = self.data[idx]
            text = preprocess_text(text)
            # labels punctuation signs
            words_labels = [(word[:-1], self.mapping[word[-1]]) if word[-1] in self.classes else (word, 0) for word in
                            text.split()]
            # labels capitalization
            words_labels = [(word.lower(), 1, label_cap) if word.istitle() else (word, 0, label_cap) for
                            (word, label_cap) in words_labels]

            self.words_labels = words_labels

        return self.create_encoding_wordids_labels(words_labels)

    def get_word_ids_deberta(self, encoding):
        """
        For each token it returns the corresponding word ID in the original text for the deberta model
        """
        word_ids = [w_id for i in range(len(encoding['input_ids'])) for w_id in encoding[i].word_ids]
        return word_ids

    def get_word_ids_roberta(self, encoding):
        """    
        For each token it returns the corresponding word ID in the original text for the roberta model
        """
        word_ids = []
        i = 0

        for inp_ids in encoding['input_ids']:

            tokens = self.tokenizer.convert_ids_to_tokens(inp_ids)

            for token in tokens:
                if token in self.tokenizer.all_special_tokens:
                    word_ids.append(None)
                elif token[0] == 'Ġ':
                    i += 1
                    word_ids.append(i)
                else:
                    word_ids.append(i)

        return word_ids


def custom_collate_fn(batch):
    """
    A custom collate function for DataLoader that processes batches of data.

    This function takes a batch of data and separates it into its constituent components. 
    It stacks the individual elements of each component to create a unified batch for 
    each type of data. Specifically, it processes and stacks encoding data, word IDs, 
    and word ID labels from the batch.

    Parameters:
    batch (list of tuples): A batch of data where each tuple contains three elements - 
                            encodings, word IDs, and word ID labels. Each element in 
                            the tuple is expected to be a tensor.

    Returns:
    tuple: A tuple containing three elements:
           - BatchEncoding: An object containing stacked 'input_ids' and 'attention_mask' 
                            from the encodings.
           - Tensor: A stacked tensor of word IDs.
           - Tensor: A stacked tensor of word ID labels.
    """
    encodings, wordids, wordids_labels = zip(*batch)

    stacked_wordids = torch.vstack(wordids)
    stacked_labels = torch.vstack(wordids_labels)

    encoding_stacked = {}
    for key in ['input_ids', 'attention_mask']:
        encoding_stacked[key] = torch.vstack([encoding[key] for encoding in encodings])

    return BatchEncoding(encoding_stacked), stacked_wordids, stacked_labels
