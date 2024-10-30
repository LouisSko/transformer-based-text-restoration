import pickle
import os
import random 
from sklearn.model_selection import train_test_split
import re


def create_data_book(args, save=True):
    """
    Creates training, validation, and test datasets from book data.

    This function reads, preprocesses, and chunks text data from books, 
    then splits it into training, validation, and test sets. 
    The datasets are optionally saved in a specified directory.

    Parameters:
    args (Namespace): An object containing the following attributes:
        - save_data_dir (str): Directory path to save the split datasets.
        - books_dir (str): Directory path where the book data is located.
        - train_size (int): The size of the training dataset.
        - valid_size (int): The size of the validation dataset.
        - test_size (int): The size of the test dataset.
    save (bool, optional): Flag to determine whether to save the datasets to disk. 
                           Defaults to True.

    Returns:
    tuple: A tuple containing three lists:
        - train_text (list): The training dataset.
        - valid_text (list): The validation dataset.
        - test_text (list): The test dataset.

    Note:
    The function sets a random seed of 42 for reproducibility.
    """
    
    if not os.path.exists(args.save_data_dir):
        os.makedirs(args.save_data_dir)

    texts = read_book_data(path=args.books_dir)  # here the text is preprocessed and chunked
    random.seed(42)  
    texts = random.sample(texts, (args.train_size + args.valid_size + args.test_size))

    train_text, test_text = train_test_split(texts, shuffle=True, test_size=args.test_size, random_state=42)
    train_text, valid_text = train_test_split(train_text, shuffle=False, test_size=args.valid_size, random_state=42)

    if save:
        with open(os.path.join(args.save_data_dir, 'train_data.pkl'), 'wb') as file:
            pickle.dump(train_text, file)
        with open(os.path.join(args.save_data_dir, 'valid_data.pkl'), 'wb') as file:
            pickle.dump(valid_text, file)
        with open(os.path.join(args.save_data_dir, 'test_data.pkl'), 'wb') as file:
            pickle.dump(test_text, file)
            
    return train_text, valid_text, test_text


def load_text_data(args):
    """
    Loads text data from pickle files into training, validation, and test datasets.

    This function reads pre-saved datasets from pickle files located in a specified 
    directory and loads them into memory. It returns the training, validation, 
    and test datasets as separate lists.

    Parameters:
    args (Namespace): An object containing the following attribute:
        - save_data_dir (str): Directory path where the datasets are saved.

    Returns:
    tuple: A tuple containing three lists:
        - train_text (list): The training dataset loaded from 'train_data.pkl'.
        - valid_text (list): The validation dataset loaded from 'valid_data.pkl'.
        - test_text (list): The test dataset loaded from 'test_data.pkl'.

    Note:
    The function assumes that the datasets are saved in pickle format and 
    requires the 'save_data_dir' attribute in the 'args' object to specify 
    the directory where the datasets are located.
    """
    
    with open(os.path.join(args.save_data_dir, 'train_data.pkl'), 'rb') as file:
        train_text = pickle.load(file)
    with open(os.path.join(args.save_data_dir, 'valid_data.pkl'), 'rb') as file:
        valid_text = pickle.load(file)
    with open(os.path.join(args.save_data_dir, 'test_data.pkl'), 'rb') as file:
        test_text = pickle.load(file)

    return train_text, valid_text, test_text


def generate_chunks(text, max_words_per_chunk):
    """
    Splits a given text into chunks based on specified punctuation marks.

    This function divides the input text into chunks at points where it encounters 
    any of the following punctuation marks: period (.), exclamation mark (!), or 
    question mark (?). The chunks are further merged to ensure that each chunk 
    contains a word count up to a specified maximum. 

    Parameters:
    text (str): The input text to be split into chunks.
    max_words_per_chunk (int): The maximum number of words allowed in each chunk.

    Returns:
    list of str: A list where each element is a chunk of the input text. The chunks 
                 are formed based on punctuation and the specified maximum word count.

    Note:
    - The function ensures that each chunk ends with a complete sentence and adheres 
      to the specified word count limit.
    - Chunks are formed by splitting the text at periods, exclamation marks, and 
      question marks, then merging these smaller pieces into larger chunks as per 
      the word count limit.
    """
    
    chunks = re.split(r'(?<=[.!?])\s', text)
    
    merged_chunks = []
    current_chunk = ""

    for chunk in chunks:
        if len(current_chunk.split()) + len(chunk.split()) <= max_words_per_chunk:
            current_chunk += (" " + chunk)
        else:
            merged_chunks.append(current_chunk)
            current_chunk = chunk

    if current_chunk:
        merged_chunks.append(current_chunk)

    return merged_chunks


def preprocess_book_data(text, max_words_per_chunk):
    """
    Preprocesses book text data and splits it into manageable chunks.

    This function performs basic preprocessing on the input text by replacing 
    tab characters, carriage returns, and newline characters with spaces. 
    It then divides the text into chunks using the 'generate_chunks' function, 
    based on specified punctuation marks and a maximum word count per chunk. 
    The first and last chunks are discarded from the output to ensure the 
    processed text starts and ends with complete, meaningful content.

    Parameters:
    text (str): The raw text data from a book or similar source.
    max_words_per_chunk (int): The maximum number of words that each chunk can contain.

    Returns:
    list of str: A list of text chunks, where each chunk is a processed 
                 segment of the input text, adhering to the specified word count limit.

    Note:
    - This function is designed to preprocess text specifically from books or 
      similar textual sources.
    - The removal of the first and last chunks aims to discard potentially 
      incomplete sentences at the beginning and end of the text.
    """
    
    text = text.replace('\t', ' ').replace('\r', ' ').replace('\n', ' ')  # .replace("\'s", "'s")

    list_of_texts = generate_chunks(text, max_words_per_chunk)

    return list_of_texts[1:-1]  # don't return the first a


def read_book_data(path='Gutenberg1/txt'):
    """
    Reads and preprocesses text data from books stored in a specified directory.

    This function iterates over all text files in a given directory, reading 
    each file's contents. It preprocesses the text data using the 
    'preprocess_book_data' function, which includes splitting the text into 
    manageable chunks. The function accumulates these chunks from all books 
    into a single list, which it then returns.

    Parameters:
    path (str, optional): The directory path where the book text files are stored. 
                          Defaults to 'Gutenberg1/txt'.

    Returns:
    list of str: A list of preprocessed text chunks from all the books in the 
                 specified directory.

    Note:
    - The function only processes files with a '.txt' extension.
    - It handles UTF-8 encoded files and will print an error message if it encounters 
      a file not encoded in UTF-8.
    - The function uses 'preprocess_book_data' to preprocess and chunk the text, with 
      a specified maximum word count per chunk.
    """
    
    books = os.listdir(path)
    books = [book for book in books if book.split('.')[1] == 'txt']

    texts = []
    
    for book in books:
    
        try:
            with open(os.path.join(path, book), 'r', encoding='utf-8') as file:
                text = file.read()

            preprocessed_text = preprocess_book_data(text, 400)
        
            texts += preprocessed_text
            
        except UnicodeDecodeError:
            print("Error: The file is not UTF-8 encoded.")
        
    return texts
    
 