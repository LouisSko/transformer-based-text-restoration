import torch.nn as nn
import torch 
import os
import logging
from transformers import BertModel, RobertaModel, ElectraModel, DebertaV2Model,  AutoTokenizer


class Model(nn.Module):
    """
    A generic neural network model class for NLP tasks using transformer architectures.

    This class can be initialized with different transformer models, specifically 
    designed to handle NLP tasks. It dynamically loads the model and tokenizer based 
    on the specified pre-trained transformer model. Additionally, it includes a 
    classification head for task-specific predictions.

    Attributes:
    args (Namespace): Configuration arguments including model specifications and other 
                      parameters.
    num_classes (int): Number of classes for the classification task. Calculated based 
                       on the number of classes in 'args' plus additional classes for 
                       capitalization prediction.
    tokenizer (AutoTokenizer): Tokenizer corresponding to the transformer model.
    model (nn.Module): The transformer model loaded from the pre-trained model.
    fc (nn.Linear): A fully connected linear layer for classification.

    Methods:
    forward: Defines the forward pass of the model.

    Note:
    - Supported models currently include variants of 'roberta' and 'deberta'.
    - The class expects 'pretrained_model' and 'classes' to be provided in 'args'.
    - The 'forward' method takes 'input_ids' and 'attention_mask' as inputs and 
      returns the logits from the model's classification head.
    """
    def __init__(self, args):
        
        super().__init__()
        self.args = args
        self.num_classes = len(self.args.classes) + 1 + 2  # +1 (for predicting capitalization) + 2 (for predicting no cap and no punc)

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model, use_auth_token=os.environ.get("HF_TOKEN"))
        # Initialize model and tokenizer based on pretrained_model argument
        if 'roberta' in self.args.pretrained_model:
            self.model = RobertaModel.from_pretrained(self.args.pretrained_model)
        elif 'deberta' in self.args.pretrained_model:
            self.model = DebertaV2Model.from_pretrained(self.args.pretrained_model)
            self.tokenizer.model_max_length = 512  # fixes a bug in the config file
        else:
            raise ValueError("Unsupported model type. Please use a valid model type such as 'roberta' or 'deberta'.")

        # Classification head
        self.fc = nn.Linear(self.model.config.hidden_size, self.num_classes)
        
    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: (batch_size, max_seq_len, hidden_size)
        logits = self.fc(last_hidden_state)  # Shape: (batch_size, max_seq_len, num_classes)

        return logits


def load_weights(model, model_weights, device=None):
    """
    Load the weights into the model.

    Args:
    model (torch.nn.Module): The PyTorch model to which the weights will be loaded.
    model_weights (str): Path to the model weights file.
    device (str, optional): The device to map the model to ('cpu' or 'cuda'). 
                            If None, it uses CUDA if available, otherwise CPU.

    Returns:
    torch.nn.Module: The model with loaded weights.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        if os.path.isfile(model_weights) and model_weights.endswith('.pth'):
            logging.info('Loading weights...')
            checkpoint = torch.load(model_weights, map_location=device)
            model.load_state_dict(checkpoint)
            logging.info('Weights successfully loaded.')
        else:
            logging.warning('Invalid path or file extension for model weights. Please provide a .pth file.')
    except Exception as e:
        logging.error(f'Error loading weights: {e}')
    
    return model
