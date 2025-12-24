"""Time Series Classification Models with LLM Integration.

This module provides model architectures for time series classification
that combine various encoder types with Large Language Model (LLM) embeddings.

Classes:
    TS_CLS: Full model with encoder + LLM for classification.
    TS_CLS_NoLLM: Encoder-only model without LLM inference.
    TS_CLS_NoEncoder: LLM-only model using text prompts for few-shot classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

import encoders

class TS_CLS(nn.Module):
    """Time Series Classifier with Encoder and LLM.
    
    Encodes time series data into LLM embedding space, then uses the LLM's
    hidden states for classification. The LLM weights are frozen; only the
    encoder and prediction head are trained.
    
    Attributes:
        llm: Frozen pretrained LLM for generating hidden representations.
        encoder: Trainable encoder that maps time series to LLM embedding dimension.
        pred_head: Linear layer mapping LLM hidden states to class logits.
    """
    
    def __init__(self, encoder, model_path, classes, window_size, device, num_kernels, k=16):
        """Initialize the TS_CLS model.
        
        Args:
            encoder: Type of encoder ('cnn', 'attention', 'mlp', 'resnet', 'sl_encoder').
            model_path: Path to pretrained LLM.
            classes: Number of output classes.
            window_size: Length of input time series.
            device: Device to place the model on.
            num_kernels: Number of convolutional kernels for encoder.
            k: Number of tokens/features to generate from encoder (default: 16).
        """
        super(TS_CLS, self).__init__()
        self.llm = AutoModel.from_pretrained(model_path).to(device)
        self.k = k
        for param in self.llm.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype="auto")
        if encoder == 'cnn':
            self.encoder = encoders.cnn.Encoder()
        elif encoder == 'attention':
            self.encoder = encoders.attention.Encoder()
        elif encoder == 'mlp':
            self.encoder = encoders.mlp.Encoder(window_size)
        elif encoder == 'resnet':
            self.encoder = encoders.resnet.Encoder()
        elif encoder == 'sl_encoder':
            self.encoder = encoders.sl_encoder.Encoder(window_size, num_kernels, self.k)
        else:
            print("ENCODER DNE")
            assert 0
        self.pred_head = nn.Linear(4096, classes)  # LLaMA hidden size = 4096
        self.classes = classes

        # Pre-compute padding token embedding for sequence construction
        tmp = self.tokenizer('[PAD]', return_tensors="pt")['input_ids'].to(device)
        self.pad_embeddings = self.llm.get_input_embeddings()(tmp).squeeze(0)

        # Pre-compute prompt embeddings for classification instruction
        prompt = f"Classify the following data with {int(self.k)} features into {self.classes} classes: "
        inputs = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
        self.input_embeddings = self.llm.get_input_embeddings()(inputs).squeeze(0) 

    def forward(self, ts):
        """Forward pass through encoder and LLM.
        
        Args:
            ts: Input time series tensor of shape [B, T] where B is batch size
                and T is time series length.
                
        Returns:
            Tensor of shape [B, classes] containing class logits.
        """
        batch_size = ts.size(0)
        
        # Encode time series to LLM embedding space
        feature = self.encoder(ts)  # [B, k, 4096]
        
        # Expand cached embeddings to batch size
        pad_embeddings = self.pad_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        input_embeddings = self.input_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, 4096]
        
        # Concatenate: [prompt_embeddings, encoded_features, padding]
        input_embeddings = torch.cat((input_embeddings, feature, pad_embeddings), dim=1).to(torch.float16)
        
        # Pass through LLM and extract last hidden state
        outputs = self.llm(inputs_embeds=input_embeddings, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        cls_hidden = hidden[:, -1, :]  # Use last token's hidden state as CLS
        result = self.pred_head(cls_hidden.float())

        return result

class TS_CLS_NoLLM(nn.Module):
    """Time Series Classifier without LLM Inference.
    
    Uses the encoder to map time series to LLM embedding dimension,
    then directly applies mean pooling and a prediction head without
    passing through the LLM. This is faster but doesn't leverage LLM
    representations.
    
    Attributes:
        encoder: Trainable encoder mapping time series to embedding space.
        pred_head: Linear layer for classification.
    """
    
    def __init__(self, encoder, model_path, classes, window_size, device, num_kernels, k=16):
        """Initialize the TS_CLS_NoLLM model.
        
        Args:
            encoder: Type of encoder ('cnn', 'attention', 'mlp', 'resnet', 'sl_encoder').
            model_path: Path to pretrained LLM (used only for tokenizer/embeddings init).
            classes: Number of output classes.
            window_size: Length of input time series.
            device: Device to place the model on.
            num_kernels: Number of convolutional kernels for encoder.
            k: Number of tokens/features to generate from encoder (default: 16).
        """
        super(TS_CLS_NoLLM, self).__init__()
        self.llm = AutoModel.from_pretrained(model_path).to(device)
        self.k = k
        for param in self.llm.parameters():
            param.requires_grad = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype="auto")
        if encoder == 'cnn':
            self.encoder = encoders.cnn.Encoder()
        elif encoder == 'attention':
            self.encoder = encoders.attention.Encoder()
        elif encoder == 'mlp':
            self.encoder = encoders.mlp.Encoder(window_size)
        elif encoder == 'resnet':
            self.encoder = encoders.resnet.Encoder()
        elif encoder == 'sl_encoder':
            self.encoder = encoders.sl_encoder.Encoder(window_size, num_kernels, self.k)
        else:
            print("ENCODER DNE")
            assert 0
        self.pred_head = nn.Linear(4096, classes)
        self.classes = classes

        inputs = self.tokenizer("", return_tensors="pt")['input_ids'].to(device)  # [1, seq_len]  
        self.input_embeddings = self.llm.get_input_embeddings()(inputs).squeeze(0) 

    def forward(self, ts):
        """Forward pass through encoder with mean pooling.
        
        Args:
            ts: Input time series tensor of shape [B, T].
                
        Returns:
            Tensor of shape [B, classes] containing class logits.
        """
        batch_size = ts.size(0)
        feature = self.encoder(ts)  # [B, k, 4096]

        # Concatenate with input embeddings and apply mean pooling
        inputs = self.input_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, 4096]
        inputs = torch.cat((inputs, feature), dim=1).to(torch.float16)
        inputs = inputs.mean(1)  # Mean pooling across sequence dimension
        result = self.pred_head(inputs.float())

        return result

class TS_CLS_NoEncoder(nn.Module):
    """LLM-only Time Series Classifier using Text Prompts.
    
    Performs few-shot classification by converting time series to text
    and using the LLM to compare with class examples. No encoder is used;
    classification relies entirely on LLM's text understanding capabilities.
    
    Attributes:
        pipeline: Text generation pipeline for LLM inference.
        tokenizer: Tokenizer for the LLM.
    """
    
    def __init__(self, model_path, classes, device: str = None):
        """Initialize the TS_CLS_NoEncoder model.
        
        Args:
            model_path: Path to pretrained LLM.
            classes: Number of output classes.
            device: Device specification (uses auto device mapping).
        """
        super(TS_CLS_NoEncoder, self).__init__()
        self.model_id = model_path
        self.classes = classes
        self.device = device

        # Initialize text generation pipeline with bfloat16 for efficiency
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )

        self.tokenizer = self.pipeline.tokenizer

    def build_class_examples_text(self, class_examples: dict) -> str:
        """Convert class examples to formatted text for the prompt.
        
        Args:
            class_examples: Dictionary mapping class labels to representative
                time series examples.
                
        Returns:
            Formatted string containing all class examples for few-shot learning.
        """
        lines = []
        for idx, (cls_key, series) in enumerate(class_examples.items(), start=1):
            if isinstance(series, (list, tuple)):
                s = ", ".join(str(x) for x in series)
            else:
                s = str(series)
            lines.append(f"**class {idx}**\n- Time Series: [{s}]{{{cls_key}}}")
        return "\n\n".join(lines)

    def forward(self, class_examples: dict, query_series) -> tuple:
        """Classify a query time series using few-shot prompting.
        
        Args:
            class_examples: Dictionary mapping class labels to representative
                time series for few-shot comparison.
            query_series: The time series to classify.
                
        Returns:
            tuple: (prediction, output_text, prompt)
                - prediction: Predicted class number (int) or None if parsing fails.
                - output_text: Raw LLM output text.
                - prompt: The constructed prompt sent to the LLM.
        """
        class_examples_text = self.build_class_examples_text(class_examples)
        q = str(query_series)

        # Construct few-shot classification prompt
        prompt = (
            f"You are an expert judge for classification tasks. There are {len(class_examples)} classes. "
            f"For each class below, a representative example time series is provided. Use these examples to judge similarity. "
            f"{class_examples_text}\n\n"
            f"Given the query time series below, find the most similar class and return a short explanation followed by the predicted class number.\n\n"
            f"Query Time Series: [{q}]\n\n"
            f"Output format (exactly):\nExplanation: <one-line explanation>\nPrediction: <class_number>\nDo not output anything else."
        )

        # Define stopping tokens
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self.pipeline(
            prompt,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.9,
        )

        output_text = outputs[0]["generated_text"]

        # Parse prediction from LLM output using regex
        import re
        match = re.search(r"Prediction:\s*(\d+)", output_text)
        if match:
            prediction = int(match.group(1))
        else:
            prediction = None  # Failed to parse prediction

        return prediction, output_text, prompt