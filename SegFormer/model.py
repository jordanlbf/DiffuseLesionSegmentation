import torch
from transformers import SegformerForSemanticSegmentation
import torch.nn as nn

def model():
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load SegFormer B0 model
    model_name = "nvidia/segformer-b4-finetuned-ade-512-512"
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)

    # Manually resize the classifier layer to match the number of labels
    model.decode_head.classifier = nn.Conv2d(768, 2, kernel_size=1)  # Adjust input channels to 768 for B2

    # Move model to GPU if available
    model.to(device)
    return model