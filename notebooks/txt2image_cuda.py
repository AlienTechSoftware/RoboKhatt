# -*- coding: utf-8 -*-
# notebooks/txt2img.py

import torch
import matplotlib.pyplot as plt
from src.diffusion.diffusion_model import load_model, evaluate_model
from torchvision import transforms

def visualize_generated_image(image_tensor, text):
    transform = transforms.ToPILImage()
    image = transform(image_tensor.squeeze(0))
    plt.imshow(image)
    plt.title(f'Text: {text}')
    plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the models
    model_cuda = load_model('.generated/trained_model_cuda.pth', device)
    # model_cpu = load_model('.generated/trained_model_cpu.pth', torch.device('cpu'))
    
    # Sample text inputs
    sample_texts = ["تم", "اختبار"]  # Add more texts as needed
    font_name = "arial.ttf"
    font_size = 30
    image_size = (512, 128)
    is_arabic = True

    if not torch.cuda.is_available():
        return

    for text in sample_texts:
        # Generate and visualize images using the CUDA model
        output_cuda = evaluate_model(model_cuda, text, font_name, font_size, image_size, is_arabic, device)
        visualize_generated_image(output_cuda, text)

        # # Generate and visualize images using the CPU model
        # output_cpu = evaluate_model(model_cpu, text, font_name, font_size, image_size, is_arabic, torch.device('cpu'))
        # visualize_generated_image(output_cpu, text)

if __name__ == "__main__":
    main()
