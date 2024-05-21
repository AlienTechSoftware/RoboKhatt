# Robo Khatt
Where Tradition Meets Technology

![RoboKhatt Logo](./logo.png)


RoboKhatt is an innovative project that leverages advanced diffusion models to generate high-quality Arabic calligraphy. By combining the artistry of traditional calligraphy with the power of artificial intelligence, RoboKhatt aims to provide beautifully rendered text that can be used in various applications.

## Features

- **Text-to-Image Generation**: Convert Arabic text into stunning calligraphic images.
- **Customizable Styles**: Supports various calligraphy styles and fonts.
- **Incremental Learning**: Starts with individual characters and scales up to words and sentences.
- **High-Quality Output**: Utilizes state-of-the-art diffusion models to ensure visually appealing results.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- PIL (Pillow)
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:
```bash
   git clone https://github.com/your_username/RoboKhatt.git
   cd RoboKhatt
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```


### Usage

#### Data Preparation:

Collect a dataset of Arabic text and corresponding images.
Use provided scripts in the `scripts/` directory to preprocess the data.
#### Training:

Train the diffusion model on your dataset using:
```bash
Copy code
python src/model_training.py --data_dir path_to_your_data --output_dir path_to_save_model
```

#### Inference:

Generate calligraphic images from text using:
```bash
Copy code
python src/model_inference.py --input_text "Your Arabic Text Here" --model_dir
```