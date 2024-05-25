# -*- coding: utf-8 -*-
# .\tests\test_img_utilities.py

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

def render_text_image(text, image_size, font_name, font_size, position, is_arabic=False):
    """
    Generates an image with the specified text, size, font, and alignment options.

    :param text: The text to be rendered on the image.
    :param image_size: A tuple (width, height) specifying the size of the image.
    :param font_name: The name of the font to be used.
    :param font_size: The size of the font to be used.
    :param position: The position to align the text. Options: 'top-left', 'top-center', 'top-right',
                     'center-left', 'center', 'center-right', 'bottom-left', 'bottom-center', 'bottom-right'.
    :param is_arabic: Boolean indicating whether the text is Arabic.
    :return: An Image object with the rendered text.
    """
    # Create a blank image with white background
    img = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(img)

    # Load the font
    font = ImageFont.truetype(font_name, font_size)

    # Handle Arabic text
    if is_arabic:
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
    else:
        bidi_text = text

    # Get the bounding box of the text
    text_bbox = draw.textbbox((0, 0), bidi_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position
    positions = {
        'top-left': (0, 0),
        'top-center': ((image_size[0] - text_width) // 2, 0),
        'top-right': (image_size[0] - text_width, 0),
        'center-left': (0, (image_size[1] - text_height) // 2),
        'center': ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2),
        'center-right': (image_size[0] - text_width, (image_size[1] - text_height) // 2),
        'bottom-left': (0, image_size[1] - text_height),
        'bottom-center': ((image_size[0] - text_width) // 2, image_size[1] - text_height),
        'bottom-right': (image_size[0] - text_width, image_size[1] - text_height),
    }

    x, y = positions.get(position, (0, 0))  # Default to top-left if position is invalid

    # Adjust for bottom alignments to ensure text is not cut off
    if position in ['bottom-left', 'bottom-center', 'bottom-right']:
        y -= font.getmetrics()[1]  # Adjust y by the font's descent to avoid clipping

    # Draw the text on the image
    draw.text((x, y), bidi_text, font=font, fill='black')

    return img
