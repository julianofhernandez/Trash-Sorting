"""
download_model.py
This file provides functionality to download additional models
"""

import misc
import open_clip

menu_options = ['1', '2', '3', '4', 'M']
menu_prompt = "1. GPU model: ViT-g-14 (5.47 GB)\n2. GPU model: ViT-H-14 (3.94 GB)\n3. CPU model: ViT-L-14 (1.71 GB)\n4. CPU model: ViT-B-16 (599 MB)\nM. Return to Menu"


def main() -> None:
    """
    Main function for interacting with the download menu.
    """
    print(menu_prompt)
    while (True):
        key = misc.read_input(menu_options)

        if key not in menu_options:
            misc.print_invalid_input()

        if key == menu_options[0]:
            print("Downloading model...")
            open_clip.create_model_and_transforms(
                'ViT-g-14', pretrained='laion2B-s12B-b42K')
            open_clip.get_tokenizer('ViT-g-14')
            print("Download complete!")

        elif key == menu_options[1]:
            print("Downloading model...")
            open_clip.create_model_and_transforms(
                'ViT-H-14', pretrained='laion2B-s32B-b79K')
            open_clip.get_tokenizer('ViT-H-14')
            print("Download complete!")

        elif key == menu_options[2]:
            print("Downloading model...")
            open_clip.create_model_and_transforms(
                'ViT-L-14', pretrained='laion2B-s32B-b82K')
            open_clip.get_tokenizer('ViT-L-14')
            print("Download complete!")

        elif key == menu_options[3]:
            print("Downloading model...")
            open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained='laion2B-s34B-b88K')
            open_clip.get_tokenizer('ViT-B-16')
            print("Download complete!")

        else:
            misc.print_menu_return()
            return False


if __name__ == "__main__":
    main()
