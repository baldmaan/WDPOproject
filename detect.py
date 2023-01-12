import json
from pathlib import Path
from typing import Dict

import click
import cv2
from tqdm import tqdm
import numpy as np
# kernel

kernel = np.ones((7,7),np.uint8)

lower_green = np.array([33, 80, 0])
upper_green = np.array([61, 255, 255])

lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])

lower_purple = np.array([110, 50, 0])
upper_purple = np.array([168, 255, 235])

lower_yellow = np.array([20, 125, 92])
upper_yellow = np.array([32, 255, 255])

# color ranges

class Color:
     def __init__(self, hsv_up, hsv_down):
         self.hsv_up = hsv_up
         self.hsv_down = hsv_down

green = Color(upper_green, lower_green)
yellow = Color(upper_yellow, lower_yellow)
purple = Color(upper_purple, lower_purple)
red = Color(upper_red, lower_red)

colors = [red, yellow, green, purple]


def create_mask(hsv, hsv_down, hsv_up):
    mask = cv2.inRange(hsv, hsv_down, hsv_up)
    erosion = cv2.erode(mask, kernel, iterations=3)
    grad = cv2.dilate(erosion, kernel, iterations=3)
    return grad


def find_contours(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        count += 1
    return count

def count_objects(color ,hsv_img):

    mask = create_mask(hsv_img, hsv_down=color.hsv_down, hsv_up=color.hsv_up)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    amount = find_contours(mask=mask)
    return amount



def detect(img_path: str) -> Dict[str, int]:
    """Object detection function, according to the project description, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each object.
    """


    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image = cv2.GaussianBlur(img, (9, 9), 0)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)


    #TODO: Implement detection method.
    colors_amount = []

    for color in colors:
        amount = count_objects(color, hsv)
        colors_amount.append(amount)

    results = {'red': colors_amount[0], 'yellow': colors_amount[1], 'green': colors_amount[2], 'purple': colors_amount[3]}


    return results


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)



if __name__ == '__main__':
    main()
