# 15 / 03 / 2024
# Bounding Box txt
import json
import argparse
import os

from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Annotations')
parser.add_argument('--p_json', type=str, required='True',help="Path to the json file")
args = parser.parse_args()

with open(args.p_json) as f:
    data = json.load(f)

dir_name = os.path.basename(args.p_json).split('.')[0]

for a in tqdm(range(len(data['annotations'])), desc=f'File being processed: {Path(args.p_json)}'):
    width, height = data['annotations'][a]['width'], data['annotations'][a]['height'] # consistent with normal image.size

    bbox = data['annotations'][a]['bbox'] # left, top, width, heigth: https://dlr-rm.github.io/BlenderProc/_modules/blenderproc/python/writer/CocoWriterUtility.html

    xywh = [round((bbox[0]/width + (bbox[2]/2)/width),4), #normalized xywh format, round to 4 decimals. [center_x, center_y, width, height]
            round((bbox[1]/height + (bbox[3]/2/height)),4),
            round(bbox[2]/width,4),
            round(bbox[3]/height,4)]

    file_number = str(data['annotations'][a]['image_id']).zfill(6)

    bboxes = " ".join(str(b) for b in xywh)

    output_directory = f"uuv/vision/data/labels/"#{dir_name}/"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(f"{output_directory}/{file_number}.txt", "a+") as f:
        f.write(str(0)  + " " + bboxes + '\n')

