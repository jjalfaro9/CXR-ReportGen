'''preprocess.py'''

import cv2
import json
import numpy as np
import pandas as pd
import pydicom

from PIL import Image
from pathlib import Path
from tqdm import tqdm


def main():
    csv = pd.read_csv('p10.csv')
    for path in tqdm(list(map(lambda x: Path(x), csv.path))):
        png_path = path.with_suffix('.png')
        json_path = path.with_suffix('.json')
        dicom = pydicom.dcmread(path.as_posix())
        view = dicom.ViewPosition
        with open(json_path, 'w') as f:
            json.dump({'view': view}, f)
        pa = dicom.pixel_array
        pa = cv2.resize(pa, dsize=(256, 256))
        pa = (255.0 / pa.max() * (pa - pa.min())).astype(np.uint8)
        image = Image.fromarray(pa)
        image.save(png_path)


if __name__ == '__main__':
    main()
