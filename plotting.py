import os

import numpy as np
from gdalconst import *
from matplotlib import pyplot as plt
from osgeo import gdal

input_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\input_file'
output_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file'
input_data_name = 'input_Clip.tif'
output_data_name = 'classification.tiff'
training_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data_tif'
training_data_name = 'Feature_comb.tif'


def import_input_data():
    print 'Read input .tif data file and import it as numpy array'

    # import
    input_data = os.path.join(input_data_path, input_data_name)
    try:
        input_dataset = gdal.Open(input_data, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    bands_data = []
    # import and stack all bands of input satellite image
    # bands from 1 - 11
    # RasterCount returns number of bands in a dataset
    for b in range(1, input_dataset.RasterCount + 1):
        # gets a band
        band = input_dataset.GetRasterBand(b)
        # reads band to memory
        im = plt.imshow(band.ReadAsArray())
        plt.show()

        bands_data.append(band.ReadAsArray())

    # stack array in depth, along third axis
    bands_data = np.dstack(bands_data)
    # store number of rows and columns (i.e. img resolution in pixels) and the number of bands (depth)
    rows, cols, number_of_bands = bands_data.shape

    # total number of samples or pixels
    tot_samples = rows * cols

    print bands_data.dtype

    return bands_data, rows, cols, number_of_bands, tot_samples


att = import_input_data()

bands = att[0]
print  bands.shape

r = bands[:, :, 3]
g = bands[:, :, 2]
b = bands[:, :, 1]

rgb = np.dstack([r, g, b])

print rgb.shape

img = plt.imshow(rgb / 255)
plt.show(img)
plt.savefig('test.png')
