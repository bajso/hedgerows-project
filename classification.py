import logging

import numpy as np
from gdalconst import *
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier

from ccreate import create_colours

logger = logging.getLogger(__name__)

COLOURS = create_colours()

input_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\input_file'
output_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file'
input_data_name = 'input_Clip.tif'
output_data_name = 'classification.tiff'
training_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data_tif'
training_data_name = 'Feature_comb.tif'


def import_input_data():
    print 'Read input .tif data file and import it as numpy array'

    # import
    input_data = input_data_path + '\\' + input_data_name
    try:
        input_dataset = gdal.Open(input_data, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # get geospatial coordinates
    geo_transform = input_dataset.GetGeoTransform()
    # projection reference
    projection = input_dataset.GetProjectionRef()

    bands = []
    # import and stack all bands of input satellite image
    # bands from 1 - 11
    # RasterCount returns number of bands in a dataset
    for b in range(1, input_dataset.RasterCount + 1):
        # gets a band
        band = input_dataset.GetRasterBand(b)
        # reads band to memory
        bands.append(band.ReadAsArray())

    # stack array in depth, along third axis
    bands = np.dstack(bands)
    # store number of rows and columns (i.e. img resolution in pixels) and the number of bands (depth)
    rows, cols, number_of_bands = bands.shape

    # total number of samples or pixels
    tot_samples = rows * cols

    return bands, rows, cols, number_of_bands, tot_samples


def import_training_data():
    print 'Read training .tif data file and import it as numpy array'

    training_data = training_data_path + '\\' + training_data_name
    try:
        training_dataset = gdal.Open(training_data, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # get geospatial coordinates
    # geo_transform = input_dataset.GetGeoTransform()
    # projection reference
    # projection = input_dataset.GetProjectionRef()

    # shapefiles are constructed of a single band
    band = training_dataset.GetRasterBand(1)
    # read
    training_pixels = band.ReadAsArray()

    # feature to raster instead of gdal rasterise function
    # when export save only non zero values
    # could merge all shapefiles into one and then rasterise or could import all raw .shp shapefiles
    # and then add all non zero pixels to same training array

    return training_pixels


input_data_attributes = import_input_data()
bands = input_data_attributes[0]
rows = input_data_attributes[1]
cols = input_data_attributes[2]
number_of_bands = input_data_attributes[3]
tot_samples = input_data_attributes[4]

input_training_attributes = import_training_data()
# only consider labelled pixels as training data (2-d array)
train = np.nonzero(input_training_attributes)
training_labels = input_training_attributes[train]
training_samples = bands[train]

# training_samples is the list of pixels to be used for training.
# In our case, a pixel is a point in the 7-dimensional space of the bands.

# training_labels is a list of class labels such that the i-th position indicates the class
#  for i-th pixel in training_samples


from datetime import datetime

start_time = datetime.now().time()
print 'Training...\nStart Time: '
print start_time

### TRAINING
rf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True,
                            n_jobs=4, verbose=True, oob_score=True, class_weight='balanced')

rf.fit(training_samples, training_labels)

stop_time = datetime.now().time()
print 'Stop Time: '
print stop_time

start_time = datetime.now().time()
print '\n\nPredicting...\nStart Time: '
print start_time

### PREDICTING
flat_pixels = bands.reshape((tot_samples, number_of_bands))
result = rf.predict(flat_pixels)
classify = result.reshape((rows, cols))

stop_time = datetime.now().time()
print 'Stop Time: '
print stop_time

### PLOTTING
from matplotlib import pyplot as plt

f = plt.figure()
f.add_subplot(1, 2, 2)
r = bands[:, :, 4]
g = bands[:, :, 3]
b = bands[:, :, 2]
rgb = np.dstack([r, g, b])
f.add_subplot(1, 2, 1)
plt.imshow(rgb / 255)
f.add_subplot(1, 2, 2)
plt.imshow(classify)
