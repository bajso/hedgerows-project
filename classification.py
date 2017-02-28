import logging
import os

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
# 'Feature_comb.tif' uses 255 as NoData values
# 'Feature_comb_0.tif' uses 0 as NoData values
training_data_name = 'Feature_comb_0.tif'  # uses 255 as NoData values


def import_input_data():
    print '\nRead input .tif data file and import it as numpy array'

    # import
    input_data = os.path.join(input_data_path, input_data_name)
    try:
        input_dataset = gdal.Open(input_data, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # get geospatial coordinates
    geo_transform = input_dataset.GetGeoTransform()
    # projection reference
    projection = input_dataset.GetProjectionRef()

    bands_data = []
    # import and stack all bands of input satellite image
    # bands from 1 - 11
    # RasterCount returns number of bands in a dataset
    for b in range(1, input_dataset.RasterCount + 1):
        # gets a band
        band = input_dataset.GetRasterBand(b)
        # reads band to memory
        bands_data.append(band.ReadAsArray())

    # stack array in depth, along third axis
    bands_data = np.dstack(bands_data)
    # store number of rows and columns (i.e. img resolution in pixels) and the number of bands (depth)
    rows, cols, number_of_bands = bands_data.shape

    # total number of samples or pixels
    tot_samples = rows * cols

    # check for pixel depth
    str = 'Input data type: {}'.format(bands_data.dtype)
    print str

    return bands_data, rows, cols, number_of_bands, tot_samples


def import_training_data():
    print '\nRead training .tif data file and import it as numpy array'

    training_data = os.path.join(training_data_path, training_data_name)
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

    s = 'Training data type: {}'.format(training_pixels.dtype)
    print s

    # Training data is 990x951 pixel array, 8 bit colour depth
    return training_pixels


if __name__ == "__main__":

    input_data_attributes = import_input_data()
    bands = input_data_attributes[0]
    rows = input_data_attributes[1]
    cols = input_data_attributes[2]
    number_of_bands = input_data_attributes[3]
    tot_samples = input_data_attributes[4]

    input_training_attributes = import_training_data()
    # only consider labelled pixels (e.g. 1=forest, 2=field, 3=grassland) .... 0=NoData
    train = np.nonzero(input_training_attributes)
    # 8823 labelled pixels
    training_labels = input_training_attributes[train]
    training_samples = bands[train]

    # training_samples - list of pixels to be used for training
    # pixel is a point in the 7-dimensional space of bands

    # training_labels - list of class labels such that the i-th position indicates the class
    # for i-th pixel in training_samples

    counter = 0
    no_data = 0
    forest = 0
    field = 0
    grassland = 0
    for i in training_labels:
        # 0 is the no-data pixel value
        if i == 0:
            no_data += 1
        if i == 1:
            forest += 1
        if i == 2:
            field += 1
        if i == 3:
            grassland += 1

        counter += 1

    print '\nForest labels: {}\nField values: {}\nGrassland values: {}\nTotal count: {}'.format(forest, field,
                                                                                                grassland, counter)

    from datetime import datetime

    start_time = datetime.now().time()
    print 'Training...\nStart Time: '
    print start_time

    ### TRAINING
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True,
                                n_jobs=-1, verbose=True, oob_score=True, class_weight='balanced')

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
    print 'PLOTTTING'
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

    f.savefig(os.path.join(output_data_path, 'output.png'))
    f.show()

    print 'DONE'
