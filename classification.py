import logging
import os

import numpy as np
from gdalconst import *
from osgeo import gdal
from sklearn.ensemble import RandomForestClassifier

from ccreate import create_colours

logger = logging.getLogger(__name__)

# populate colours array with some random colours
COLOURS = create_colours()

# file and folder paths
input_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\input_file'
output_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file'
output_data_path_tif = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file_tif'
input_data_name_unit16 = 'input_Clip.tif'
input_data_name_unit8 = 'input_Clip_u.tif'
output_data_name = 'classification.tiff'
training_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data'
training_data_path_tif = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data_tif'
# 'Feature_comb.tif' uses 255 as NoData values
# 'Feature_comb_0.tif' uses 0 as NoData values
training_data_name = 'Feature_comb_0.tif'


def import_input_data(folder_name, file_name):

    print '\nRead input .tif data file and import it as numpy array'

    # import as read only
    input_data = os.path.join(folder_name, file_name)
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
    # bands from 1 - 7
    # RasterCount returns number of bands in a dataset
    for b in range(1, input_dataset.RasterCount + 1):
        # gets a band
        band = input_dataset.GetRasterBand(b)
        # reads band to memory
        bands_data.append(band.ReadAsArray())

    # stack array in depth, along third axis
    bands_data = np.dstack(bands_data)
    # store number of rows and columns (i.e. img resolution in pixels) and the number of bands (depth)
    rows, cols, number_of_bands = bands_data.shape  # 1000,1000,7

    # total number of samples or pixels
    tot_samples = rows * cols

    # check for pixel depth
    print 'Input data type: {}'.format(bands_data.dtype)

    return bands_data, rows, cols, number_of_bands, tot_samples, geo_transform, projection


def import_training_data(folder_name, file_name):
    print '\nRead training .tif data file and import it as numpy array'

    training_data = os.path.join(folder_name, file_name)
    try:
        training_dataset = gdal.Open(training_data, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # shapefiles are constructed of a single band
    band = training_dataset.GetRasterBand(1)
    # read to memory
    training_pixels = band.ReadAsArray()

    # feature to raster instead of gdal rasterize function
    # could merge all shapefiles into one and then rasterize or could import all raw .shp shapefiles
    # and then add all non zero pixels to same training array

    print 'Training data type: {}'.format(training_pixels.dtype)

    # Training data is 990x951 pixel array, 8 bit colour depth
    return training_pixels


def write_geotiff(folder_name, file_name, data, geo_transform, projection):
    output_data = os.path.join(folder_name, file_name)

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    # create an image of type unit8
    # output is only 1 band
    dataset = driver.Create(output_data, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(projection)
    band = dataset.GetRasterBand(1)
    band.WriteArray(data)

    ct = gdal.ColorTable()
    for pixel_value in range(1, 4):
        color_hex = COLOURS[pixel_value]
        # creating colours for r g and b
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        ct.SetColorEntry(pixel_value, (r, g, b, 255))

    band.SetColorTable(ct)

    dataset = None  # Close the file
    return


if __name__ == "__main__":

    # import unit16 img
    input_data_attributes = import_input_data(input_data_path, input_data_name_unit16)
    # import unit8 img
    # input_data_attributes = import_input_data(input_data_path, input_data_name_unit8)

    # input img attributes
    bands_data = input_data_attributes[0]
    rows = input_data_attributes[1]
    cols = input_data_attributes[2]
    number_of_bands = input_data_attributes[3]
    tot_samples = input_data_attributes[4]
    geo_transform = input_data_attributes[5]
    projection = input_data_attributes[6]

    # import training img
    input_training_attributes = import_training_data(training_data_path_tif, training_data_name)
    # only consider labelled pixels (e.g. 1=forest, 2=field, 3=grassland) .... 0=NoData
    training_data = np.nonzero(input_training_attributes)
    # 8823 labelled pixels, 941490 all pixels

    # training_labels - list of class labels such that the i-th position indicates the class for i-th pixel in training_samples
    training_labels = input_training_attributes[training_data]
    # training_samples - list of pixels to be used for training, a pixel is a point in the 7-dimensional space of bands
    training_samples = bands_data[training_data]

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

    ### TRAINING
    print '\nTraining...'
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', bootstrap=True,
                                n_jobs=-1, verbose=True, oob_score=True, class_weight='balanced')

    rf.fit(training_samples, training_labels)


    ### PREDICTING
    print '\nPredicting...'

    # reshape for classification input - needs to be array of pixels
    flat_pixels = bands_data.reshape((tot_samples, number_of_bands))
    result = rf.predict(flat_pixels)
    # reshape back into image form
    classify = result.reshape((rows, cols))

    # write to file
    write_geotiff(output_data_path_tif, output_data_name, classify, geo_transform, projection)


    ### PLOTTING
    print 'Plotting'
    from matplotlib import pyplot as plt

    f = plt.figure()
    f.add_subplot(1, 2, 2)
    # bands from 0-6
    r = bands_data[:, :, 3]
    g = bands_data[:, :, 2]
    b = bands_data[:, :, 1]
    rgb = np.dstack([r, g, b])
    f.add_subplot(1, 2, 1)
    plt.imshow(rgb / 255)
    f.add_subplot(1, 2, 2)
    plt.imshow(classify)

    f.savefig(os.path.join(output_data_path, 'output.png'))
    f.show()

    print 'DONE'
