import os

import matplotlib.pyplot as plt
import numpy as np
from gdalconst import *
from matplotlib import colors
from osgeo import gdal
from skimage import exposure
from skimage.segmentation import quickshift, slic
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Populate colours array with some random colours
COLOURS = ['test', '#F6C5AF', '#5AAA95', '#9A031E', '#000000', '#A3178F']
# COLOURS =  create_colours()

# Parameters
classifier = None
class_s = 'rf'  # for naming purposes
no_trees = 100
no_neighbours = 10
no_estimators = no_trees if class_s == 'rf' else no_neighbours
features_comb = 3
kfv_splits = 2
iteration = 0

# Folder paths
image_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\input_file'
output_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file'
output_data_path_rf = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file\rf'
output_data_path_knn = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file\knn'
training_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data'
validation_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file\validation'
segmentation_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file\segmentation'

# File paths
input_data_name = 'input_Clip.tif'
output_data_name = 'c_out_f{f}_{c}_{n}_i{i}.tiff'.format(f=features_comb, c=class_s, n=no_estimators, i=iteration)
accuracy_score_name = 'accuracy_scores.txt'

# 'Feature_comb__1,2,3.tif' different areas of training attributes
# All have 0 set as NoData
training_data_name = 'Feature_comb{}.tif'.format(features_comb)


def import_image_data(folder_name, file_name):
    print '\nRead input .tif data file and import it as numpy array'

    # import as read only
    path = os.path.join(folder_name, file_name)
    try:
        image_data = gdal.Open(path, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # get geospatial coordinates
    geo_transform = image_data.GetGeoTransform()
    # projection reference
    projection = image_data.GetProjectionRef()

    bands = []
    # import and stack all bands of input satellite image
    # bands from 1 - 7
    # RasterCount returns number of bands
    for b in range(1, image_data.RasterCount + 1):
        # gets a band
        band = image_data.GetRasterBand(b)
        # reads band to memory
        bands.append(band.ReadAsArray())

    # stack array in depth, along third axis
    bands = np.dstack(bands)
    # store number of rows and columns (i.e. img resolution in pixels) and the number of bands (depth)
    rows, cols, number_of_bands = bands.shape  # 1000,1000,7

    # total number of samples or pixels
    tot_samples = rows * cols

    # check for pixel depth
    print 'Input data type: {}'.format(bands.dtype)
    # close dataset
    image_data = None

    return bands, rows, cols, number_of_bands, tot_samples, geo_transform, projection


def import_training_data(folder_name, file_name):
    print '\nRead training .tif data file and import it as numpy array'

    path = os.path.join(folder_name, file_name)
    try:
        training_data = gdal.Open(path, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # shapefiles are constructed of a single band
    band = training_data.GetRasterBand(1)
    # read to memory
    training_array = band.ReadAsArray()

    # feature to raster instead of gdal rasterize function
    # could merge all shapefiles into one and then rasterize or could import all raw .shp shapefiles
    # and then add all non zero pixels to same training array

    print 'Training data type: {}'.format(training_array.dtype)
    # close dataset
    training_data = None

    # Training data is ~990x951 pixel array, 8 bit colour depth
    return training_array


def write_geotiff(folder_name, file_name, data, geo_transform, projection):
    path = os.path.join(folder_name, file_name)
    print 'Saving as: \'{}\''.format(file_name)

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    # create an image of type unit8
    # output is only 1 band
    output_data = driver.Create(path, cols, rows, 1, gdal.GDT_Byte)
    output_data.SetGeoTransform(geo_transform)
    output_data.SetProjection(projection)
    band = output_data.GetRasterBand(1)
    band.WriteArray(data)

    colour_table = gdal.ColorTable()
    for label_colour in range(1, 6):
        color_hex = COLOURS[label_colour]
        # creating colours for r g and b, as a 16bit int
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        colour_table.SetColorEntry(label_colour, (r, g, b, 255))

    band.SetColorTable(colour_table)

    # Close the file
    output_data = None


def count_samples(training_labels):
    counter = 0

    no_data = 0
    forest = 0
    field = 0
    grassland = 0
    other = 0
    hedgerow = 0
    for l in training_labels:
        # 0 is the no-data pixel value
        if l == 0:
            no_data += 1
        if l == 1:
            forest += 1
        if l == 2:
            field += 1
        if l == 3:
            grassland += 1
        if l == 4:
            other += 1
        if l == 5:
            hedgerow += 1

        counter += 1

    print '\nForest labels: {fo}\nField values: {fi}\nGrassland values: {g}\nHedgerow values: {h}\nOther values: {o}\nTotal count: {tot}'.format(
        fo=forest, fi=field, g=grassland, h=hedgerow, o=other, tot=counter)

    # weight normalization = 'balanced setting'
    weights = {1: forest / float(counter), 2: field / float(counter), 3: grassland / float(counter),
               4: other / float(counter)}

    return weights


def write_to(data, name, file_name):
    file_path = os.path.join(validation_path, file_name)

    if os.path.isfile(file_path) and iteration > 1:
        f = open(file_path, 'a+')  # appends in the end
    else:
        f = open(file_path, 'w+')  # creates the file

    f.write(name + '\n')
    for s in data:
        f.write(str(s))
        f.write('\n')
    f.write('\n')

    f.close()


def validation(classifier, test_samples, test_labels):
    class_names = ['Forest', 'Field', 'Grassland', 'Other']

    # predicts only the test data for comparison against test labels
    predicted_labels = classifier.predict(test_samples)

    # verification_pixels = input_training_attributes
    # for_verification = training_data
    # verification_labels = verification_pixels[for_verification]
    # predicted_labels = classify[for_verification]


    confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
    confusion_matrix_str = "Confusion matrix:\n\n{}\n".format(confusion_matrix)
    print confusion_matrix_str

    classification_report = "Classification report:\n\n{}".format(
        metrics.classification_report(test_labels, predicted_labels, target_names=class_names))
    print classification_report

    classification_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    classification_accuracy_str = "Classification accuracy: {}".format(classification_accuracy)
    print classification_accuracy_str

    # write report
    data = [confusion_matrix_str, classification_report, classification_accuracy_str]
    class_acc.append(classification_accuracy)

    write_to(data, 'Iteration: {}\n'.format(iteration), accuracy_score_name)

    validation_plot(class_names, confusion_matrix)


def validation_plot(class_names, conf_matrix):
    plt_colour = plt.cm.BuPu  # blue purpleish
    plt_title = 'Confusion Matrix {c}_{n}_i{i}'.format(c=class_s, n=no_estimators, i=iteration)
    plt_tick_marks = np.arange(len(class_names))

    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt_colour)

    # sample annotation
    for x in xrange(len(class_names)):
        for y in xrange(len(class_names)):
            plt.annotate(str(conf_matrix[x][y]), xy=(y, x), ha='center', va='center')

    plt.title(plt_title)
    plt.colorbar()
    plt.xlabel('Predicted samples')
    plt.ylabel('True samples')
    # custom labels
    plt.xticks(plt_tick_marks, class_names, rotation=45)
    plt.yticks(plt_tick_marks, class_names)
    # fit to figure
    plt.tight_layout()

    path = os.path.join(validation_path, plt_title + '.png')
    plt.savefig(path)

    plt.clf()


def plot(img, n_seg, plt_title):
    plt_colours = colors.ListedColormap(np.random.rand(n_seg, 3))
    plt.imshow(img, cmap=plt_colours, interpolation='none')
    plt.tight_layout()

    path = os.path.join(segmentation_path, plt_title + '.png')
    plt.savefig(path)

    plt.clf()


def obia(img_samples):
    # segmentation tools work with values between 0 and 1, hence image needs to be rescaled
    rescale_img = exposure.rescale_intensity(img_samples)

    segments_quick = quickshift(rescale_img, kernel_size=7, max_dist=3, ratio=0.35, convert2lab=False)
    n_segments_quick = len(np.unique(segments_quick))
    print n_segments_quick
    plot(segments_quick, n_segments_quick, 'quickshift')

    # felzenszwalb segmentation cannot be used with multi-band images - see result

    n_seg = [500, 1000, 4000, 8000, 10000, 15000]
    c = [0.1, 0.05, 0.01, 0.005]
    sig = [0, 0.25]
    slico = False

    for k in range(len(sig)):
        for i in range(len(n_seg)):
            for j in range(len(c)):
                segments_slic = slic(rescale_img, n_segments=n_seg[i], compactness=c[j], max_iter=10, sigma=sig[k],
                                     convert2lab=False, slic_zero=slico)
                n_segments_slic = len(np.unique(segments_slic))
                print n_segments_slic
                plot(segments_slic, n_segments_slic, 'slic_{}_{}_{}_{}'.format(n_seg[i], c[j], sig[k], slico))


if __name__ == "__main__":

    image_dataset = import_image_data(image_data_path, input_data_name)

    # image attributes
    input_image_samples = image_dataset[0]
    rows = image_dataset[1]
    cols = image_dataset[2]
    number_of_bands = image_dataset[3]
    tot_samples = image_dataset[4]
    geo_transform = image_dataset[5]
    projection = image_dataset[6]

    # training attributes
    training_dataset = import_training_data(training_data_path, training_data_name)
    # only consider labelled pixels (e.g. 1=forest, 2=field, 3=grassland, 4=other) .... 0=NoData
    shapefile_data = np.nonzero(training_dataset)

    # training_labels - list of class labels such that the i-th position indicates the class for i-th pixel in training_samples
    # search for shapefile pixels in the training array (returns labels)
    training_labels = training_dataset[shapefile_data]
    # training_samples - list of pixels to be used for training, a pixel is a point in the 7-dimensional space of bands
    # search for shapefile pixels in the input image data array (returns training samples that correspond to the labels returned from training array)
    training_samples = input_image_samples[shapefile_data]

    # count samples and return weights of each of the classes
    weights = count_samples(training_labels)

    ### OBIA
    obia(input_image_samples)


    ### TRAINING
    print '\nTraining...'

    rf = RandomForestClassifier(n_estimators=no_estimators, criterion='gini', bootstrap=True, max_features='auto',
                                n_jobs=-1, verbose=True, oob_score=True, class_weight='balanced')

    knn = KNeighborsClassifier(n_neighbors=no_estimators, weights='uniform', algorithm='auto', metric='minkowski',
                               n_jobs=-1)

    if class_s == 'rf':
        classifier = rf
        output_data_path = output_data_path_rf
    else:
        classifier = knn
        output_data_path = output_data_path_knn
    output_data_name = 'c_out_f{f}_{c}_{n}_i{i}.tiff'.format(f=features_comb, c=class_s, n=no_estimators, i=iteration)

    # K-fold cross validation - splits in 10 equal chunks (16056/10 = 1606)
    kf = KFold(n_splits=kfv_splits)
    cvs = cross_val_score(classifier, training_samples, training_labels, cv=kf, n_jobs=-1, pre_dispatch='2*n_jobs')
    print '\nCross val score: {}\n'.format(cvs)

    class_acc = []
    for train, test in kf.split(training_samples):
        iteration += 1

        head, tail = output_data_name.split('.')
        output_data_name = head[:-1] + str(iteration) + '.' + tail

        X_train, X_test, y_train, y_test = training_samples[train], training_samples[test], training_labels[train], \
                                           training_labels[test]

        classifier.fit(X_train, y_train)
        # rf.fit(training_samples, training_labels)

        ### PREDICTING
        print '\nPredicting...'

        # reshape for classification input - needs to be array of pixels
        flat_pixels = input_image_samples.reshape((tot_samples, number_of_bands))
        result = classifier.predict(flat_pixels)
        # reshape back into image form
        classify = result.reshape((rows, cols))

        # WRITING TO FILE
        write_geotiff(output_data_path, output_data_name, classify, geo_transform, projection)

        # EVALUATING
        validation(classifier, X_test, y_test)

    # write cross validation scores
    write_s = []
    for i in range(kfv_splits):
        write_s.append('{0:10f} {1:10f}'.format(cvs[i], class_acc[i]))
        print write_s[i]

    write_to(write_s, '\nCross Validation Scores / Classification Accuracy Scores\n', accuracy_score_name)

    print '\nDONE'
