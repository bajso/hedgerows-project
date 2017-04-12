import cPickle as pickle
import os
import re

import easygui as e
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
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

COLOURS = ['#FFFFFF', '#F6C5AF', '#5AAA95', '#9A031E', '#000000']


def import_image_data(folder_path):
    # import as read only
    try:
        image_data = gdal.Open(folder_path, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # get geospatial coordinates
    geo_transform = image_data.GetGeoTransform()
    # projection reference - convert Earth sphere img to flat 2D screen, usually Mercator projection
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

    # close dataset
    image_data = None

    return bands, rows, cols, number_of_bands, geo_transform, projection


def import_training_data(folder_path):
    try:
        training_data = gdal.Open(folder_path, GA_ReadOnly)
    except RuntimeError:
        print 'Cannot open desired path'
        exit(1)

    # shapefiles are constructed of a single band
    band = training_data.GetRasterBand(1)
    # read to memory
    training_array = band.ReadAsArray()

    ### DELETE COMMENTS
    # feature to raster instead of gdal rasterize function
    # could merge all shapefiles into one and then rasterize or could import all raw .shp shapefiles
    # and then add all non zero pixels to same training array

    # close dataset
    training_data = None

    # Training data is ~990x951 pixel array, 8 bit colour depth
    return training_array


def write_geotiff(folder_name, file_name, data, geo_transform, projection):
    path = os.path.join(folder_name, file_name)
    # check if valid path
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    print '\nSaving Classification As: \'{}\'\n'.format(file_name)

    driver = gdal.GetDriverByName('GTiff')
    rows, cols = data.shape
    # create an image of type unit8
    # output is only 1 band
    output_data = driver.Create(path, cols, rows, 1, gdal.GDT_Byte)
    output_data.SetGeoTransform(geo_transform)
    output_data.SetProjection(projection)
    band = output_data.GetRasterBand(1)

    # list of colours
    custom_colours = COLOURS
    # create a colour table
    colour_table = gdal.ColorTable()
    # add colours to the table
    # start from the second colour on the list, first is plain white
    for n in range(1, len(custom_colours)):
        # convert from hex to rgb values
        colour_entry = colors.hex2color(custom_colours[n])
        # convert to 0-255 colour range
        colour_entry = [int(255 * x) for x in colour_entry]
        r = colour_entry[0]
        g = colour_entry[1]
        b = colour_entry[2]
        a = 255  # alpha or black band
        # store in colour table
        colour_table.SetColorEntry(n, (r, g, b, a))

    band.SetColorTable(colour_table)

    # write the image
    band.WriteArray(data)

    # Close the file
    output_data = None


def write_scores(data, name, file_name):
    path = os.path.join(accuracy_score_path, file_name)
    # check if valid path
    if not os.path.exists(accuracy_score_path):
        os.makedirs(accuracy_score_path)

    if os.path.isfile(path) and iteration > 1:
        f = open(path, 'a+')  # appends in the end
    else:
        f = open(path, 'w+')  # creates the file

    f.write(name + '\n')
    for s in data:
        f.write(str(s))
        f.write('\n')
    f.write('\n')

    print 'Saving Accuracy Report'
    f.close()


def validation(classifier, test_samples, test_labels, class_acc):
    class_names = ['Forest', 'Field', 'Grassland', 'Other']

    ### DELETE COMMENT
    # predicts only the test data for comparison against test labels
    predicted_labels = classifier.predict(test_samples)

    confusion_matrix = metrics.confusion_matrix(test_labels, predicted_labels)
    confusion_matrix_str = "Confusion matrix:\n\n{}\n".format(confusion_matrix)

    classification_report = "Classification report:\n\n{}".format(
        metrics.classification_report(test_labels, predicted_labels, target_names=class_names))
    print classification_report

    classification_accuracy = metrics.accuracy_score(test_labels, predicted_labels)
    classification_accuracy_str = "Classification accuracy: {}".format(classification_accuracy)
    print classification_accuracy_str

    # write report
    data = [confusion_matrix_str, classification_report, classification_accuracy_str]
    class_acc.append(classification_accuracy)

    write_scores(data, 'Iteration: {}\n'.format(iteration), accuracy_score_name)

    plot_confusion_matrix(class_names, confusion_matrix)


def plot_confusion_matrix(class_names, conf_matrix):
    plt_colour = plt.cm.BuPu  # blue purpleish
    plt_title = 'Confusion Matrix {c}_{n}_i{i}'.format(c=classifier_s, n=no_estimators, i=iteration)
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
    # check if valid path
    if not os.path.exists(validation_path):
        os.makedirs(validation_path)

    plt.savefig(path)
    plt.clf()


def save_load_segments(img, seg_result, folder_path, file_path, write):
    # check if valid path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # creates empty ndarray of type int
    seg_file = np.ndarray(shape=(1000, 1000), dtype=int)
    if write:
        # write results to a file
        with open(file_path, 'wb') as file_out:
            pickle.dump(seg_result, file_out, -1)
        print '\nFile saved at: ', file_path
    else:
        # check if the result exists
        if not os.path.isfile(file_path):
            # run the segmentation if it does not
            seg_file = perform_segmentation(img)
            return seg_file

        # read and return results from a file
        with open(file_path, 'rb') as file_in:
            seg_file = pickle.load(file_in)
        print '\nSegmentation file loaded'

    return seg_file


def save_load_objects(img, segments, obj, folder_path, file_path, write):
    # check if valid path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    objects_file = None
    if write:
        # write results to a file
        with open(file_path, 'wb') as file_out:
            pickle.dump(obj, file_out, -1)
        print '\nFile saved at: ', file_path
    else:
        # check if the result exists
        if not os.path.isfile(file_path):
            # create objects if it does not
            objects_file = create_objects(img, segments)
            return objects_file[1]  # returns only object samples

        # read and return results from a file
        with open(file_path, 'rb') as file_in:
            objects_file = pickle.load(file_in)
        print '\nObjects file loaded'

    return objects_file


def perform_segmentation(scaled_image):
    def input_seg_plot(img, n_seg, file_path, plt_title):
        # create random colours for all segments
        plt_colours = colors.ListedColormap(np.random.rand(n_seg, 3))
        plt.imshow(img, cmap=plt_colours, interpolation='none')
        # to fit properly with all labels
        plt.tight_layout()

        path = os.path.join(file_path, 'plots', plt_title + '.png')
        # check if valid path
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        plt.savefig(path)
        print 'Fig saved in %s' % path

        plt.clf()

    def quick_seg(img):
        # params
        k_size = 7
        max_d = 2
        ratio = 0.35

        segments_data = quickshift(img, kernel_size=k_size, max_dist=max_d, ratio=ratio, convert2lab=False)
        n_segments = len(np.unique(segments_data))

        # plot results
        name = 'quick_k{k}_d{d}_r{r}'.format(k=k_size, d=max_d, r=ratio)
        input_seg_plot(segments_data, n_segments, path_quickshift, name)
        # store segments
        file_path = os.path.join(path_quickshift, 'segments_' + name + '.pkl')
        save_load_segments(None, segments_data, path_quickshift, file_path, True)

        return segments_data

    def slic_seg(img):
        # param
        n_seg = 10000
        segments_data = slic(img, n_segments=n_seg, compactness=0.1, max_iter=10, sigma=0, convert2lab=False,
                             slic_zero=False)
        n_segments = len(np.unique(segments_data))

        # plot results
        name = 'slic_n{n}'.format(n=n_seg)
        input_seg_plot(segments_data, n_segments, path_slic, name)
        # store segments
        file_path = os.path.join(path_slic, 'segments_' + name + '.pkl')
        save_load_segments(None, segments_data, path_slic, file_path, True)

        return segments_data

    if seg_algo == 'quickshift':
        segments = quick_seg(scaled_image)
    else:
        segments = slic_seg(scaled_image)

    return segments


def create_objects(scaled_image, segments):
    def compute_statistics(s_pixels):
        # Compute statistics for each Band
        # min, max, mean, variance, skewness, kurtosis
        attributes = []
        number_of_pixels, number_of_bands = s_pixels.shape

        for band in range(number_of_bands):
            statistics = sp.stats.describe(s_pixels[:, band])
            # min and max is a tuple
            b_stats = list(statistics[1])
            # add them all together
            b_stats += list(statistics[2:])
            # check if any attribute is NaN due to division with neg values
            i = 0
            for s in b_stats:
                if np.isnan(s):
                    # replace with zero
                    b_stats[i] = float(0)
                i += 1

            attributes += b_stats

        return attributes

    object_samples = []
    object_labels = []

    n_segments = np.unique(segments)
    for s in n_segments:
        train_segments = scaled_image[segments == s]
        # compute statistics for each object
        object_samples.append(compute_statistics(train_segments))
        # assign a class label to each object
        object_labels.append(s)

    file_path_o = os.path.join(segmentation_path, 'objects' + '.pkl')
    file_path_l = os.path.join(segmentation_path, 'object_labels' + '.pkl')
    save_load_objects(None, None, object_samples, segmentation_path, file_path_o, True)
    save_load_objects(None, None, object_labels, segmentation_path, file_path_l, True)

    print("Created %d objects" % len(object_samples))

    return object_samples, object_labels


def obia(img_samples, t_dataset):
    def plot_training_segments():

        def plot(segments_data):
            # use predefined colours
            custom_colours = COLOURS
            colour_map = colors.ListedColormap(custom_colours)
            plt.imshow(segments_data, cmap=colour_map)
            # for 4 different features + NoData
            plt.colorbar(ticks=[0, 1, 2, 3, 4])
            plt.tight_layout()

            path = os.path.join(segmentation_path, 'train_segments' + '.png')
            # check if valid path
            if not os.path.exists(validation_path):
                os.makedirs(validation_path)

            plt.savefig(path)
            print 'Fig saved in %s' % path

            plt.clf()

        # DELETE BEFORE SUBMITTING AND ONLY USE PICS

        # label non-training segments with 0
        # create a copy only for plotting purposes
        seg_img = np.copy(segmentation)
        # threshold needs to be higher than the max number of segments
        t = len(n_segments)
        for l in labels:
            class_label = t + l
            # label all segments that match with class label
            for segment_label in training_segments[l]:
                seg_img[seg_img == segment_label] = class_label

        # all segments that do no appear in marked training areas are labelled as 0
        seg_img[seg_img <= t] = 0
        # rest are marked with class labels, e.g 1-forest, 2-field, etc. by subtracting the threshold value
        seg_img[seg_img > t] -= t

        # plot training segments
        plot(seg_img)

    # segmentation tools work with values between 0 and 1, hence image needs to be rescaled
    scaled_img = exposure.rescale_intensity(img_samples)

    # either load existing segments or create new ones
    segmentation = None
    if load_segments:
        segmentation = save_load_segments(scaled_img, None, segmentation_path, segments_path, False)
    else:
        segmentation = perform_segmentation(scaled_img)

    n_segments = np.unique(segmentation)
    # no of class labels of training data, 0 is marked as NoData
    labels = np.unique(t_dataset)[1:]

    # check which segments are in shapefile areas
    training_segments = {}
    for l in labels:
        # part of the same class if training labels and segment labels match
        class_segments = segmentation[t_dataset == l]
        # set builds unordered collection of unique objects - no duplication
        training_segments[l] = set(class_segments)

    # check if segments contain training pixels of different classes
    # |= is set union that updates the set instead of returning a new one == update()
    segments_union = set()
    intersect = set()
    training_segments_true = {}
    for class_segments in training_segments.values():
        # for all segments with same label check if any intersect with different labels
        intersect.update(segments_union.intersection(class_segments))
        segments_union.update(class_segments)

    # if they do, remove them from training segments array
    i = 1
    for class_segments in training_segments.values():
        training_segments_true[i] = class_segments - intersect
        i += 1

    # assign back to the old name
    training_segments = None
    training_segments = training_segments_true
    training_segments_true = None

    # create and plot training segments
    plot_training_segments()

    # create objects / training data
    objects = None
    object_labels = None
    if load_objects:
        file_path_o = os.path.join(objects_path, 'objects' + '.pkl')
        file_path_l = os.path.join(objects_path, 'object_labels' + '.pkl')
        objects = save_load_objects(scaled_img, segmentation, None, segmentation_path, file_path_o, False)
        object_labels = save_load_objects(scaled_img, segmentation, None, segmentation_path, file_path_l, False)
    else:
        objects, object_labels = create_objects(scaled_img, segmentation)

    training_labels = []
    training_objects = []
    for l in labels:

        class_objects = []
        for i, features in enumerate(objects):
            if object_labels[i] in training_segments[l]:
                class_objects.append(features)

        # add objects
        training_objects.extend(class_objects)
        # generate labels
        training_labels.extend(l for n in range(len(class_objects)))

    return training_objects, training_labels, objects, object_labels, segmentation


def object_classification(training_dataset, input_image_samples):
    # OBIA
    training_samples, training_labels, object_samples, \
    object_labels, segmentation = obia(input_image_samples, training_dataset)

    # list to ndarray
    training_labels = np.asarray(training_labels)
    training_samples = np.asarray(training_samples)

    if cross_validate:
        cross_validation_object(training_samples, training_labels, object_samples, object_labels, segmentation)

    else:
        # train the model
        classifier.fit(training_samples, training_labels)
        # predict
        result = classifier.predict(object_samples)

        # back to pixels
        classify = np.copy(segmentation)
        for n_segments, l in zip(object_labels, result):
            classify[classify == n_segments] = l

        # write to file
        write_geotiff(output_data_path, output_data_name, classify, geo_transform, projection)


def pixel_classification(training_dataset, input_image_samples):

    # only consider labelled pixels (e.g. 1=forest, 2=field, 3=grassland, 4=other) .... 0=NoData
    shapefile_data = np.nonzero(training_dataset)
    # training_labels - list of class labels such that the i-th position indicates the class for i-th pixel in training_samples
    # search for shapefile pixels in the training array (returns labels)
    training_labels = training_dataset[shapefile_data]
    # training_samples - list of pixels to be used for training, a pixel is a point in the 7-dimensional space of bands
    # search for shapefile pixels in the input image data array (returns training samples that correspond to the labels returned from training array)
    training_samples = input_image_samples[shapefile_data]

    if cross_validate:
        cross_validation_pixel(training_samples, training_labels)

    else:
        # train the model
        classifier.fit(training_samples, training_labels)
        # total number of samples or pixels in an image (1000*1000)

        tot_samples = rows * cols
        # reshape for classification input - needs to be array of pixels
        flat_pixels = input_image_samples.reshape((tot_samples, number_of_bands))
        result = classifier.predict(flat_pixels)
        # reshape back into image form
        classify = result.reshape((rows, cols))

        # write to file
        write_geotiff(output_data_path, output_data_name, classify, geo_transform, projection)


def cross_validation_object(training_samples, training_labels, object_samples, object_labels, segmentation):
    # K-fold cross validation - splits in kfv_splits equal chunks
    kf = KFold(n_splits=kfv_splits)

    cvs = cross_val_score(classifier, training_samples, training_labels, cv=kf, n_jobs=-1, pre_dispatch='2*n_jobs')
    print '\nBuilt-in Cross Validation Score:\n{}\n'.format(cvs)

    class_acc = []
    iteration = 0
    output_data_name = 'classified_{c}_{n}_i{i}.tiff'.format(c=classifier_s, n=no_estimators, i=iteration)

    for train, test in kf.split(training_samples):
        iteration += 1

        head, tail = output_data_name.split('.')
        output_data_name = head[:-1] + str(iteration) + '.' + tail

        # split training dataset to training and testing samples
        X_train, X_test, y_train, y_test = training_samples[train], training_samples[test], training_labels[train], \
                                           training_labels[test]

        # train the model
        classifier.fit(X_train, y_train)
        # predict
        result = classifier.predict(object_samples)

        # back to pixels
        classify = np.copy(segmentation)
        for n_segments, l in zip(object_labels, result):
            classify[classify == n_segments] = l

        # write to file
        write_geotiff(output_data_path, output_data_name, classify, geo_transform, projection)

        # evaluation
        validation(classifier, X_test, y_test, class_acc)

    # write cross-validation scores
    write_s = []
    for i in range(kfv_splits):
        write_s.append('{0:10f} {1:10f}'.format(cvs[i], class_acc[i]))
        print write_s[i]

    write_scores(write_s, '\nCross Validation Scores / Classification Accuracy Scores\n', accuracy_score_name)


def cross_validation_pixel(training_samples, training_labels):
    # K-fold cross validation - splits in kfv_splits equal chunks
    kf = KFold(n_splits=kfv_splits)

    cvs = cross_val_score(classifier, training_samples, training_labels, cv=kf, n_jobs=-1, pre_dispatch='2*n_jobs')
    print '\nBuilt-in Cross Validation Score:\n{}\n'.format(cvs)

    class_acc = []
    iteration = 0
    output_data_name = 'classified_{c}_{n}_i{i}.tiff'.format(c=classifier_s, n=no_estimators, i=iteration)

    for train, test in kf.split(training_samples):
        iteration += 1

        head, tail = output_data_name.split('.')
        output_data_name = head[:-1] + str(iteration) + '.' + tail

        X_train, X_test, y_train, y_test = training_samples[train], training_samples[test], training_labels[train], \
                                           training_labels[test]

        # train the model
        classifier.fit(X_train, y_train)

        # total number of samples or pixels in an image (1000*1000)
        tot_samples = rows * cols
        # reshape for classification input - needs to be array of pixels
        flat_pixels = input_image_samples.reshape((tot_samples, number_of_bands))

        # predict
        result = classifier.predict(flat_pixels)
        # reshape back into image form
        classify = result.reshape((rows, cols))

        # write to file
        write_geotiff(output_data_path, output_data_name, classify, geo_transform, projection)

        # evaluation
        validation(classifier, X_test, y_test, class_acc)

    # write cross-validation scores
    write_s = []
    for i in range(kfv_splits):
        write_s.append('{0:10f} {1:10f}'.format(cvs[i], class_acc[i]))

    write_scores(write_s, '\nCross Validation Scores / Classification Accuracy Scores\n', accuracy_score_name)


def collect_params():
    # PATHS
    global image_data_path, training_data_path, output_data_path

    print '\nSelect Input Image File'
    image_data_path = e.fileopenbox(msg="Select File", title="Import Input Image File",
                                    filetypes=("tiff files", "*.tif"), multiple=False)
    print '\nSelect Training Data File'
    training_data_path = e.fileopenbox(msg="Select File", title="Import Training Data File",
                                       filetypes=("tiff files", "*.tif"), multiple=False)
    print '\nSelect Output Folder'
    output_data_path = e.diropenbox(title="Select Output Folder")

    global path_rf, path_knn, validation_path, accuracy_score_path, segmentation_path, path_quickshift, path_slic
    path_rf = os.path.join(output_data_path, 'rf')
    path_knn = os.path.join(output_data_path, 'knn')
    validation_path = os.path.join(output_data_path, 'validation')
    accuracy_score_path = os.path.join(validation_path, 'scores')
    segmentation_path = os.path.join(output_data_path, 'segmentation')
    path_quickshift = os.path.join(segmentation_path, 'quickshift')
    path_slic = os.path.join(segmentation_path, 'slic')

    # PARAMS
    global classifier_s, no_estimators, do_obia, load_segments, load_objects, seg_algo, cross_validate, kfv_splits, iteration

    # CONFIG
    print '\nClassification Configuration'
    conf_txt = "Set Classification Configuration"
    conf_title = "Classification Configuration"
    conf_fields = ['Classifier', 'Number Of Estimators', 'Image Analysis Type', 'Segmentation Algorithm',
                   'Load Segments', 'Load Objects', 'Cross-Validation', 'k-splits']
    conf_values = ['random-forests / knn', '100 for RF / 10 for KNN', 'object-based / pixel-based', 'quickshift / slic',
                   'True / False', 'True / False', 'True / False', '10']
    config = e.multenterbox(msg=conf_txt, title=conf_title, fields=conf_fields, values=conf_values)

    # DEFAULTS
    classifier_s = config[0] if (config[0] == 'random-forests' or config[0] == 'knn') else 'random-forests'
    no_estimators = int(config[1]) if re.match('^[0-9]{1,4}$', config[1]) else (
    10 if config[0] == 'knn' else 100)  # is number
    do_obia = True if config[2] == 'object-based' else False
    seg_algo = config[3] if (config[3] == 'quickshift' or config[3] == 'slic') else 'quickshift'
    load_segments = True if config[4] == 'True' else False  # is file
    load_objects = True if config[5] == 'True' else False  # is folder
    cross_validate = True if config[6] == 'True' else False
    kfv_splits = int(config[7]) if re.match('^[0-9]{1,2}$', config[7]) else 10
    iteration = 0

    global segments_path, objects_path
    if load_objects:
        objects_path = e.diropenbox(title="Select Folder With Segmentation Objects")
    if load_segments:
        segments_path = e.fileopenbox(msg="Select File", title="Import Segmentation File", multiple=False)

    # OUTPUT
    global output_data_name, accuracy_score_name
    output_data_name = 'classified_{c}_{n}_i{i}.tiff'.format(c=classifier_s, n=no_estimators, i=iteration)
    accuracy_score_name = 'accuracy_scores_{c}_{n}.txt'.format(c=classifier_s, n=no_estimators)


if __name__ == "__main__":

    collect_params()

    image_dataset = import_image_data(image_data_path)

    # image attributes
    input_image_samples = image_dataset[0]
    rows = image_dataset[1]
    cols = image_dataset[2]
    number_of_bands = image_dataset[3]
    geo_transform = image_dataset[4]
    projection = image_dataset[5]

    # training attributes
    training_dataset = import_training_data(training_data_path)

    # classifiers
    rf = RandomForestClassifier(n_estimators=no_estimators, criterion='gini', bootstrap=True, max_features='auto',
                                n_jobs=-1, verbose=False, oob_score=True, class_weight='balanced')

    knn = KNeighborsClassifier(n_neighbors=no_estimators, weights='distance', algorithm='auto', metric='minkowski',
                               n_jobs=-1)

    if classifier_s == 'random-forests':
        classifier = rf
        output_data_path = path_rf
    else:
        classifier = knn
        output_data_path = path_knn

    print '\nObject-Based Classification Starting\n'

    if do_obia:
        print '\nObject-Based Classification Starting\n'
        object_classification(training_dataset, input_image_samples)

    # pixel-based image analysis
    else:
        print '\nPixel-Based Classification Starting\n'
        pixel_classification(training_dataset, input_image_samples)

    print '\nDONE'
