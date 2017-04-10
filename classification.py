import cPickle as pickle
import os

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
training_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\training_data'
output_data_path = r'C:\Users\Dida\Desktop\Hedgerows Data\ArcMap Data\project_2\output_file'
output_data_path_rf = os.path.join(output_data_path, 'rf')
output_data_path_knn = os.path.join(output_data_path, 'knn')
validation_path = os.path.join(output_data_path, 'validation')
accuracy_score_path = os.path.join(validation_path, 'scores')
segmentation_path = os.path.join(output_data_path, 'segmentation')
segmentation_path_quickshift = os.path.join(segmentation_path, 'quickshift')
segmentation_path_slic = os.path.join(segmentation_path, 'slic')

# File paths
input_data_name = 'input_Clip.tif'
output_data_name = 'c_out_f{f}_{c}_{n}_i{i}.tiff'.format(f=features_comb, c=class_s, n=no_estimators, i=iteration)
accuracy_score_name = 'accuracy_scores_{c}_{n}.txt'.format(c=class_s, n=no_estimators)

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
    # projection reference - display img from a sphere - Earth to flat 2D screen, usually Mercator projection - check ArcMap Img Properties for more info
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

    # check for pixel depth
    print 'Input data type: {}'.format(bands.dtype)
    # close dataset
    image_data = None

    return bands, rows, cols, number_of_bands, geo_transform, projection


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
        a = 255  # alpha or blackband
        print colour_entry
        # store in colour table
        colour_table.SetColorEntry(n, (r, g, b, a))

    band.SetColorTable(colour_table)

    # write the image
    band.WriteArray(data)

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


def write_scores(data, name, file_name):
    file_path = os.path.join(accuracy_score_path, file_name)

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

    write_scores(data, 'Iteration: {}\n'.format(iteration), accuracy_score_name)

    confusion_matrix_plot(class_names, confusion_matrix)


def confusion_matrix_plot(class_names, conf_matrix):
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


def save_or_load_segments(seg_result, folder_path, file_name, write):
    path = os.path.join(folder_path, file_name + '.pkl')

    # creates empty ndarray of type int
    seg_file = np.ndarray(shape=(1000, 1000), dtype=int)
    if write:
        # write results to a file
        with open(path, 'wb') as file_out:
            pickle.dump(seg_result, file_out, -1)
        print '\nFile saved at: ', path
    else:
        # read and return results from a file
        with open(path, 'rb') as file_in:
            seg_file = pickle.load(file_in)
        print '\nFile loaded'

    return seg_file


def save_or_load_objects(obj, folder_path, file_name, write):
    path = os.path.join(folder_path, file_name + '.pkl')

    objects_file = None
    if write:
        # write results to a file
        with open(path, 'wb') as file_out:
            pickle.dump(obj, file_out, -1)
        print '\nFile saved at: ', path
    else:
        # read and return results from a file
        with open(path, 'rb') as file_in:
            objects_file = pickle.load(file_in)
        print '\nFile loaded'

    return objects_file


def obia(img_samples, t_dataset):
    print '\n\nObject-Based Image Analysis\n'

    def quick_seg(image):
        k_size = 5
        max_d = 2
        ratio = 0.35
        segments_data = quickshift(image, kernel_size=k_size, max_dist=max_d, ratio=ratio, convert2lab=False)
        n_segments = len(np.unique(segments_data))

        # plot results
        name = 'quick_k{k}_d{d}_r{r}'.format(k=k_size, d=max_d, r=ratio)
        input_seg_plot(segments_data, n_segments, segmentation_path_quickshift, name)
        # store segments
        save_or_load_segments(segments_data, segmentation_path_quickshift, 'segments_' + name, True)

        return segments_data

    def slic_seg(image):
        n_seg = 8000
        segments_data = slic(image, n_segments=n_seg, compactness=0.1, max_iter=10, sigma=0, convert2lab=False,
                             slic_zero=False)
        n_segments = len(np.unique(segments_data))

        # plot results
        name = 'slic_n{n}'.format(n=n_seg)
        input_seg_plot(segments_data, n_segments, segmentation_path_slic, name)
        # store segments
        save_or_load_segments(segments_data, segmentation_path_slic, 'segments_' + name, True)

        return segments_data

    def training_segments_plot():

        def plot(segments_data):
            # use predefined colours
            custom_colours = COLOURS
            colour_map = colors.ListedColormap(custom_colours)
            plt.imshow(segments_data, cmap=colour_map)
            # for 4 different features + NoData
            plt.colorbar(ticks=[0, 1, 2, 3, 4])
            plt.tight_layout()

            path = os.path.join(segmentation_path, 'train_seg_img' + '.png')
            plt.savefig(path)
            print 'Fig saved in %s' % path

            plt.clf()

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

    def input_seg_plot(img, n_seg, file_path, plt_title):
        # create random colours for all segments
        plt_colours = colors.ListedColormap(np.random.rand(n_seg, 3))
        plt.imshow(img, cmap=plt_colours, interpolation='none')
        # to fit properly with all labels
        plt.tight_layout()

        path = os.path.join(file_path, 'plots', plt_title + '.png')
        plt.savefig(path)
        print 'Fig saved in %s' % path

        plt.clf()

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

    def create_objects():
        objects = []
        object_labels = []
        for s in n_segments:
            segments = rescale_img[segmentation == s]
            # compute statistics for each object
            objects.append(compute_statistics(segments))
            # assign a class label to each object
            object_labels.append(s)

        save_or_load_objects(objects, segmentation_path, 'objects', True)
        save_or_load_objects(object_labels, segmentation_path, 'object_labels', True)

        print("Created %i objects" % len(objects))

        return objects, object_labels

    # segmentation tools work with values between 0 and 1, hence image needs to be rescaled
    rescale_img = exposure.rescale_intensity(img_samples)

    # either load existing segments or create new ones
    load_segments = True
    seg_algo = 'quick'
    segmentation = None
    if load_segments:
        segmentation = save_or_load_segments(None, segmentation_path_quickshift, 'segments_quick_k7_d3_r0.35', False)
    else:
        if seg_algo == 'quick':
            segmentation = quick_seg(rescale_img)
        else:
            segmentation = slic_seg(rescale_img)
    n_segments = np.unique(segmentation)
    print 'No. of segments: %i' % len(n_segments)

    # no of class labels of training data, 0 is marked as NoData
    labels = np.unique(t_dataset)[1:]
    print 'No. of class labels: %i' % len(labels)

    # check which segments are in shapefile areas
    training_segments = {}
    for l in labels:
        # part of the same class if training labels and segment labels match
        class_segments = segmentation[t_dataset == l]
        # set builds unordered collection of unique objects - no duplication
        training_segments[l] = set(class_segments)
        ### BEFORE SUBMIT DETELE PRINT STATEMENT
        print("Segments in class %i: %i" % (l, len(training_segments[l])))

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
        print 'Training_true: ', len(training_segments_true[i])
        print 'Diff: ', (len(class_segments) - len(training_segments_true[i]))
        i += 1

    # assign back to the old name
    training_segments = None
    training_segments = training_segments_true
    training_segments_true = None

    # create and plot training segments
    training_segments_plot()

    # create objects / training data
    load_objects = True
    objects = None
    object_labels = None
    if load_objects:
        objects = save_or_load_objects(objects, segmentation_path, 'objects', False)
        object_labels = save_or_load_objects(object_labels, segmentation_path, 'object_labels', False)
    else:
        objects, object_labels = create_objects()

    print 'Done'


if __name__ == "__main__":

    image_dataset = import_image_data(image_data_path, input_data_name)

    # image attributes
    input_image_samples = image_dataset[0]
    rows = image_dataset[1]
    cols = image_dataset[2]
    number_of_bands = image_dataset[3]
    geo_transform = image_dataset[4]
    projection = image_dataset[5]

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
    obia(input_image_samples, training_dataset)

    ### TRAINING
    rf = RandomForestClassifier(n_estimators=no_estimators, criterion='gini', bootstrap=True, max_features='auto',
                                n_jobs=-1, verbose=True, oob_score=True, class_weight='balanced')

    knn = KNeighborsClassifier(n_neighbors=no_estimators, weights='distance', algorithm='auto', metric='minkowski',
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
    print '\nCross val score: {}'.format(cvs)

    print '\nTraining...'
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

        # total number of samples or pixels in an image (1000*1000)
        tot_samples = rows * cols
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

    write_scores(write_s, '\nCross Validation Scores / Classification Accuracy Scores\n', accuracy_score_name)

    print '\nDONE'
