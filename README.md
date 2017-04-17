Mapping Hedgerows Using Machine Learning



How to use the classification.py script


The main file, classification.py, is a simple image classification script supporting both object-based and pixel-based approaches. The user can choose between RandomForests and k-Nearest Neighbours classification algorithms. Furthermore, if an object-based approach is chosen, a user can specify which segmentation algorithm to use. There is also an option for k-fold cross-validation.

As the classification.py script relies heavily on the external libraries, the created executable file exceeded the 10MB size constraint. A classification.py script and a list of dependencies were submitted instead.

To run the classification.py script, Python and dependencies from the 'hedgerows-req' file need to be installed. As some problems may occur if using the default package installer (pip) to install the 'gdal' and 'libgdal' libraries, I suggest downloading and installing the required packages with the Anaconda scientific platform for python.

Anaconda can be obtained on the following site.
https://www.continuum.io/downloads

Most of the packages required are already included in the default Anaconda distribution. Additional packages can be installed with a command:
conda install package_name

In addition, classification.py script has to be run from the Anaconda prompt.


Input Data

The classification.py script works with .tif image file data. Again, the data required exceeded the 10MB size constraint.

All required data can be downloaded from the following site:
https://emckclac-my.sharepoint.com/personal/k1461612_kcl_ac_uk/_layouts/15/guestaccess.aspx?folderid=1be6c536e229d4feaa59dd43e98ffa2d9&authkey=AQWrwEn4eNpubHJf1eHG3fA

The ‘data’ folder contains:
-	‘input_Clip.tif’ – the input image for classification
-	‘Training_image.tif’ – the training data for classification
-	‘segments_quick_k7_d3_r0.35.pkl’ – sample segmentation of the input_Clip.tif with Quickshift algorithm
-	‘segments_slic_n8000.pkl’ – sample segmentation of the input_Clip.tif with Slic algorithm
-	‘objects.pkl’ and ‘object_labels.pkl’ – objects used as training data in the object-based classification, created from the segments_quick_k7_d3_r0.35.pkl segmentation file


Script Guide

When the classification script is started, it will prompt the user for the following inputs.

1.	Import Input Image File – Select a .tif image file on which to perform classification

2.	Import Training Data File – Select a .tif image file which serves as the training data

3.	Select Output Folder - the location where you wish to store the classification output (sample 'output' folder has been included in the script directory).

4.	Classification configuration
a.	Classifier – ‘random-forests’ for RandomForests algorithm, ‘knn’ for k-Nearest Neighbours algorithm [default=random-forests]
b.	Number Of Estimators – number of trees in RandomForests or number of neighbours in k-Nearest Neighbours, number between 10 and 999 [default= 100 for RandomForests and 10 for k-Nearest Neighbours]
c.	Image Analysis Type – ‘object-based’ for object approach, ‘pixel-based’ for pixel approach [default=pixel-based]
d.	Segmentation Algorithm – (only if chosen the object-based approach) ‘quickshift’ for Quickshift algorithm, ‘slic’ for Slic algorithm [default=quickshift]
e.	Load Segments - (only if chosen the object-based approach) ‘True’ or ‘False’ [default=False]
f.	Load Objects - (only if chosen the object-based approach) ‘True’ or ‘False’ [default=False]
g.	Cross-Validation – ‘True’ to perform k-fold cross-validation or ‘False’ to not [default=False]
h.	k-splits – (only if chosen the cross-validation) the k parameter in k-fold cross-validation, number between 2 and 99 [default=10]

5.	If the object-based approach is chosen:
a.	If Load Objects is True – Select Folder with Segmentation Objects
    Segmentation objects are automatically created when fresh segmentation is run. They are used to skip the step of expensive computation of object statistics.
    Select folder containing ‘objects.pkl’ and ‘object_labels.pkl’ files.
b.	If Load Segments is True – Import Segmentation File
    Segmentation file is automatically created when fresh segmentation is run. It is used to skip the step of expensive input image segmentation.
    Select segmentation file e.g. ‘segments_quick_k7_d3_r0.35.pkl’.

If ‘Import Input Image File’ path or ‘Import Training Data File’ path or ‘Select Output Folder’ path is not specified, the application will close with exit code 1.
If ‘Classification Configuration’ is not specified, the application will continue running with default parameters.
If ‘Select Folder with Segmentation Objects’ path or ‘Import Segmentation File’ path is not specified, the application will continue running with the default segmentation process.