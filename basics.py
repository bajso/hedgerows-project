_author_ = "gregor"

from sklearn.datasets import load_iris
from osgeo import gdal


iris = load_iris()

print iris.data

s = "hedgerows"
s.lower()