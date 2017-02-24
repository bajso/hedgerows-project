_author_ = "gregor"

from sklearn.datasets import load_iris

iris = load_iris()

print iris.data

s = "hedgerows"
s.lower()

print s + " are quite {}".format("fucking amazing")
