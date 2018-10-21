from pyspark                  import SparkContext
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy                    import array

print("rdd script")

sc   = SparkContext()
data = sc.textFile("bezdekIris.csv")

parsedData = data.map(lambda line: array([float(x) for x in [line.split(',')]))
parsedData.cache()

print("after map, before kmeans train")

clusters = KMeans.train(parsedData, 4, maxIterations=20)

print("Within Set Sum of Squared Error = " + str(clusters.computeCost(parsedData)))

for center in clusters.clusterCenters: 
	print(center)

sc.stop()

