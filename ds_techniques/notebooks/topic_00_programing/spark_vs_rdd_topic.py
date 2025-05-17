import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("MyApp").getOrCreate()


from pyspark.sql import SparkSession
import numpy as np

def explain_rdd_basics():
    """
    Demonstrates basic RDD (Resilient Distributed Dataset) operations in PySpark
    """
    # Create a Spark session
    spark = SparkSession.builder.appName("RDD_Examples").getOrCreate()
    sc = spark.sparkContext

    # 1. Creating RDDs
    print("1. Creating RDDs:")
    # From a list
    numbers_rdd = sc.parallelize([1, 2, 3, 4, 5])
    # From a text file
    # text_rdd = sc.textFile("example.txt")  # Uncomment to read from file
    
    # 2. Basic Transformations
    print("\n2. Basic Transformations:")
    # Map: Apply function to each element
    squared_rdd = numbers_rdd.map(lambda x: x**2)
    print("Squared numbers:", squared_rdd.collect())
    
    # Filter: Keep elements that satisfy condition
    even_rdd = numbers_rdd.filter(lambda x: x % 2 == 0)
    print("Even numbers:", even_rdd.collect())
    
    # 3. Key-Value RDD Operations
    print("\n3. Key-Value RDD Operations:")
    key_value_rdd = sc.parallelize([(1, "a"), (2, "b"), (1, "c")])
    
    # GroupByKey
    grouped = key_value_rdd.groupByKey().mapValues(list)
    print("Grouped by key:", grouped.collect())
    
    # ReduceByKey
    word_counts = sc.parallelize([("apple", 1), ("banana", 1), ("apple", 1)])
    reduced = word_counts.reduceByKey(lambda a, b: a + b)
    print("Word counts:", reduced.collect())
    
    # 4. Actions
    print("\n4. RDD Actions:")
    # Count
    print("Count:", numbers_rdd.count())
    # Sum
    print("Sum:", numbers_rdd.sum())
    # Collect
    print("Collected data:", numbers_rdd.collect())
    
    # 5. Advanced Operations
    print("\n5. Advanced Operations:")
    # Flatmap
    text_rdd = sc.parallelize(["Hello World", "How are you"])
    words_rdd = text_rdd.flatMap(lambda x: x.split())
    print("Words after flatMap:", words_rdd.collect())
    
    # Distinct
    duplicate_rdd = sc.parallelize([1, 1, 2, 2, 3, 3, 3])
    distinct_rdd = duplicate_rdd.distinct()
    print("Distinct values:", distinct_rdd.collect())
    
    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    explain_rdd_basics()
