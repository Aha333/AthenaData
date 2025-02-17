from typing import Tuple
import time
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

def create_spark_session(app_name: str = "PySpark Practice") -> SparkSession:
    """Create or get a Spark Session"""
    return SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()

def generate_sample_data(n_rows: int = 1000000) -> pd.DataFrame:
    """Generate sample data for practice"""
    np.random.seed(42)
    
    return pd.DataFrame({
        'user_id': range(n_rows),
        'age': np.random.randint(18, 80, n_rows),
        'gender': np.random.choice(['M', 'F'], n_rows),
        'purchase_amount': np.random.normal(100, 30, n_rows),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows),
    })

def compare_performance(
    pandas_func, 
    spark_func, 
    pandas_df: pd.DataFrame, 
    spark_df = None
) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """
    Compare performance between Pandas and PySpark operations
    
    Parameters:
    -----------
    pandas_func : callable
        Function that operates on pandas DataFrame
    spark_func : callable
        Function that operates on spark DataFrame
    pandas_df : pd.DataFrame
        Input pandas DataFrame
    spark_df : pyspark.sql.DataFrame, optional
        Input spark DataFrame. If None, will be created from pandas_df
        
    Returns:
    --------
    Tuple[float, float, pd.DataFrame, pd.DataFrame] : 
        (pandas_time, spark_time, pandas_result, spark_result)
    """
    # Pandas timing
    start = time.time()
    pandas_result = pandas_func(pandas_df)
    pandas_time = time.time() - start
    
    # Get or create Spark DataFrame
    if spark_df is None:
        spark = create_spark_session()
        spark_df = spark.createDataFrame(pandas_df)
    
    # PySpark timing
    start = time.time()
    spark_result = spark_func(spark_df)
    # Convert to Pandas for comparison (if it's not already collected)
    if hasattr(spark_result, 'toPandas'):
        spark_result = spark_result.toPandas()
    spark_time = time.time() - start
    
    return pandas_time, spark_time, pandas_result, spark_result

# ===================== BEGINNER LEVEL =====================

def basic_operations_example(n_rows: int = 1000000):
    """
    Demonstrate basic operations in both Pandas and PySpark:
    - DataFrame creation
    - Basic filtering
    - Column selection
    - Simple aggregation
    """
    print("=== BEGINNER LEVEL EXAMPLES ===")
    
    # Generate sample data
    pdf = generate_sample_data(n_rows)
    spark = create_spark_session()
    sdf = spark.createDataFrame(pdf)
    
    # 1. Basic Filtering
    def pandas_filter(df):
        return df[df['age'] > 30]
    
    def spark_filter(df):
        return df.filter(F.col('age') > 30)
    
    p_time, s_time, p_result, s_result = compare_performance(pandas_filter, spark_filter, pdf, sdf)
    print(f"\nFiltering Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")
    
    # 2. Column Selection
    def pandas_select(df):
        return df[['user_id', 'age', 'purchase_amount']]
    
    def spark_select(df):
        return df.select('user_id', 'age', 'purchase_amount')
    
    p_time, s_time, p_result, s_result = compare_performance(pandas_select, spark_select, pdf, sdf)
    print(f"\nColumn Selection Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")
    
    # 3. Basic Aggregation
    def pandas_agg(df):
        return df.groupby('category')['purchase_amount'].mean()
    
    def spark_agg(df):
        return df.groupBy('category').agg(F.mean('purchase_amount'))
    
    p_time, s_time, p_result, s_result = compare_performance(pandas_agg, spark_agg, pdf, sdf)
    print(f"\nBasic Aggregation Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")

# ===================== INTERMEDIATE LEVEL =====================

def intermediate_operations_example(n_rows: int = 1000000):
    """
    Demonstrate intermediate operations:
    - Window functions
    - Complex aggregations
    - Joins
    """
    print("\n=== INTERMEDIATE LEVEL EXAMPLES ===")
    
    # Generate main data
    pdf1 = generate_sample_data(n_rows)
    spark = create_spark_session()
    sdf1 = spark.createDataFrame(pdf1)
    
    # Generate additional data for joins
    pdf2 = pd.DataFrame({
        'category': ['A', 'B', 'C', 'D'],
        'category_name': ['Electronics', 'Clothing', 'Food', 'Books'],
        'discount_rate': [0.1, 0.2, 0.15, 0.25]
    })
    sdf2 = spark.createDataFrame(pdf2)
    
    # 1. Window Functions
    def pandas_window(df):
        return df.assign(
            avg_by_gender=df.groupby('gender')['purchase_amount'].transform('mean'),
            rank_by_amount=df.groupby('category')['purchase_amount'].rank(method='dense')
        )
    
    def spark_window(df):
        window_gender = Window.partitionBy('gender')
        window_category = Window.partitionBy('category').orderBy(F.desc('purchase_amount'))
        
        return df.select(
            '*',
            F.avg('purchase_amount').over(window_gender).alias('avg_by_gender'),
            F.dense_rank().over(window_category).alias('rank_by_amount')
        )
    
    p_time, s_time, p_result, s_result = compare_performance(
        pandas_window, spark_window, pdf1, sdf1
    )
    print(f"\nWindow Functions Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")
    
    # 2. Joins with aggregation
    def pandas_join(df):
        agg_df = df.groupby('category')['purchase_amount'].agg(['mean', 'count']).reset_index()
        return agg_df.merge(pdf2, on='category', how='left')
    
    def spark_join(df):
        agg_df = df.groupBy('category').agg(
            F.avg('purchase_amount').alias('mean'),
            F.count('purchase_amount').alias('count')
        )
        return agg_df.join(sdf2, on='category', how='left')
    
    p_time, s_time, p_result, s_result = compare_performance(
        pandas_join, spark_join, pdf1, sdf1
    )
    print(f"\nJoins Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")

# ===================== ADVANCED LEVEL =====================

def advanced_operations_example(n_rows: int = 1000000):
    """
    Demonstrate advanced operations:
    - Custom UDFs
    - Complex transformations
    - Performance optimization techniques
    """
    print("\n=== ADVANCED LEVEL EXAMPLES ===")
    
    pdf = generate_sample_data(n_rows)
    spark = create_spark_session()
    
    # 1. Custom Functions/UDFs
    # Pandas: Simple function application
    def calculate_discount(amount, age):
        return amount * (1 - (age/100))
    
    def pandas_udf(df):
        return df.assign(
            discounted_amount=df.apply(
                lambda x: calculate_discount(x['purchase_amount'], x['age']), 
                axis=1
            )
        )
    
    # Spark: Registered UDF
    spark_calculate_discount = F.udf(
        lambda amount, age: float(amount * (1 - (age/100))),
        FloatType()
    )
    
    def spark_udf(df):
        return df.withColumn(
            'discounted_amount',
            spark_calculate_discount('purchase_amount', 'age')
        )
    
    p_time, s_time, p_result, s_result = compare_performance(pandas_udf, spark_udf, pdf, None)
    print(f"\nCustom Functions Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")
    
    # 2. Complex Transformations
    def pandas_complex(df):
        return df.assign(
            age_group=pd.qcut(df['age'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        ).groupby(['age_group', 'gender'])\
          .agg({
              'purchase_amount': ['mean', 'std', 'count'],
              'user_id': 'nunique'
          }).reset_index()
    
    def spark_complex(df):
        # First calculate the percentiles
        percentiles = df.select(
            F.percentile_approx('age', [0.25, 0.5, 0.75]).alias('percentiles')
        ).collect()[0]['percentiles']
        
        # Then use these values to create age groups
        df_with_groups = df.withColumn(
            'age_group',
            F.when(F.col('age') <= percentiles[0], 'Q1')
             .when(F.col('age') <= percentiles[1], 'Q2')
             .when(F.col('age') <= percentiles[2], 'Q3')
             .otherwise('Q4')
        )
        
        # Finally perform the groupBy aggregations
        return df_with_groups.groupBy('age_group', 'gender').agg(
            F.mean('purchase_amount').alias('purchase_mean'),
            F.stddev('purchase_amount').alias('purchase_std'),
            F.count('purchase_amount').alias('purchase_count'),
            F.countDistinct('user_id').alias('unique_users')
        )
    
    p_time, s_time, p_result, s_result = compare_performance(pandas_complex, spark_complex, pdf, None)
    print(f"\nComplex Transformations Performance:")
    print(f"Pandas: {p_time:.2f}s")
    print(f"Spark: {s_time:.2f}s")

if __name__ == "__main__":
    # Run examples with different data sizes
    for n_rows in [1000, 100000, 1000000]:
        print(f"\n{'='*20} Testing with {n_rows} rows {'='*20}")
        basic_operations_example(n_rows)
        intermediate_operations_example(n_rows)
        advanced_operations_example(n_rows) 