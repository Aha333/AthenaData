# transform vs action 
Spark 中，平行于 行动操作（Action） 的概念是 转换操作（Transformation）。这两者一起构成了 Spark 中的操作模型。
Transform 也包含了两大类
1. mapping 比如 map(), select()
2. non mapping 比如 filter(), groupby()
3. 聚合操作， 比如 
reduceByKey对每个 key 的值进行聚合，生成一个新的 RDD；
groupByKey： 将数据按 key 分组，并返回一个包含所有值的集合
aggregateByKey：基于每个 key 对数据进行聚合操作。

有意思的是： 在 Spark 中，reduce 是一个 行动操作（Action），而 reduceByKey 则是一个 转换操作（Transformation）。reduce() 会触发 计算并返回结果。reduceByKey() 会 创建一个新的 RDD，表示聚合后的数据，但并不会立即执行。


# pyspark
* 'F.col()' is referencing existing col.
* 'F.col()' is one of Spark's lazy evaluation mechanism. When you pass F.col("column_name"), Spark doesn’t immediately compute the result; instead, it builds a logical plan that Spark will later execute when an action (like collect(), show(), or write()) is triggered.
* On the other hand, withColumn() is a method used to create a new column (or modify an existing column). It expects the first parameter to be the name of the new column (a string) and the second parameter to be an expression (which can be a Column object).
* wrong: df.select(F.col('age')) + 10 vs. right way -> df.select(F.col('age') + 10). 注意 df.select()返回的是一个DataFrame，所以是不能直接加10 的。
* Column expression的概念   In the PySpark execution engine, the column doesn’t actually hold data. Instead, it represents an operation that will be performed on the data when the DataFrame is evaluated. SQL Expression Strings: The Column object can also be thought of as an expression that's internally translated into a corresponding SQL expression. For instance:
F.col('age') can be understood as a reference to a column.
F.col('age') > 30 might translate to a SQL expression like age > 30.
* 在 PySpark 中，Column 就是一个 wrapper，它封装了列的操作。例如，当你执行 F.col('age') + 10 时，这个表达式并没有立即执行计算，而是创建了一个新的 Column 对象，这个对象描述了“列 age 的每个值加 10”的操作。Column 对象本身只是一个描述操作的表达式，而不是实际的数据
* 同样的， 聚合表达式 就是通过聚合函数（如 F.mean(), F.sum() 等）返回的 Column 对象，这些对象描述了如何对数据进行聚合计算。
* 总之， column 对象， 就是一个描述操作的表达式expression，而不是实际的数据。这就是为什么表达式可以+10的原因F.col('age') + 10 是一个valid expression

# PySpark DataFrame 的设计哲学
*  它没有像 Pandas 中的 index 单一索引 或 行标签, 为了高效地处理大量数据，PySpark 的设计避免了引入行索引的概念. 不保证行的顺序. 如果一定要给一个index， 一个方法是可以是使用 monotonically_increasing_id() 来创建唯一标识符；df_with_index = df.withColumn("index", monotonically_increasing_id()；但是请注意：生成的 ID 可能不是顺序递增的， 而且不同的 Spark 版本对 ID 的生成可能会有所不同，因此它不完全是一个递增的、连续的数字。另一个方法是df.withColumn("index", row_number().over(window_spec))
* Column 代表了 DataFrame 中的一列数据，但它并不仅仅是一个存储数据的容器，而是一个表达式（Expression），它封装了对数据列的各种操作和计算。Column 是一个表达式, 可以是简单的列引用，也可以是复杂的计算、聚合或转换操作。Column 是懒计算的：在 PySpark 中，操作 Column 时，并不会立即执行计算，而是会构建一个查询计划。当你触发一个 行动（如 .show()、.collect() 等）时，计算才会实际执行。
* 一个有趣的事情 df.select(expression) 和 pandasdf[] 操作符很像， 但是更灵活可能。 sparkdf.select(column_expression) 的column expression，可以是简单的 F.col(col_str), 也可以是复杂的， 比如df.select(F.col('age') + 10)  # 对 'age' 列进行操作，返回新的 DataFrame。 甚至更复杂的

col_expression = F.when(F.col('age') > 30, 'Senior').otherwise('Junior').alias('age_group')
df.select(col_expression)
这里就非常有意思了！这里初看以为是一个filter呢， 其实不是！
* Pandas 中的 df[] 既可以用作 select 也可以用作 filter，而在 PySpark 中，select 和 filter 是两个不同的函数，各自有不同的作用。

* 既然没有index， 那么groupby的col最后是一个普通的col了。 如果要对这个改名字， 有两种办法
* 1. 使用 withColumnRenamed()
* 2. 使用 alias()
具体如下

result = df.groupBy(F.col('category').alias('new_category')).agg(
    F.sum('value').alias('total_value'),
    F.avg('value').alias('avg_value')
)  # alias directly when group by 
-
result = df.groupBy('category').agg(
    F.sum('value').alias('total_value'),
    F.avg('value').alias('avg_value')
)
result = result.withColumnRenamed('category', 'new_category') # withColumnRenamed 


# df.withcolumn() 和 pandas 的 df.assign() 很像， 
* 但是注意， 后者是inplace的， 前者不是。
* dfpandas.assign(new_col=df['A'] + df['B'])	
* df.withColumn("new_col", col("A") + col("B"))

# sparksql

* Spark SQL 和 PySpark 内置操作可以互换使用：从功能上来说，PySpark 的内置操作和 Spark SQL 是可以互换的，因为它们都依赖于 Spark SQL 引擎的执行。但在实际开发中，你可能会选择其中之一，取决于你的编程风格、业务需求以及优化需求。

因此，无论是使用 DataFrame API 还是 SQL 查询，背后执行的计算计划是一样的，Spark SQL 会尽可能优化查询，使其高效执行。

如果你习惯编写 Python 代码，且需要进行复杂的数据操作，PySpark 内置操作可能更适合。
如果你更熟悉 SQL 或者需要执行更复杂的 SQL 查询，使用 Spark SQL 会更直观。

from pyspark.sql import functions as F

df.filter(df.age > 30).groupBy("gender").agg(F.avg("purchase_amount").alias("avg_purchase"))

vs

SELECT gender, AVG(purchase_amount) as avg_purchase
FROM df
WHERE age > 30
GROUP BY gender

## 使用 Spark SQL 查询 DataFrame
如果你已经有了一个 Spark DataFrame，并且希望使用 Spark SQL 对这个 DataFrame 进行操作，你可以将 DataFrame 注册为临时视图（temporary view），然后通过 Spark SQL 查询它。
注册 DataFrame 为临时视图的操作本身不会影响性能。临时视图仅是一个 SQL 查询时的抽象，底层数据并没有发生任何变化。它仅仅是为了让你通过 SQL 查询方便地访问 DataFrame。
使用临时视图或使用 DataFrame API 都会被转换为相同的查询计划，最后通过 Spark SQL 执行。

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

* 初始化 Spark 会话
spark = SparkSession.builder.appName("SparkSQLExample").getOrCreate()

假设我们有一个 DataFrame df
df = spark.createDataFrame([
    ("M", 100),
    ("F", 150),
    ("M", 200),
    ("F", 120),
    ("M", 50)
], ["gender", "purchase_amount"])

* 1. 注册 DataFrame 为临时视图
df.createOrReplaceTempView("sales_data")

* 2. 使用 Spark SQL 查询临时视图
result = spark.sql("""
    SELECT gender, AVG(purchase_amount) AS avg_purchase
    FROM sales_data
    GROUP BY gender
""")

* 3. 查看查询结果
result.show()

如果临时视图的查询涉及到非常大的数据集，尤其是没有正确分区的情况下，可能会导致某些节点的计算压力过大，影响性能。为了提高效率，建议合理地对数据进行分区，或者使用 cache() 和 persist() 方法来缓存一些常用数据。


注意了 在这个一一下 df.select()类似于sql里面的select。
# 再看 spark select vs sql select, 还有column expression
df.select(
    F.col("age") + 10,  # age 列加上 10
    F.when(F.col("age") > 18, "adult").otherwise("child").alias("age_group")  # 生成新的列
)
VS 对应的就是
SELECT age + 10, 
       CASE WHEN age > 18 THEN 'adult' ELSE 'child' END AS age_group
FROM df;

# 关于collect操作
df.select(...).collect() 返回的是一个 Python 列表，其中每个元素是一个 Row 对象。
Row 对象 可以通过字段名称或者位置索引来访问其内容。！！！有点意思！！
like Row(id=1, name='Tom')

这里Row 对象并不是 Python 的内置数据类型，它是 PySpark 中的一种特殊数据类型，定义在 pyspark.sql 模块中，用来表示 Spark DataFrame 中的一行数据。可以把它理解为一种类，它类似于 Python 的 tuple 或 namedtuple，但具备一些额外的功能。

from pyspark.sql import functions as F

* 按照 'gender' 列进行分组，计算每个分组的 25%, 50%, 75% 分位数。在提取 50% 分位数（即中位数）
result = df.groupby('gender').agg(
    F.percentile_approx('age', [0.25, 0.5, 0.75]).alias('percentiles')
) 
result = result.withColumn('median_age', F.col('percentiles')[1])


如果要取出来这个数字， 我们就可以collect了
percentiles = df.select(
    F.percentile_approx('age', [0.25, 0.5, 0.75]).alias('percentiles')
).collect()[0]['percentiles']

q1 = percentiles[0]
q2 = percentiles[1]
q3 = percentiles[2]

# UDF
虽然 Spark SQL 进行了优化，但 UDF 的执行是 Python 环境中的逐行计算，效率较低，因此应该尽量避免 UDF。
# cache or persist

第一，甚至 df = spark.read.csv("path/to/data.csv") 也没有读取数据， 只是一个DataFrame 的 DAG plan， 是惰性操作。
如果不cache， 那么， 每次你调用 df.show()，都会重新读取 CSV 文件！如果你调用了 cache()，那么 Spark 会将数据缓存到内存中。这样在第一次 df.show() 执行时，它会读取 CSV 文件并将其存储在内存中。之后如果你再次使用 df.show() 或其他的转化操作，Spark 会直接从缓存中读取数据，而不是重新读取 CSV 文件。

第二， 就算标记缓存cache， 也不是真的执行了！标记该 DataFrame 或 RDD 为缓存状态。
真正的计算发生在你触发一个动作时

第三，cache() 是 persist() 的简化版，默认使用 StorageLevel.MEMORY_AND_DISK 存储方式




# RDD vs DataFrame
RDD 的元素和 DataFrame 中的 Row 都是数据集的最小单位，但 Row 是 DataFrame 特有的结构化表示方式，而 RDD 元素则可以是任意类型的数据，不一定有固定的结构。

# RDD

RDD 是 DataFrame 的底层实现，但 DataFrame 提供了更多的 SQL 风格的操作。
RDD 是一个 数据集，它可以包含多列数据，但它本身并没有明确的列和行的结构。它只是一个元素的集合，通常每个元素是一个 Python 对象（如元组、字典、列表等）。

在 DataFrame 中实现 RDD 时，每一行数据实际上就是 RDD 的一个元素。

rdd = spark.sparkContext.parallelize([(1, "Alice", 28), (2, "Bob", 34)])
df = spark.createDataFrame([(1, "Alice", 28), (2, "Bob", 34)], ["id", "name", "age"])

# map & reduce
* map() 是用来对 RDD 中的每个元素应用函数，生成一个新的 RDD。操作是针对 RDD 中的 每一个元素 进行处理的。
* map() 不会改变 RDD 中元素的数量（即每个输入元素都会对应一个输出元素），它只会根据提供的函数转换每个元素。
* reduce() 是用来对 RDD 中的所有元素进行聚合，通常是通过传入一个二元操作符来实现。在 PySpark 中，确实 必须接受两个参数，这两个参数是 两两合并的元素。通过一个二元（接受两个参数）的 聚合函数 来逐步合并 RDD 中的所有元素，最终得到一个单一的输出值。
* 大多数情况下你不需要自己实现 map 和 reduce，因为 PySpark 的 DataFrame API 和 RDD API 已经提供了丰富的操作，可以让你方便地进行数据处理和聚合，而无需手动实现这些操作。


# 下面是底层逻辑 

df_selected = df.select("id", "name")
对应
rdd = df.rdd.map(lambda row: (row.id, row.name))

df_filtered = df.filter(df["salary"] > 1500)
对应
rdd = df.rdd.filter(lambda row: row.salary > 1500)


df_grouped = df.groupBy("id").agg({"salary": "sum"})
对应
grouped_rdd = df.rdd.map(lambda row: (row.id, row.salary))
aggregated_rdd = grouped_rdd.reduceByKey(lambda x, y: x + y)


df_joined = df1.join(df2, "id")
对应
rdd1 = df1.rdd.map(lambda row: (row.id, row.name)) #这里第一个元素是key
rdd2 = df2.rdd.map(lambda row: (row.id, row.salary))
joined_rdd = rdd1.join(rdd2)

