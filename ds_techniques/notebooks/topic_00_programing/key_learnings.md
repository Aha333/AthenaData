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





