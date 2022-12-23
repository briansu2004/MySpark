# MySpark

Spark is written in Scala.

Apache Spark is a unified analytics engine for processing large volumes of data. It can run workloads 100 times faster and offers over 80 high-level operators that make it easy to build parallel apps. Spark can run on Hadoop, Apache Mesos, Kubernetes, standalone, or in the cloud, and can access data from multiple sources.

## Spark Streaming

### Spark Streaming With Scala

- Spark Structured Streaming
  - Streaming DataFrames
  - Streaming Aggregations
  - Streaming Joins
  - Streaming Datasets
- Low-Level Spark Streaming with DStreams
  - DStreams
  - DStreams Transformations
- Spark Streaming Integrations
  - Kafka & Structured Streaming
  - Kafka & DStreams
  - JDBC with PostgreSQL
  - Cassandra
- Advanced Spark Streaming
  - Event Time Windows
  - Processing Time Windows
  - Watermarking

### Real-time Stream Processing applications using Apache Spark

- Spark Structured Streaming
  - Streaming APIs: DStreams vs Structured Streaming
  - Stream processing model
  - Streaming sources, sinks and output mode
  - Fault tolerance and restarts
- Spark Streaming with Kafka
  - Streaming from Kafka source
  - Working with Kafka sinks
  - Multi-query Streams Application
  - Kafka serialization and deserialization for Spark
  - Creating Kafka AVRO sinks
  - Working with Kafka AVRO source
- Windows and Aggregates
  - Stateless vs Stateful transformations
  - Event time and Windowing
  - Tumbling Window aggregate
  - Watermarking your windows
  - Watermark and output modes
  - Sliding Windows
- Stream Processing and Joins
  - Join Stream to static source
  - Join Stream to another Stream
  - Streaming Watermark
  - Streaming Outer Joins

## A good example for Spark to beat MapReduce

A credit card company needs to handel tons of fraud data in realtime.

With Hadoop & MapReduce

- need to install Hive/Impala for SQL

- need to install Storm for realtime processing

- need to install mahout for machine learning

With Spark

- Spark streaming for data streaming process

- Spark MLib for machine learning

- 100x faster than MapReduce

## Spark mini project

Perform the following analytics on the data

- Show the number of students in the file.
- Show the total marks achieved by Female and Male students
- Show the total number of students that have passed and failed. 50+ marks are required to pass the course.
- Show the total number of students enrolled per course
- Show the total marks that students have achieved per course
- Show the average marks that students have achieved per course
- Show the minimum and maximum marks achieved per course
- Show the average age of male and female students

## Partitioning considerations

- Number of executors

- RAM

- CPU

## Keywords

Spark Structured API

Spark SQL

Spark Streaming

Spark Core

RDD

Sprak-shell

PySpark

SparkContext

Scala / Python: Dynamic Type Inference

But the best practice is - don't let Scala to guess the data type.

Sprak Streaming vs Apache Storm vs Apache Flink vs Kafka Streams

Micro batch architecture

Dynamic Resource Allocation

Databricks

RDD

Spark Configuration Object (settings and configuration)

```dos
Create Spark configuration Object 
-> Initialize the SparkContext object with Spark configuration object 
-> Spark Session
```

## The Spark Ecosystem

Spark Core

It provides the base functionality for the components on top of it, like scheduling and monitoring of jobs in a cluster and handling faults. Though Spark is developed in the Scala programming language, it also provides an API for other languages like R, SQL, Python, and Java. These languages, especially R and Python, are the most popular in data science, which is one of the reasons Spark is so popular.

Spark SQL + Dataframes

This is a module that provides a structured data processing interface via SQL, which is a language used for communicating with a database. It also provides the Dataframe format, which is used to structure data into columns and rows.

Streaming

In some applications, we need a result from a data processing within a specific time limit, failing which it becomes useless. For instance, to detect fraud in a credit card transaction. This module can be used in situations we need real-time performance.

ML/Mlib

Machine Learning (ML) is everywhere today, from recommendations on sites like Netflix and LinkedIn to advanced technologies like self-driving cars. This module provides state of the art algorithms to learn from data and make models with it, to be able to make predictions.

GraphX

A module that can handle graph-structured data at scale. One can think of visualizing Facebook relations or analyzing flight data.

## Why Is Spark so Popular?

There are a couple of reasons why Spark is so popular and why there is a massive demand for people with Spark skills -

Speed

The main benefits is the fast processing of big datasets. It is fast because it processes the data in memory, and it uses cluster technology. This means that a particular task can be divided into sub-tasks, and these sub-tasks are computed on different hosts in a cluster. This clustering technology is not new as it’s also used in Hadoop MapReduce, sometimes called the predecessor of Spark. While in Hadoop MapReduce, all tasks are performed on disk, Spark performs them in-memory, which is a lot faster. Spark is nearly 100 times as fast as Hadoop MapReduce. A couple of years ago, it broke the record for sorting a petabyte. It is also possible to use Spark on your local machine without using a cluster.  I have used it this way in my last project where I had to predict click-through rates for a marketing company. Loading the client’s data and building the prediction model took about 15 minutes at first. When I added Spark, which can utilize all cores on my machine, it took only 3 minutes.

Ease of Use

It provides support for the main languages used in data processing like Java, Scala, R, and Python. The documentation is pretty good, and it’s relatively easy to create a simple application in your preferred language.  It also provides a way to use it interactively, which is handy to experiment with before you write your program.

Supports Many Use Cases

It is a complete framework that supports multiple use cases. Ranging from Machine Learning to Stream processing and Graph processing, Spark has quite a lot of functionality available that gets you up and running quickly.

Integration with Other Technologies

Spark can run on different cluster technologies like the Hadoop file-system, YARN, and Amazon web services (AWS). AWS, which has been supporting Spark for some time, now, has the advantage that you don’t have to set up and maintain a cluster yourself, saving you valuable time.

## Images

![](image/README/spark_components_01.png)

![](image/README/spark_components_02.png)

![](image/README/spark_architecture_01.png)

![](image/README/spark_shell_01.png)

![](image/README/pyspark_01.png)

![](image/README/scala_iwq_lazy.png)

![](image/README/stream_processing_workflow.png)

![](image/README/streaming_context.png)

![](image/README/processing_spark_dstreams.png)

![](image/README/streaming_source_and_destination.png)

![](image/README/spark_rdd.png)

![](image/README/rdd_workflow.png)

![](image/README/type_of_rdd.png)

![](image/README/ParallelCollectionRDD.png)

![](image/README/ShuffledRDD.png)

![](image/README/PairRDD.png)

![](image/README/HadoopRDD.png)

![](image/README/operations_on_RDD.png)

![](image/README/operations_on_RDD_transformation.png)

![](image/README/list_of_rdd_transformations.png)

![](image/README/spark_example_01.png)

![](image/README/spark_example_02.png)

![](image/README/spark_example_03.png)

![](image/README/spark_dataframe.png)

![](image/README/spark_dataframe_example_01.png)

![](image/README/spark_dataframe_example_02.png)

![](image/README/spark_sql_context.png)

![](image/README/spark_dataframe_example_03.png)

![](image/README/spark_dataframe_example_04.png)

![](image/README/spark_dataframe_example_05.png)

![](image/README/spark_dataframe_example_06.png)

![](image/README/spark_dataframe_example_07.png)

![](image/README/spark_dataframe_example_08.png)

![](image/README/spark_dataframe_example_09.png)

## Keywords + Code

Spark Structured API

```dos
org.apache.spark.sql.SparkSession.Builder
def master(master: String): SparkSession.Builder
Sets the Spark master URL to connect to, such as "local" to run locally, "local[4]" to run locally with 4 cores, or "spark://master:7077" to run on a Spark standalone cluster.

org.apache.spark.sql.SparkSession.Builder
def getOrCreate(): SparkSession
Gets an existing SparkSession or, if there is no existing one, creates a new one based on the options set in this builder.
This method first checks whether there is a valid thread-local SparkSession, and if yes, return that one. It then checks whether there is a valid global default SparkSession, and if yes, return that one. If no valid global default SparkSession exists, the method creates a new SparkSession and assigns the newly created SparkSession as the global default.
In case an existing SparkSession is returned, the non-static config options specified in this builder will be applied to the existing SparkSession.

e.g.

  // the entry point to the Spark structured API
  val spark = SparkSession.builder()
    .appName("Spark Recap")
    .master("local[2]")
    .getOrCreate()
```

DF

```dos
org.apache.spark.sql.SparkSession
def read: DataFrameReader
Returns a DataFrameReader that can be used to read non-streaming data in as a DataFrame.
sparkSession.read.parquet("/path/to/file.parquet")
sparkSession.read.schema(schema).json("/path/to/file.json")

e.g.
  // read a DF
  val cars = spark.read
    .format("json")
    .option("inferSchema", "true")
    .load("src/main/resources/data/cars")


org.apache.spark.sql.Dataset
@varargs
def select(cols: Column*): sql.DataFrame
Selects a set of column based expressions.
ds.select($"colA", $"colB" + 1)

e.g.

  // select
  val usefulCarsData = cars.select(
    col("Name"), // column object
    $"Year", // another column object (needs spark implicits)
    (col("Weight_in_lbs") / 2.2).as("Weight_in_kg"),
    expr("Weight_in_lbs / 2.2").as("Weight_in_kg_2")
  )

```
