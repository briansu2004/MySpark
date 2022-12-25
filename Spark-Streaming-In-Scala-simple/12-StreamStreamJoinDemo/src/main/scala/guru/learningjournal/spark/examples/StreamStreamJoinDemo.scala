package guru.learningjournal.spark.examples

import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, expr, from_json, to_timestamp}
import org.apache.spark.sql.streaming.Trigger
import org.apache.spark.sql.types.{StringType, StructField, StructType}

object StreamStreamJoinDemo extends Serializable {
  @transient lazy val logger: Logger = Logger.getLogger(getClass.getName)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local[3]")
      .appName("Stream Stream Join Demo")
      .config("spark.streaming.stopGracefullyOnShutdown", "true")
      .config("spark.sql.shuffle.partitions", 2)
      .getOrCreate()

    val impressionSchema = StructType(List(
      StructField("InventoryID", StringType),
      StructField("CreatedTime", StringType),
      StructField("Campaigner", StringType)
    ))

    val clickSchema = StructType(List(
      StructField("InventoryID", StringType),
      StructField("CreatedTime", StringType)
    ))

    val kafkaImpressionDF = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "impressions")
      .option("startingOffsets", "earliest")
      .load()

    val impressionsDF = kafkaImpressionDF
      .select(from_json(col("value").cast("string"), impressionSchema).alias("value"))
      .selectExpr("value.InventoryID as ImpressionID", "value.CreatedTime", "value.Campaigner")
      .withColumn("ImpressionTime", to_timestamp(col("CreatedTime"), "yyyy-MM-dd HH:mm:ss"))
      .drop("CreatedTime")


    val kafkaClickDF = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", "localhost:9092")
      .option("subscribe", "clicks")
      .option("startingOffsets", "earliest")
      .load()

    val clicksDF = kafkaClickDF.select(
      from_json(col("value").cast("string"), clickSchema).alias("value"))
      .selectExpr("value.InventoryID as ClickID", "value.CreatedTime")
      .withColumn("ClickTime", to_timestamp(col("CreatedTime"), "yyyy-MM-dd HH:mm:ss"))
      .drop("CreatedTime")

    val joinExpr = "ImpressionID == ClickID"
    val joinType = "inner"

    val joinedDF = impressionsDF.join(clicksDF,expr(joinExpr), joinType)

    val outputQuery = joinedDF.writeStream
      .format("console")
      .outputMode("append")
      .option("checkpointLocation", "chk-point-dir")
      .trigger(Trigger.ProcessingTime("1 minute"))
      .start()

    logger.info("Waiting for Query")
    outputQuery.awaitTermination()

  }
}
