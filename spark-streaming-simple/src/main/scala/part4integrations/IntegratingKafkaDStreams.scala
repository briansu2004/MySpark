package part4integrations

import java.util

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord}
import org.apache.kafka.common.serialization.{StringDeserializer, StringSerializer}
import org.apache.spark.sql.SparkSession
import org.apache.spark.streaming.kafka010.{ConsumerStrategies, KafkaUtils, LocationStrategies}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object IntegratingKafkaDStreams {

  val spark = SparkSession.builder()
    .appName("Spark DStreams + Kafka")
    .master("local[2]")
    .getOrCreate()

  val ssc = new StreamingContext(spark.sparkContext, Seconds(1))

  val kafkaParams: Map[String, Object] = Map(
    "bootstrap.servers" -> "localhost:9092",
    "key.serializer" -> classOf[StringSerializer], // send data to kafka
    "value.serializer" -> classOf[StringSerializer],
    "key.deserializer" -> classOf[StringDeserializer], // receiving data from kafka
    "value.deserializer" -> classOf[StringDeserializer],
    "auto.offset.reset" -> "latest",
    "enable.auto.commit" -> false.asInstanceOf[Object]
  )

  val kafkaTopic = "sutek"

  def readFromKafka() = {
    val topics = Array(kafkaTopic)
    val kafkaDStream = KafkaUtils.createDirectStream(
      ssc,
      LocationStrategies.PreferConsistent,
      /*
       Distributes the partitions evenly across the Spark cluster.
       Alternatives:
       - PreferBrokers if the brokers and executors are in the same cluster
       - PreferFixed
      */
      ConsumerStrategies.Subscribe[String, String](topics, kafkaParams + ("group.id" -> "group1"))
      /*
        Alternative
        - SubscribePattern allows subscribing to topics matching a pattern
        - Assign - advanced; allows specifying offsets and partitions per topic
       */
    )

    val processedStream = kafkaDStream.map(record => (record.key(), record.value()))
    processedStream.print()

    ssc.start()
    ssc.awaitTermination()
  }

  def writeToKafka() = {
    val inputData = ssc.socketTextStream("localhost", 2023)

    // transform data
    val processedData = inputData.map(_.toUpperCase())

    processedData.foreachRDD { rdd =>
      rdd.foreachPartition { partition =>
        // inside this lambda, the code is run by a single executor

        val kafkaHashMap = new util.HashMap[String, Object]()
        kafkaParams.foreach { pair =>
          kafkaHashMap.put(pair._1, pair._2)
        }

        // producer can insert records into the Kafka topics
        // available on this executor
        val producer = new KafkaProducer[String, String](kafkaHashMap)

        partition.foreach { value =>
          val message = new ProducerRecord[String, String](kafkaTopic, null, value)
          // feed the message into the Kafka topic
          producer.send(message)
        }

        producer.close()
      }
    }

    ssc.start()
    ssc.awaitTermination()
  }

  def main(args: Array[String]): Unit = {
    // docker compose up

    // docker exec -it sutek-sparkstreaming-kafka bash
    // cd /opt/kafka       // cd /opt/kafka_2.13-2.8.1
    // bin/kfaka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic sutek

    // bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic sutek
//    readFromKafka()

    // ncat -l -p 2023
    // bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic sutek
    writeToKafka()

    // docker compose down
  }

}
