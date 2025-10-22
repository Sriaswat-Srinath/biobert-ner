// src/main/scala/LambdaNerStreamProcessor.scala

import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import play.api.libs.json._
import com.johnsnowlabs.nlp.util.io.ReadAs // Needed for TextMatcher

object LambdaNerStreamProcessor extends App {

  // --- CONFIGURATION ---
  val INPUT_TOPIC = "pubmed-stream-scala"
  val KAFKA_BROKERS = "localhost:9092"
  
  // *** OUTPUT SINKS ***
  val KAFKA_OUTPUT_TOPIC = "pubmed-enriched-ner" // Speed Layer (Real-Time)
  val HDFS_ARCHIVE_PATH = "hdfs://localhost:9000/archive/pubmed_ner_data" // Batch Layer (Archive)

  // *** LOCAL RESOURCE CONFIGURATION (for low RAM/C-Drive fix) ***
  val SPARK_TEMP_DIR = "E:/SparkTemp" 
  val DICTIONARY_PATH = s"$SPARK_TEMP_DIR/biomed_terms.txt" // Your local dictionary path

  // 1. Create a Spark Session with Minimal Memory Configs
  val spark = SparkSession.builder()
    .appName("PubMedLambdaNERProcessor")
    .master("local[*]")
    
    // --- LOW MEMORY CONFIGS (SAFEST FOR 4GB RAM) ---
    .config("spark.driver.memory", "1024m") 
    .config("spark.executor.memory", "1024m")
    
    // --- DISK SPACE FIXES (CRITICAL) ---
    .config("spark.driver.extraJavaOptions", s"-Djava.io.tmpdir=$SPARK_TEMP_DIR")
    .config("spark.executor.extraJavaOptions", s"-Djava.io.tmpdir=$SPARK_TEMP_DIR")
    .config("spark.hadoop.hadoop.tmp.dir", SPARK_TEMP_DIR)
    
    .getOrCreate()

  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  // 2. Define the Schema for the Incoming Raw Kafka Value
  val rawDataSchema = new StructType()
    .add("pmid", StringType)
    .add("title", StringType)
    .add("abstract", StringType)

  // 3. Define the Dictionary-Based Pipeline Stages (Low-Memory NER)

  val documentAssembler = new DocumentAssembler()
    .setInputCol("raw_text") 
    .setOutputCol("document")
  
  val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
  
  // *** RULE-BASED NER: TextMatcher for dictionary lookup ***
  val dictionaryMatcher = new TextMatcher() 
    .setInputCols("document", "token")
    .setOutputCol("dictionary_entities")
    .setEntities("file:///" + DICTIONARY_PATH, ReadAs.TEXT)
    .setCaseSensitive(false)
    .setEntityValue("ENTITY")

  // Fit the Pipeline
  val ruleBasedPipelineModel = new Pipeline().setStages(Array(
    documentAssembler, tokenizer, dictionaryMatcher
  )).fit(spark.emptyDataset[String].toDF("raw_text"))


  // 4. Define the Streaming ETL Logic (Read from Kafka)
  val rawKafkaStream = spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKERS)
    .option("subscribe", INPUT_TOPIC)
    .option("startingOffsets", "latest") 
    .load()

  val parsedStream = rawKafkaStream
    .select(
      col("key"),
      from_json(col("value").cast("string"), rawDataSchema).as("data")
    )
    .withColumn("pmid", col("key").cast("string"))
    .withColumn("raw_text", concat_ws(" ", col("data.title"), col("data.abstract")))
    .select(
      col("pmid"),
      col("raw_text"),
      col("data.title").as("title"),
      col("data.abstract").as("abstract")
    )

  // C. Apply the Rule-Based Pipeline
  val enrichedStream = ruleBasedPipelineModel.transform(parsedStream)

  // 5. Transform NER Annotations to Final Output Formats

  // Create the base DataFrame containing all flattened entities
  val flattenedEntities = enrichedStream
    .select(
      $"pmid",
      $"title", 
      $"abstract",
      explode($"dictionary_entities").as("entity")
    )
    .withColumn("entity_text", $"entity.result")
    .withColumn("entity_type", $"entity.metadata.entity")
    .withColumn("timestamp", current_timestamp())

  // A. KAFKA OUTPUT (Aggregated JSON for Speed Layer)
  val kafkaOutputDf = flattenedEntities
    .groupBy("pmid", "title", "abstract")
    .agg(
      collect_list(
        struct(
          $"entity_text".as("text"),
          $"entity_type".as("type")
        )
      ).as("extracted_entities"),
      max($"timestamp").as("timestamp")
    )
    .withColumn("value", to_json(struct(col("*"))))
    .select(
      $"pmid".as("key"), 
      $"value"
    )

  // B. HDFS OUTPUT (Parquet/Columnar for Batch Layer)
  val hdfsOutputDf = flattenedEntities
    .withColumn("date_part", date_format($"timestamp", "yyyy-MM-dd")) // Partition column 1

  // 6. Write the Result to DUAL SINKS

  // --- SINK 1: KAFKA (SPEED LAYER) ---
  val kafkaQuery = kafkaOutputDf.writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", KAFKA_BROKERS)
    .option("topic", KAFKA_OUTPUT_TOPIC)
    .option("checkpointLocation", s"$SPARK_TEMP_DIR/spark/checkpoints/kafka_ner_lambda") 
    .outputMode("update") 
    .start()

  // --- SINK 2: HDFS (BATCH LAYER) ---
  val hdfsQuery = hdfsOutputDf.writeStream
    .format("parquet") 
    .option("path", HDFS_ARCHIVE_PATH) 
    .option("checkpointLocation", s"$SPARK_TEMP_DIR/spark/checkpoints/hdfs_ner_lambda") 
    .partitionBy("date_part", "entity_type") 
    .outputMode("append") 
    .start()
    
  println(s"--- Spark Streaming Processor Started (Lambda). Kafka Sink: $KAFKA_OUTPUT_TOPIC, HDFS Sink: $HDFS_ARCHIVE_PATH ---")
  
  // Keep the application running indefinitely
  kafkaQuery.awaitTermination()
}