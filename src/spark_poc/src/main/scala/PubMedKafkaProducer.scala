// src/main/scala/PubMedKafkaProducer.scala

import org.apache.kafka.clients.producer.{KafkaProducer, ProducerRecord, Callback, RecordMetadata}
import org.apache.kafka.common.serialization.StringSerializer
import akka.actor.typed.ActorSystem
import akka.actor.typed.scaladsl.Behaviors
import play.api.libs.json._

import java.util.Properties
import scala.concurrent.{Await, ExecutionContext, Future, TimeoutException}
import scala.util.control.NonFatal
import java.lang.Void
import scala.concurrent.duration._

object PubMedKafkaProducer extends App {
  
  // --- CONFIGURATION ---
  val INPUT_TOPIC = "pubmed-stream-scala"
  val KAFKA_BROKERS = "localhost:9092"
  
  // --- CUSTOM BATCH DEFINITION (Data to be sent once) ---
  val CUSTOM_MESSAGES: Seq[String] = Seq(
    // Message 1: Will trigger a successful dictionary match (if terms are in dictionary)
    Json.obj(
      "pmid" -> "00000001",
      "title" -> "Therapeutic uses of Aspirin and Paracetamol.",
      "abstract" -> "Aspirin is often prescribed for chronic inflammation, but its use alongside paracetamol is common."
    ).toString(),
    
    // Message 2: Will test the CFTR/Disease matching logic
    Json.obj(
      "pmid" -> "00000002",
      "title" -> "CFTR gene variants in Cystic Fibrosis patients.",
      "abstract" -> "Mutations in the CFTR gene frequently lead to severe pulmonary infection."
    ).toString(),

    // Message 3: Contains terms that should NOT match (to test noise rejection)
    Json.obj(
      "pmid" -> "00000003",
      "title" -> "General Science Review of the Quarter.",
      "abstract" -> "The latest trends in astronomy and oceanography reveal huge cosmic distances and deep-sea trenches."
    ).toString()
  )
  
  // Kafka Producer Configuration
  val kafkaProps = new Properties()
  kafkaProps.put("bootstrap.servers", KAFKA_BROKERS)
  kafkaProps.put("key.serializer", classOf[StringSerializer].getName)
  kafkaProps.put("value.serializer", classOf[StringSerializer].getName)
  kafkaProps.put("acks", "all")
  
  // Actor System Setup (Akka) - Required for structured logging
  implicit val system: ActorSystem[Void] = ActorSystem(Guardian.behavior, "PubMedProducerBatch")
  implicit val ec: ExecutionContext = system.executionContext
  
  val producer = new KafkaProducer[String, String](kafkaProps)

  // --- KAFKA DELIVERY CALLBACK ---
  val deliveryCallback = new Callback {
    override def onCompletion(metadata: RecordMetadata, exception: Exception): Unit = {
      if (exception != null) {
        system.log.error(s"Kafka Delivery Failed: ${exception.getMessage}")
      }
    }
  }

  // --- MAIN BATCH EXECUTION FUNCTION ---
  def runBatchProcessor(): Unit = {
    system.log.info("Starting CUSTOM BATCH job to send pre-defined messages...")
    
    val produceFuture = Future {
      CUSTOM_MESSAGES.foreach { rawJson =>
        // Assuming your dictionary needs to match both parts for testing:
        val json = Json.parse(rawJson)
        val pmid = (json \ "pmid").as[String] 
        
        // Produce record, using PMID as the key
        val record = new ProducerRecord[String, String](INPUT_TOPIC, pmid, rawJson)
        producer.send(record, deliveryCallback)
      }
      producer.flush() // Ensure all messages are sent before exiting
      system.log.info(s"✅ BATCH SUCCESS: Produced ${CUSTOM_MESSAGES.size} records to Kafka.")
    }
    
    // CRITICAL: Block the main thread SYNCHRONOUSLY until the process finishes
    try {
      // Wait up to 10 seconds for the producer queue to clear
      Await.result(produceFuture, 10.seconds) 
    } catch {
      case e: TimeoutException => system.log.error("BATCH FAILED: Timeout reached while waiting for producer queue.")
      case NonFatal(e) => system.log.error(s"BATCH FAILED: Uncaught error during execution: ${e.getMessage}")
    }
    
    // Shut down resources cleanly
    producer.close()
    system.terminate()
    system.log.info("BATCH JOB FINISHED. Shutting down JVM.")
  }

  // Akka Guardian 
  object Guardian {
    def behavior: Behaviors.Receive[Void] = Behaviors.receiveMessage { _ => Behaviors.unhandled }
  }
  
  // Start the Batch Processor 
  runBatchProcessor()
}