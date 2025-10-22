name := "SparkNerPoc"
version := "1.0"
scalaVersion := "2.12.15"

val sparkVersion      = "3.3.0"
val sparkNlpVersion   = "5.1.4"
val akkaVersion       = "2.6.19"
val akkaHttpVersion   = "10.2.7"
val kafkaClientVersion= "3.0.0"
val playJsonVersion   = "2.9.4" // Use latest compatible

libraryDependencies ++= Seq(
  // Spark
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.johnsnowlabs.nlp" %% "spark-nlp" % sparkNlpVersion,

  // Kafka
  "org.apache.kafka" % "kafka-clients" % kafkaClientVersion,

  // Akka
  "com.typesafe.akka" %% "akka-actor-typed" % akkaVersion,
  "com.typesafe.akka" %% "akka-stream" % akkaVersion,
  "com.typesafe.akka" %% "akka-http" % akkaHttpVersion,
  "com.typesafe.akka" %% "akka-http-core" % akkaHttpVersion,

  // Play JSON
  "com.typesafe.play" %% "play-json" % playJsonVersion
)

// Increase JVM memory for Spark jobs
javaOptions += "-Xmx6G"

// Akka resolver (required for some Akka artifacts)
resolvers += "Akka libraries" at "https://repo.akka.io/maven"

// CRITICAL FIX: Merge strategy to handle resource conflicts
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}