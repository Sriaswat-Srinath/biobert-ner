import json
from kafka import KafkaProducer

def create_kafka_producer(server='localhost:9092'):
    """Creates a Kafka producer that sends JSON messages."""
    return KafkaProducer(
        bootstrap_servers=[server],
        # Serialize the JSON dictionary to a byte string
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

if __name__ == "__main__":
    # The Kafka topic where raw data will be sent
    topic_name = "raw_biomedical_text"

    # Initialize the producer
    producer = create_kafka_producer()

    # Example of a new, raw text document arriving for analysis
    new_document = {
        "doc_id": "PMID_NEW_456",
        "text": "A new study on the effects of aspirin on p53 gene expression in cancer patients."
    }

    # Send the new document to the Kafka topic
    print(f"Sending document '{new_document['doc_id']}' to topic '{topic_name}'...")
    producer.send(topic_name, value=new_document)

    # Ensure all messages are sent before the script exits
    producer.flush()
    print("✅ Message sent successfully.")
