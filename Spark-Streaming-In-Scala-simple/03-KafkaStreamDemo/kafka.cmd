@rem docker compose up

docker exec -it sutek-sparkstreaming-kafka bash

@rem cd /opt/kafka
@rem bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic invoices

@rem bin/kafka-console-producer.sh --bootstrap-server localhost:9092 --topic invoices

@rem bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic invoices

@rem docker compose down
