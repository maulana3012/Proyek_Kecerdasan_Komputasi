import os
from elasticsearch import Elasticsearch

ES_HOST = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch_service:9200")

es_client = Elasticsearch(hosts=[ES_HOST])

def init_es_indices():
    """Inisialisasi Index Elasticsearch jika belum ada."""
    if es_client.ping():
        if not es_client.indices.exists(index="attendance_logs"):
            es_client.indices.create(
                index="attendance_logs",
                body={
                    "mappings": {
                        "properties": {
                            "student_id": {"type": "integer"},
                            "nim": {"type": "keyword"},
                            "nama": {"type": "text"},
                            "prodi": {"type": "keyword"},
                            "kelas": {"type": "keyword"},
                            "confidence_score": {"type": "float"},
                            "detected_at": {"type": "date"}
                        }
                    }
                }
            )