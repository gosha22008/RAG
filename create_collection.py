from qdrant_client import QdrantClient
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

import torch
import gc


COLLECT_NAME = "wb"
EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CHUNK_SIZE = []

# local_data 
local_data = Path.cwd() / "data"

# Проверка памяти GPU
def check_gpu_memory():
    """Выводит информацию о памяти GPU"""
    if not torch.cuda.is_available():
        print("GPU не доступен")
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    total = torch.cuda.get_device_properties(0).total_memory / 1024**2

    print(f"📊 GPU Memory Status:")
    print(f"   Total: {total:.0f} MB")
    print(f"   Allocated: {allocated:.1f} MB ({allocated/total*100:.1f}%)")
    print(f"   Reserved: {reserved:.1f} MB ({reserved/total*100:.1f}%)")
    print(f"   Free: {total - reserved:.1f} MB")


# Очистка памяти перед загрузкой новой модели
def cleanup_memory():
    """Полная очистка памяти GPU"""
    # Удаляем все большие объекты
    for obj in gc.get_objects():
        try:
          if isinstance(obj, torch.Tensor):
              del obj
        except:
          pass

    # Очищаем кэш
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Сборщик мусора
    gc.collect()

    print("✅ Память очищена")
    check_gpu_memory()


def drop_embed_model_from_gpu():
    Settings.embed_model = None
    cleanup_memory()


def load_data(dir):
    # считываем простым reader'ом файл оферты в формате txt
    reader = SimpleDirectoryReader(input_dir=dir, required_exts=[".pdf"])
    docs = reader.load_data()

    # делим документы на чанки
    nodes = SentenceSplitter().get_nodes_from_documents(documents=docs,
                                                        show_progress=True)
    return docs, nodes

 
def create_collect():
    # загрузка embedding модели
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    
    client = QdrantClient(url="http://127.0.0.1:6333")

    if client.collection_exists(collection_name=COLLECT_NAME):
        client.delete_collection(collection_name=COLLECT_NAME)

    # инициализируем векторное хранилище
    vector_store = QdrantVectorStore(collection_name=COLLECT_NAME,
                                     client=client)
    # создаём индекс
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # загружаем документы
    _, nodes = load_data(local_data)
    

    # создаём индекс документов
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        show_progress=True 
    )
    # пытаемся удалить embedding модель с GPU
    drop_embed_model_from_gpu()


    print("Коллекция успешно создана и проиндексирована")
    # return client, index

create_collect()

# client, index = get_client_index()