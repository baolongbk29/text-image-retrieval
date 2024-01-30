from typing import List

from dotenv import dotenv_values
from pinecone import Pinecone

config = dotenv_values("../database/.env")
api_key = config["API_KEY"]


def get_index(index_name: str) -> Pinecone.Index:
    """Get index in Pinecone vector database

    Args:
        index_name (str): Index name

    Returns:
        pinecone.Index
    """
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def search(index: str, input_emb: List[float], top_k: int) -> List[int]:
    """Search the IDs of top similar images

    Args:
        index (str): index name
        input_emb (List[float]): input embedding
        top_k (int): number of top similar images

    Returns:
        List[int]: The IDs of top similar images
    """
    matching = index.query(vector=input_emb, top_k=top_k, include_values=True)[
        "matches"
    ]
    match_ids = [match_id["id"] for match_id in matching]
    return match_ids
