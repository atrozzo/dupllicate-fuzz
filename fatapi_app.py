from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Tuple, Any
import itertools
import re
from rapidfuzz import fuzz

app = FastAPI(
    title="Duplicate Detection Service",
    description=(
        "Detects potential duplicate records using a simple weighted similarity heuristic. "
        "Records can be provided in the request body or loaded at startup for query-based matching."
    ),
    version="0.2.0",
)


class Record(BaseModel):

    id: int = Field(..., description="Unique identifier for the record")
    name: str = Field(..., description="Full name of the person")
    hair_colour: str = Field(..., description="Hair colour")
    race: str = Field(..., description="Race or ethnic background")
    eye_colour: str = Field(..., description="Eye colour")
    height_cm: float = Field(..., description="Height in centimetres")
    address: str | None = Field(None, description="Registered address of the person")


class DedupRequest(BaseModel):
    records: List[Record]


class PhysQuery(BaseModel):

    hair_colour: str | None = Field(None, description="Hair colour to match")
    race: str | None = Field(None, description="Race to match")
    eye_colour: str | None = Field(None, description="Eye colour to match")
    height_cm: float | None = Field(None, description="Approximate height in centimetres")
    height_tolerance: float = Field(5.0, description="Tolerance in centimetres for height matching")



LOADED_RECORDS: List[Record] = []


@app.on_event("startup")
def load_data() -> None:

    global LOADED_RECORDS

    LOADED_RECORDS = [
        # Original records with addresses
        Record(id=1, name="John Smith", hair_colour="brown", race="White", eye_colour="blue", height_cm=180, address="123 Main Street, London, UK"),
        Record(id=2, name="Jon Smith", hair_colour="brown", race="White", eye_colour="blue", height_cm=181, address="124 Main Street, London, UK"),
        Record(id=3, name="John Smithe", hair_colour="brown", race="White", eye_colour="blue", height_cm=179, address="125 Main Street, London, UK"),
        Record(id=4, name="Jane Doe", hair_colour="blonde", race="White", eye_colour="green", height_cm=165, address="1 Park Avenue, London, UK"),
        Record(id=5, name="Jane Do", hair_colour="blonde", race="White", eye_colour="green", height_cm=166, address="2 Park Avenue, London, UK"),
        Record(id=6, name="Juan Garcia", hair_colour="black", race="Hispanic", eye_colour="brown", height_cm=170, address="10 High Street, Oxford, UK"),
        Record(id=7, name="Juana Garcia", hair_colour="black", race="Hispanic", eye_colour="brown", height_cm=170, address="11 High Street, Oxford, UK"),
        Record(id=8, name="Mary Johnson", hair_colour="red", race="White", eye_colour="hazel", height_cm=160, address="50 Queen's Road, Manchester, UK"),
        Record(id=9, name="Mairy Johnson", hair_colour="red", race="White", eye_colour="hazel", height_cm=161, address="51 Queen's Road, Manchester, UK"),
        Record(id=10, name="Mark Spencer", hair_colour="brown", race="Black", eye_colour="brown", height_cm=190, address="100 King's Road, Liverpool, UK"),
        # New records to double the dataset
        Record(id=11, name="Mark Spenser", hair_colour="brown", race="Black", eye_colour="brown", height_cm=191, address="101 King's Road, Liverpool, UK"),
        Record(id=12, name="Mary Jonson", hair_colour="red", race="White", eye_colour="hazel", height_cm=159, address="102 King's Road, Liverpool, UK"),
        Record(id=13, name="John Jones", hair_colour="brown", race="White", eye_colour="blue", height_cm=182, address="3 Park Avenue, London, UK"),
        Record(id=14, name="Jane Roe", hair_colour="blonde", race="White", eye_colour="green", height_cm=164, address="4 Park Avenue, London, UK"),
        Record(id=15, name="Juanita Garcia", hair_colour="black", race="Hispanic", eye_colour="brown", height_cm=170, address="12 High Street, Oxford, UK"),
        Record(id=16, name="Maria Gonzalez", hair_colour="black", race="Hispanic", eye_colour="brown", height_cm=165, address="13 High Street, Oxford, UK"),
        Record(id=17, name="Mary Jane", hair_colour="blonde", race="White", eye_colour="brown", height_cm=168, address="52 Queen's Road, Manchester, UK"),
        Record(id=18, name="Michael Smith", hair_colour="brown", race="White", eye_colour="green", height_cm=175, address="53 Queen's Road, Manchester, UK"),
        Record(id=19, name="Michelle Smith", hair_colour="brown", race="White", eye_colour="green", height_cm=175, address="200 King's Road, Liverpool, UK"),
        Record(id=20, name="Jake Doe", hair_colour="brown", race="White", eye_colour="blue", height_cm=170, address="201 King's Road, Liverpool, UK"),
    ]


def preprocess_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z ]", "", name).lower()


def name_similarity(name1: str, name2: str) -> float:

    return fuzz.token_sort_ratio(preprocess_name(name1), preprocess_name(name2)) / 100.0


def similarity_score(r1: Record, r2: Record, height_threshold: float = 5.0) -> float:

    # Weights for each component
    weights = {
        "name": 0.4,
        "hair": 0.2,
        "race": 0.2,
        "eye": 0.1,
        "height": 0.1,
    }
    # Compute individual similarities
    name_sim = name_similarity(r1.name, r2.name)
    hair_sim = 1.0 if r1.hair_colour.lower() == r2.hair_colour.lower() else 0.0
    race_sim = 1.0 if r1.race.lower() == r2.race.lower() else 0.0
    eye_sim = 1.0 if r1.eye_colour.lower() == r2.eye_colour.lower() else 0.0
    height_sim = 1.0 if abs(r1.height_cm - r2.height_cm) <= height_threshold else 0.0

    total = sum(weights.values())
    weighted = (name_sim * weights["name"] + hair_sim * weights["hair"] +
                race_sim * weights["race"] + eye_sim * weights["eye"] +
                height_sim * weights["height"]) / total
    return weighted


def cluster_records(records: List[Record], threshold: float = 0.75) -> List[Dict[str, Any]]:
    """Group records into potential duplicate clusters.

    Performs pairwise comparisons using the similarity_score function and
    assigns records to the same cluster if their similarity meets or exceeds
    the threshold.  Clustering is transitive: if A matches B and B matches C,
    then all three will be in the same cluster.

    Returns a list of dictionaries with keys `group_id` and `records`.
    """
    # Disjoint-set (union-find) to manage clusters
    parent: Dict[int, int] = {i: i for i in range(len(records))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pj] = pi

    # Compare all unique pairs
    for i, j in itertools.combinations(range(len(records)), 2):
        if similarity_score(records[i], records[j]) >= threshold:
            union(i, j)

    # Group indices by root parent
    clusters: Dict[int, List[int]] = {}
    for idx in range(len(records)):
        root = find(idx)
        clusters.setdefault(root, []).append(idx)

    # Build response structure
    result: List[Dict[str, Any]] = []
    for group_id, (root, member_indices) in enumerate(clusters.items(), start=1):
        result.append({
            "group_id": group_id,
            "records": [records[i] for i in member_indices]
        })
    return result


@app.post("/dedup")
def dedup(request: DedupRequest, threshold: float = 0.75) -> Dict[str, Any]:
    """Endpoint to detect duplicates.

    Accepts a JSON payload containing an array of records and returns
    potential duplicate clusters.  The similarity threshold can be
    overridden using the query parameter `threshold` (default is 0.75).
    """
    clusters = cluster_records(request.records, threshold=threshold)
    return {"clusters": clusters}


@app.post("/search")
def search_records(query: PhysQuery, threshold: float = 0.75) -> Dict[str, Any]:
    """Search loaded records based on physical characteristics and cluster matches.

    The application maintains a list of records loaded at startup.  This
    endpoint filters those records according to the provided physical
    characteristics:

    * hair_colour, race and eye_colour are matched case-insensitively.
    * height_cm matches if the absolute difference from the record's height
      is within `height_tolerance`.

    After filtering, the matching records are clustered using the same
    similarity heuristic as the `/dedup` endpoint.  The response contains
    the list of clusters and the matching records.
    """
    if not any([query.hair_colour, query.race, query.eye_colour, query.height_cm is not None]):
        return {"error": "At least one physical characteristic must be provided"}
    matches: List[Record] = []
    for rec in LOADED_RECORDS:
        if query.hair_colour and rec.hair_colour.lower() != query.hair_colour.lower():
            continue
        if query.race and rec.race.lower() != query.race.lower():
            continue
        if query.eye_colour and rec.eye_colour.lower() != query.eye_colour.lower():
            continue
        if query.height_cm is not None:
            if abs(rec.height_cm - query.height_cm) > query.height_tolerance:
                continue
        matches.append(rec)
    # If no matches, return empty result
    if not matches:
        return {"clusters": [], "matches": []}
    clusters = cluster_records(matches, threshold=threshold)
    return {"clusters": clusters, "matches": matches}


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint providing a brief description."""
    return {"message": "Duplicate detection service. Use POST /dedup with your records."}
