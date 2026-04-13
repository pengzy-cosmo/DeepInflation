"""Encyclopedia RAG built directly on OpenAI embeddings and LanceDB."""

import json
import logging
from hashlib import md5
from pathlib import Path

import lancedb
import tiktoken
from lancedb.rerankers import RRFReranker
from openai import OpenAI

_PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = _PROJECT_ROOT / "data/models"
MODEL_LIST_PATH = _PROJECT_ROOT / "data/model_list.json"
LANCEDB_DIR = _PROJECT_ROOT / "tmp/lancedb"
TABLE_NAME = "encyclopedia_chunks"
CONFIG_FILE = "embedding_config.json"
CHUNK_TOKENS = 500
PARENT_MAX_TOKENS = 5000
EMBED_BATCH_SIZE = 128

logger = logging.getLogger(__name__)

_enc = None


def _tokens(text: str) -> int:
    global _enc
    if _enc is None:
        try:
            _enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _enc = False

    if _enc is False:
        return max(1, len(text.split()))

    return len(_enc.encode(text))


class EncyclopediaRAG:
    """Search chunks, return parent documents."""

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.db = lancedb.connect(str(LANCEDB_DIR))
        self.config_path = Path(LANCEDB_DIR) / CONFIG_FILE
        self.parent_store: dict[str, dict] = {}
        self.table = None

        logger.debug("[Encyclopedia] Initializing...")
        self._open_or_build_index()
        logger.debug("[Encyclopedia] Ready %d parents", len(self.parent_store))

    def _open_or_build_index(self) -> None:
        """Reuse an existing index when config matches; otherwise rebuild it."""
        saved = {}
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                saved = json.load(f)

        config_matches = (
            saved.get("embedding_model") == self.embedding_model
            and saved.get("chunk_tokens") == CHUNK_TOKENS
            and saved.get("parent_max_tokens") == PARENT_MAX_TOKENS
        )

        if TABLE_NAME in self.db.table_names() and config_matches:
            self.table = self.db.open_table(TABLE_NAME)
            self.parent_store = saved.get("parent_store", {})
            if self.table.count_rows() > 0 and self.parent_store:
                logger.debug(
                    "[Encyclopedia] Loaded %d chunks, %d parents", self.table.count_rows(), len(self.parent_store)
                )
                return

        self._build_index()

    def _build_index(self) -> None:
        """Build parent docs, chunk them, embed them, then store the index."""
        logger.debug("[Encyclopedia] Building index...")

        metadata_by_model = {}
        if MODEL_LIST_PATH.exists():
            with open(MODEL_LIST_PATH, encoding="utf-8") as f:
                for entry in json.load(f):
                    metadata_by_model[entry["Model"]] = {
                        "potential_latex": entry.get("Potential $V(\\phi)$", ""),
                        "parameters": entry.get("Parameters", ""),
                    }

        self.parent_store = {}
        rows = []

        for md_file in sorted(MODELS_DIR.glob("*.md")):
            content = md_file.read_text(encoding="utf-8").strip()
            if len(content) < 100:
                continue

            model_name = md_file.stem
            metadata = metadata_by_model.get(model_name, {})
            # Parent documents are the units returned to the model. Large markdown
            # files are split by top-level sections so retrieval returns coherent
            # context instead of one oversized blob.
            parents = [(model_name, content)]
            if _tokens(content) > PARENT_MAX_TOKENS:
                parents = self._split_by_sections(content, model_name)

            for title, text in parents:
                parent_id = md5(title.encode()).hexdigest()[:16]
                self.parent_store[parent_id] = {
                    "title": title,
                    "content": text,
                    "metadata": metadata,
                    "model": model_name,
                }

                body = text
                lines = text.split("\n")
                if lines and lines[0].startswith("# "):
                    body = "\n".join(lines[1:]).strip()

                # The vector index works on smaller paragraph chunks, but each
                # chunk still points back to its parent document for final output.
                for chunk in self._chunk_by_paragraphs(body):
                    rows.append(
                        {
                            "id": md5(f"{parent_id}:{chunk}".encode()).hexdigest(),
                            "parent_id": parent_id,
                            "model": model_name,
                            "title": title,
                            "text": chunk.replace("\x00", " "),
                        }
                    )

            logger.debug("[Encyclopedia] Indexed %s: %d parent(s)", model_name, len(parents))

        if not rows:
            raise RuntimeError("No encyclopedia documents found to index")

        # Embeddings are generated in batches before writing to LanceDB so the
        # table can serve hybrid vector + full-text retrieval.
        texts = [row["text"] for row in rows]
        embeddings = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            response = self.client.embeddings.create(model=self.embedding_model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
            logger.debug("[Encyclopedia] Embedded %d/%d chunks", min(start + len(batch), len(texts)), len(texts))

        for row, embedding in zip(rows, embeddings, strict=True):
            row["vector"] = embedding

        if TABLE_NAME in self.db.table_names():
            self.db.drop_table(TABLE_NAME)
        self.table = self.db.create_table(TABLE_NAME, data=rows, mode="overwrite")
        self.table.create_index(vector_column_name="vector", replace=True)
        self.table.create_fts_index("text", replace=True)

        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "embedding_model": self.embedding_model,
                    "chunk_tokens": CHUNK_TOKENS,
                    "parent_max_tokens": PARENT_MAX_TOKENS,
                    "parent_store": self.parent_store,
                },
                f,
                ensure_ascii=False,
            )

        logger.debug("[Encyclopedia] Built %d chunks", self.table.count_rows())

    def _split_by_sections(self, content: str, model_name: str) -> list[tuple[str, str]]:
        """Split a large markdown file into parent documents by top-level headings."""
        sections = []
        current_lines = []
        current_title = model_name
        is_first = True

        for line in content.split("\n"):
            if line.startswith("# "):
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if _tokens(text) > 50:
                        sections.append((current_title, text))
                current_title = model_name if is_first else f"{model_name} - {line[2:].strip()}"
                current_lines = [line]
                is_first = False
            else:
                current_lines.append(line)

        if current_lines:
            text = "\n".join(current_lines).strip()
            if _tokens(text) > 50:
                sections.append((current_title, text))

        return sections or [(model_name, content)]

    def _chunk_by_paragraphs(self, text: str) -> list[str]:
        """Group nearby paragraphs into retrieval chunks under the token budget."""
        if _tokens(text) <= CHUNK_TOKENS:
            return [text]

        chunks = []
        current = []
        current_tokens = 0

        for paragraph in text.split("\n\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            paragraph_tokens = _tokens(paragraph)
            if paragraph_tokens > CHUNK_TOKENS:
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                chunks.append(paragraph)
                continue

            if current and current_tokens + paragraph_tokens > CHUNK_TOKENS:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0

            current.append(paragraph)
            current_tokens += paragraph_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def search(self, query: str, num_chunks: int = 10, num_parents: int = 3) -> list[dict]:
        if self.table is None:
            return []

        # Retrieve chunk candidates with hybrid search, then merge scores back to
        # parent documents so the caller receives full model descriptions.
        query_embedding = self.client.embeddings.create(model=self.embedding_model, input=query).data[0].embedding
        frame = (
            self.table.search(query_embedding, query_type="hybrid")
            .text(query)
            .rerank(RRFReranker())
            .limit(num_chunks)
            .to_pandas()
        )

        if frame.empty:
            return []

        scores: dict[str, float] = {}
        for rank, row in enumerate(frame.itertuples(index=False), start=1):
            parent_id = getattr(row, "parent_id", None)
            if parent_id:
                scores[parent_id] = scores.get(parent_id, 0.0) + 1.0 / (rank + 1)

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [
            {**self.parent_store[parent_id], "score": scores[parent_id]}
            for parent_id in ranked_ids[:num_parents]
            if parent_id in self.parent_store
        ]


_rag: EncyclopediaRAG | None = None


def init_rag(
    api_key: str,
    base_url: str | None = None,
    embedding_model: str = "text-embedding-3-small",
) -> EncyclopediaRAG:
    """Initialize the encyclopedia singleton."""
    global _rag
    _rag = EncyclopediaRAG(api_key, base_url, embedding_model)
    return _rag


def search_encyclopedia(query: str, top_k: int = 3) -> str:
    """Search the Encyclopaedia Inflationaris and return parent documents."""
    if _rag is None:
        return json.dumps({"success": False, "error": "Encyclopedia not initialized"})

    top_k = max(1, min(5, top_k))

    try:
        results = _rag.search(query, num_chunks=4 * top_k, num_parents=top_k)
    except Exception as exc:
        return json.dumps({"success": False, "error": str(exc)})

    if not results:
        return json.dumps(
            {
                "success": False,
                "message": "No matching models found. Try different keywords.",
                "results": [],
            }
        )

    return json.dumps(
        {
            "success": True,
            "count": len(results),
            "citation": "Encyclopaedia Inflationaris (https://arxiv.org/abs/1303.3787)",
            "results": [
                {
                    "title": result["title"],
                    "content": result["content"],
                    "potential_latex": result["metadata"].get("potential_latex", ""),
                    "parameters": result["metadata"].get("parameters", ""),
                }
                for result in results
            ],
        },
        ensure_ascii=False,
    )
