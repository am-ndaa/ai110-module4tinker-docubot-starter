"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            words = set(text.lower().split())
            for word in words:
                if word not in index:
                    index[word] = []
                index[word].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def meaningful_query_words(self, query):
        stopwords = {
            "where",
            "do",
            "i",
            "the",
            "is",
            "are",
            "a",
            "an",
            "and",
            "or",
            "to",
            "for",
            "on",
            "in",
            "of",
            "what",
            "which",
            "when",
            "how",
            "who",
            "please",
            "find",
            "can",
            "could",
            "should",
            "it",
            "that",
            "this",
            "with",
            "by",
            "as",
            "at",
            "from",
            "not",
            "be",
            "was",
            "were",
            "has",
            "have",
            "had",
        }
        words = []
        for word in query.lower().split():
            token = word.strip(".,!?\"'()[]{}:;")
            if token and token not in stopwords:
                words.append(token)
        return words

    def is_vague_query(self, query):
        meaningful_words = self.meaningful_query_words(query)
        return len(meaningful_words) < 2

    def score_document(self, query, text):
        """
        Return a simple relevance score based on meaningful query terms.
        """
        query_words = self.meaningful_query_words(query)
        if not query_words:
            return 0

        text_lower = text.lower()
        score = 0
        for word in query_words:
            score += text_lower.count(word)
        print(f"SCORE: {score}")
        return score

    def retrieve(self, query, top_k=5):
        """
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        results = []  # Initialize list for results
        for filename, text in self.documents:  # Loop through all documents
            sections = []
            current_section = []
            for line in text.splitlines():
                if line.strip():
                    current_section.append(line)
                else:
                    if current_section:
                        sections.append("\n".join(current_section).strip())
                        current_section = []
            if current_section:
                sections.append("\n".join(current_section).strip())

            for section in sections:
                score = self.score_document(query, section)
                if score > 0:  # Only include sections with at least one match
                    results.append((score, filename, section))

        results.sort(reverse=True)  # Sort by score descending
        return [(filename, section) for score, filename, section in results[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=5):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        if self.is_vague_query(query):
            return "I cannot answer that question based on these docs."

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I cannot answer that question based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.is_vague_query(query):
            return "I cannot answer that question based on these docs."

        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I cannot answer that question based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
