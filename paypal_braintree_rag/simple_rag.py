import os
import sys
import argparse
import time
from dotenv import load_dotenv

# Load env vars from .env in this directory
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# Add this directory to path so helper_functions and evaluation/ are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helper_functions import *
from evaluation.evalute_rag import *
from corpus.embed import load_index
from langchain_openai import ChatOpenAI


class SimpleRAG:
    """
    A class to handle the Simple RAG process for document chunking and query retrieval.
    Loads from a pre-built FAISS corpus index (corpus/data/faiss_index/) by default,
    or falls back to encoding a PDF if --path is provided.
    """

    def __init__(self, path=None, chunk_size=256, chunk_overlap=200, n_retrieved=3):
        """
        Initializes the SimpleRAG retriever.

        Args:
            path (str|None): Path to a PDF file. If None, loads from the pre-built corpus index.
            chunk_size (int): Chunk size used when encoding a PDF (ignored for corpus mode).
            chunk_overlap (int): Overlap used when encoding a PDF (ignored for corpus mode).
            n_retrieved (int): Number of chunks to retrieve per query.
        """
        print("\n--- Initializing Simple RAG Retriever ---")

        start_time = time.time()
        if path:
            print(f"Loading from PDF: {path}")
            self.vector_store = encode_pdf(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            self.time_records = {'Loading': time.time() - start_time}
            print(f"PDF encoding time: {self.time_records['Loading']:.2f}s")
        else:
            print("Loading pre-built Braintree corpus from FAISS index...")
            self.vector_store = load_index()
            self.time_records = {'Loading': time.time() - start_time}
            print(f"Index load time: {self.time_records['Loading']:.2f}s")
            print(f"Vectors in index: {self.vector_store.index.ntotal}")

        # MMR retriever — fetches more candidates then picks the most diverse k
        self.chunks_query_retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": n_retrieved, "fetch_k": n_retrieved * 4},
        )

        # LLM for answer generation
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.qa_chain = create_question_answer_from_context_chain(llm)

    def run(self, query, show_context_chunks=False):
        """
        Retrieves context for the given query and generates an answer.

        Args:
            query (str): The query to answer.
            show_context_chunks (bool): If True, also print the raw retrieved chunks.
        """
        # Retrieve relevant chunks (MMR ensures diversity — no duplicates)
        start_time = time.time()
        context = retrieve_context_per_question(query, self.chunks_query_retriever)
        self.time_records['Retrieval'] = time.time() - start_time
        print(f"Retrieval Time: {self.time_records['Retrieval']:.2f} seconds")

        if show_context_chunks:
            print("\n--- Retrieved Context ---")
            show_context(context)

        # Generate answer from context
        context_text = "\n\n".join(context)
        result = self.qa_chain.invoke({"question": query, "context": context_text})

        print(f"\n{'='*60}")
        print(f"Question: {query}")
        print(f"{'='*60}")
        print(f"Answer:\n{result.answer_based_on_content}")
        print(f"{'='*60}\n")


# Function to validate command line inputs
def validate_args(args):
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if args.chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")
    if args.n_retrieved <= 0:
        raise ValueError("n_retrieved must be a positive integer.")
    return args


# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Query the Braintree RAG corpus (or encode a PDF).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Query the pre-built Braintree corpus (run corpus/embed.py first)
  python simple_rag.py --query "How do I create a transaction?"

  # Encode a PDF instead
  python simple_rag.py --path data/my_doc.pdf --query "What is Braintree?"
        """,
    )
    parser.add_argument("--path",         type=str, default=None,
                        help="Path to a PDF file. Omit to use the pre-built corpus index.")
    parser.add_argument("--chunk_size",   type=int, default=256)
    parser.add_argument("--chunk_overlap",type=int, default=200)
    parser.add_argument("--n_retrieved",  type=int, default=3,
                        help="Number of chunks to retrieve per query (default 3).")
    parser.add_argument("--query",        type=str, default="What is Braintree?",
                        help="Query to test the retriever.")
    parser.add_argument("--evaluate",     action="store_true",
                        help="Run evaluation suite after retrieval.")
    parser.add_argument("--show-context", action="store_true",
                        help="Also print the raw retrieved chunks alongside the answer.")

    return validate_args(parser.parse_args())


# Main function to handle argument parsing and call the SimpleRAGRetriever class
def main(args):
    simple_rag = SimpleRAG(
        path=args.path,           # None → load corpus index
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        n_retrieved=args.n_retrieved,
    )

    # Retrieve context and generate answer
    simple_rag.run(args.query, show_context_chunks=args.show_context)

    # Evaluate the retriever's performance on the query (if requested)
    if args.evaluate:
        evaluate_rag(simple_rag.chunks_query_retriever)


if __name__ == '__main__':
    # Call the main function with parsed arguments
    main(parse_args())
