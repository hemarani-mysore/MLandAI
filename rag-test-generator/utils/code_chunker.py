"""
utils/code_chunker.py — AST-based smart code chunker
=====================================================
Responsible for:
  - Splitting CodeFile objects into smaller, meaningful chunks
  - Using Python AST to split at function/class boundaries (not mid-line)
  - Falling back to character-based splitting for non-Python files
  - Returning structured CodeChunk objects ready for embedding

Why smart chunking matters:
  Naive splitting (every 1000 chars) can cut a function in half, making
  the chunk meaningless. AST-based splitting keeps each function/class
  intact so the embedding represents a complete unit of logic.
"""

import ast
from dataclasses import dataclass
from utils.file_parser import CodeFile


@dataclass
class CodeChunk:
    """
    A chunk of code extracted from a CodeFile, ready to be embedded.

    Attributes:
        chunk_id    : Unique ID — "<relative_path>:<chunk_index>"
        file_path   : Relative path of the source file
        language    : Language of the source file
        content     : The actual code text for this chunk
        chunk_type  : "function", "class", "module_level", or "text_split"
        start_line  : Line number where this chunk starts (1-indexed)
        end_line    : Line number where this chunk ends
        name        : Function/class name if chunk_type is function/class
    """
    chunk_id   : str
    file_path  : str
    language   : str
    content    : str
    chunk_type : str
    start_line : int
    end_line   : int
    name       : str = ""


def _chunk_python_by_ast(code_file: CodeFile) -> list[CodeChunk]:
    """
    Use Python's built-in AST module to split a .py file at
    function and class boundaries.

    Strategy:
      1. Parse the file into an AST
      2. Walk top-level nodes: FunctionDef, AsyncFunctionDef, ClassDef
      3. Each becomes its own chunk
      4. Any lines NOT inside a function/class become a "module_level" chunk

    Falls back to text splitting if the file has syntax errors.
    """
    source_lines = code_file.content.splitlines(keepends=True)
    chunks = []

    try:
        tree = ast.parse(code_file.content)
    except SyntaxError:
        # If AST parse fails, fall back to text splitting
        return _chunk_by_text(code_file)

    # Collect top-level function and class definitions
    top_level_nodes = [
        node for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    # Track which lines are covered by a function/class
    covered_lines: set[int] = set()
    for node in top_level_nodes:
        for lineno in range(node.lineno, node.end_lineno + 1):
            covered_lines.add(lineno)

    idx = 0

    # Module-level code (imports, constants, etc.) — lines not in any function/class
    module_lines = [
        line for i, line in enumerate(source_lines, start=1)
        if i not in covered_lines
    ]
    if "".join(module_lines).strip():
        chunks.append(CodeChunk(
            chunk_id   = f"{code_file.path}:{idx}",
            file_path  = code_file.path,
            language   = code_file.language,
            content    = "".join(module_lines),
            chunk_type = "module_level",
            start_line = 1,
            end_line   = len(source_lines),
            name       = "",
        ))
        idx += 1

    # One chunk per function/class
    for node in top_level_nodes:
        chunk_lines = source_lines[node.lineno - 1 : node.end_lineno]
        chunk_type  = "class" if isinstance(node, ast.ClassDef) else "function"

        chunks.append(CodeChunk(
            chunk_id   = f"{code_file.path}:{idx}",
            file_path  = code_file.path,
            language   = code_file.language,
            content    = "".join(chunk_lines),
            chunk_type = chunk_type,
            start_line = node.lineno,
            end_line   = node.end_lineno,
            name       = node.name,
        ))
        idx += 1

    return chunks


def _chunk_by_text(code_file: CodeFile, chunk_size: int = 1000, overlap: int = 200) -> list[CodeChunk]:
    """
    Simple character-based chunking for non-Python files (TS, JS, TSX).

    We don't have an AST for TypeScript, so we split by character count
    with overlap to preserve context at chunk boundaries.

    overlap means the last `overlap` characters of chunk N are repeated
    at the start of chunk N+1. This ensures a function that spans a boundary
    is still fully represented in at least one chunk.
    """
    content = code_file.content
    chunks  = []
    idx     = 0
    pos     = 0

    while pos < len(content):
        end     = min(pos + chunk_size, len(content))
        snippet = content[pos:end]

        # Count line numbers for metadata
        start_line = content[:pos].count("\n") + 1
        end_line   = content[:end].count("\n") + 1

        chunks.append(CodeChunk(
            chunk_id   = f"{code_file.path}:{idx}",
            file_path  = code_file.path,
            language   = code_file.language,
            content    = snippet,
            chunk_type = "text_split",
            start_line = start_line,
            end_line   = end_line,
        ))

        idx += 1
        pos += chunk_size - overlap   # step forward, leaving overlap behind

        if pos >= len(content):
            break

    return chunks


def chunk_code_file(
    code_file  : CodeFile,
    chunk_size : int = 1000,
    overlap    : int = 200,
) -> list[CodeChunk]:
    """
    Main entry point. Choose the right chunking strategy based on language.

    Python  → AST-based (splits at function/class boundaries)
    Others  → Text-based (character splits with overlap)
    """
    if code_file.language == "python":
        return _chunk_python_by_ast(code_file)
    else:
        return _chunk_by_text(code_file, chunk_size=chunk_size, overlap=overlap)


def chunk_all_files(
    code_files : list[CodeFile],
    chunk_size : int = 1000,
    overlap    : int = 200,
) -> list[CodeChunk]:
    """
    Chunk every file in the list. Returns a flat list of all chunks.
    """
    all_chunks = []
    for code_file in code_files:
        chunks = chunk_code_file(code_file, chunk_size=chunk_size, overlap=overlap)
        all_chunks.extend(chunks)
    return all_chunks
