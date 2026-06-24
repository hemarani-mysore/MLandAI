"""
tests/test_embedding.py — Unit tests for Chunk 3
=================================================
Tests the code chunker. Embedding/ChromaDB tests are skipped here
because they require a live API key — those are tested manually via
python agents/embedding_agent.py

Run with:
    pytest tests/test_embedding.py -v
"""

import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.file_parser import CodeFile
from utils.code_chunker import chunk_code_file, chunk_all_files, CodeChunk


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

def make_python_file(content: str) -> CodeFile:
    return CodeFile(
        path="test.py", absolute_path="/test.py",
        language="python", content=content,
        size_bytes=len(content), extension=".py",
    )

def make_ts_file(content: str) -> CodeFile:
    return CodeFile(
        path="test.ts", absolute_path="/test.ts",
        language="typescript", content=content,
        size_bytes=len(content), extension=".ts",
    )


# ─────────────────────────────────────────────
# Python AST chunking
# ─────────────────────────────────────────────

PYTHON_WITH_FUNCTIONS = """\
import os

X = 1

def hello():
    return "hello"

def add(a, b):
    return a + b

class MyClass:
    def method(self):
        pass
"""

def test_python_creates_function_chunks():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    types  = [c.chunk_type for c in chunks]
    assert "function" in types

def test_python_creates_class_chunk():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    types  = [c.chunk_type for c in chunks]
    assert "class" in types

def test_python_creates_module_level_chunk():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    types  = [c.chunk_type for c in chunks]
    assert "module_level" in types

def test_python_chunk_names_captured():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    names  = [c.name for c in chunks]
    assert "hello" in names
    assert "add"   in names
    assert "MyClass" in names

def test_python_chunk_ids_unique():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    ids    = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))

def test_python_chunk_content_contains_def():
    file   = make_python_file(PYTHON_WITH_FUNCTIONS)
    chunks = chunk_code_file(file)
    fn_chunks = [c for c in chunks if c.chunk_type == "function"]
    for c in fn_chunks:
        assert "def " in c.content

def test_python_syntax_error_falls_back_to_text():
    bad_python = "def broken(\n  no closing paren"
    file   = make_python_file(bad_python)
    chunks = chunk_code_file(file)
    # Should still produce chunks (text fallback)
    assert len(chunks) > 0
    assert all(c.chunk_type == "text_split" for c in chunks)


# ─────────────────────────────────────────────
# Text-based chunking (TypeScript)
# ─────────────────────────────────────────────

TS_CONTENT = "x" * 2500   # 2500 chars

def test_ts_produces_text_split_chunks():
    file   = make_ts_file(TS_CONTENT)
    chunks = chunk_code_file(file, chunk_size=1000, overlap=200)
    assert all(c.chunk_type == "text_split" for c in chunks)

def test_ts_chunk_count():
    file   = make_ts_file(TS_CONTENT)
    # 2500 chars, step=800 (1000-200 overlap): chunks at 0, 800, 1600, 2400 → 4
    chunks = chunk_code_file(file, chunk_size=1000, overlap=200)
    assert len(chunks) == 4

def test_ts_chunk_overlap():
    content = "A" * 1000 + "B" * 1000
    file    = make_ts_file(content)
    chunks  = chunk_code_file(file, chunk_size=1000, overlap=200)
    # Second chunk should start 800 chars in, so it includes the last 200 A's
    assert chunks[1].content[:200] == "A" * 200

def test_ts_chunk_ids_unique():
    file   = make_ts_file(TS_CONTENT)
    chunks = chunk_code_file(file, chunk_size=1000, overlap=200)
    ids    = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


# ─────────────────────────────────────────────
# chunk_all_files
# ─────────────────────────────────────────────

def test_chunk_all_files_aggregates():
    f1 = make_python_file("def a(): pass\ndef b(): pass\n")
    f2 = make_ts_file("const x = 1;\n")
    chunks = chunk_all_files([f1, f2])
    file_paths = {c.file_path for c in chunks}
    assert "test.py" in file_paths
    assert "test.ts" in file_paths

def test_chunk_all_files_empty():
    assert chunk_all_files([]) == []

def test_chunk_metadata_fields():
    file   = make_python_file("def foo():\n    return 1\n")
    chunks = chunk_code_file(file)
    for c in chunks:
        assert isinstance(c.chunk_id,   str)
        assert isinstance(c.file_path,  str)
        assert isinstance(c.language,   str)
        assert isinstance(c.content,    str)
        assert isinstance(c.chunk_type, str)
        assert isinstance(c.start_line, int)
        assert isinstance(c.end_line,   int)
