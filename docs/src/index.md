```@meta
CurrentModule = RAGTools
```

# RAGTools

RAGTools.jl is a battle-tested package for building Retrieval-Augmented Generation (RAG) applications in Julia. Originally part of [PromptingTools.jl](https://svilupp.github.io/PromptingTools.jl/dev/), it has been carved out into a standalone package after proving its value in production use cases for over a year.

The package focuses on high-performance, in-memory RAG pipelines that leverage Julia's speed to avoid the complexity of cloud-hosted vector databases. It seamlessly integrates with PromptingTools.jl to support a wide range of AI models and providers. However, if you need vector database support, you can simply overload the necessary functions in the pipeline (see the [RAG Interface](interface.md) for more details).

If you're curious about some specific features, check out the [Features](#features) section at the bottom of the page.

## Quick Start

Import the package:
```julia
using RAGTools
```

Key functions:
- `build_index`: Create a RAG index from documents (returns `ChunkIndex`)
- `airag`: Generate answers using RAG (combines `retrieve` and `generate!`)
- `retrieve`: Find relevant document chunks for a question
- `generate!`: Create an answer from retrieved chunks
- `annotate_support`: Highlight which parts of answers are supported by documents
- `build_qa_evals`: Generate question-answer pairs for evaluating RAG performance

## Basic Example

Index some documents:
```julia
# Create sample documents
sentences = [
    "Find the most comprehensive guide on Julia programming language for beginners published in 2023.",
    "Search for the latest advancements in quantum computing using Julia language.",
    "How to implement machine learning algorithms in Julia with examples.",
    "Looking for performance comparison between Julia, Python, and R for data analysis.",
    "Find Julia language tutorials focusing on high-performance scientific computing.",
    "Search for the top Julia language packages for data visualization and their documentation.",
    "How to set up a Julia development environment on Windows 10.",
    "Discover the best practices for parallel computing in Julia.",
    "Search for case studies of large-scale data processing using Julia.",
    "Find comprehensive resources for mastering metaprogramming in Julia.",
    "Looking for articles on the advantages of using Julia for statistical modeling.",
    "How to contribute to the Julia open-source community: A step-by-step guide.",
    "Find the comparison of numerical accuracy between Julia and MATLAB.",
    "Looking for the latest Julia language updates and their impact on AI research.",
    "How to efficiently handle big data with Julia: Techniques and libraries.",
    "Discover how Julia integrates with other programming languages and tools.",
    "Search for Julia-based frameworks for developing web applications.",
    "Find tutorials on creating interactive dashboards with Julia.",
    "How to use Julia for natural language processing and text analysis.",
    "Discover the role of Julia in the future of computational finance and econometrics."
]

# Build the index
index = build_index(sentences; 
    chunker_kwargs=(; sources=map(i -> "Doc$i", 1:length(sentences))))
```

Generate an answer:
```julia
# Simple query
question = "What are the best practices for parallel computing in Julia?"
msg = airag(index; question)

# Get detailed results including intermediate steps (see RAG Interface section to understand the different steps)
result = airag(index; question, return_all=true)

# Pretty print with support annotations
pprint(result)
```

## Extending the Pipeline

The package is designed to be modular and extensible:

1. Use the default pipeline with `SimpleIndexer`:
```julia
index = build_index(SimpleIndexer(), sentences)
```

2. Or customize any step by implementing your own methods:
```julia
# Example structure of the pipeline
result = retrieve(index, question)  # Get relevant chunks
result = generate!(index, result)   # Generate answer
```

For more advanced usage, see the [RAG Interface](interface.md) section that describes the different steps in the pipeline and how to customize them.

## "Citation" Annotations

RAGTools provides powerful support annotation capabilities through its pretty-printing system. Use `pprint` to automatically analyze and display how well the generated answer is supported by the source documents:

```julia
pprint(result)
```

Example output (with color highlighting in terminal):
```plaintext
--------------------
QUESTION(s)
--------------------
- What are the best practices for parallel computing in Julia?

--------------------
ANSWER
--------------------
Some of the best practices for parallel computing in Julia include:[1,0.7]
- Using [3,0.4]`@threads` for simple parallelism[1,0.34]
- Utilizing `Distributed` module for more complex parallel tasks[1,0.19]
- Avoiding excessive memory allocation
- Considering task granularity for efficient workload distribution

--------------------
SOURCES
--------------------
1. Doc8
2. Doc15
3. Doc5
4. Doc2
5. Doc9
```

### Understanding the Output

The annotation system helps you validate the generated answers:

- **Color Coding**:
  - Uncolored text: High match with source documents
  - Blue text: Partial match with sources
  - Magenta text: No match (model-generated)
- **Source Citations**: `[3,0.4]` indicates source document #3 with 40% match score

For web applications, use `print_html` to generate HTML-formatted output with styling:
```julia
print_html(result)  # Great for Genie.jl/Stipple.jl applications
```

## Features

RAGTools.jl offers a rich set of features for building production-ready RAG applications:

1. **Simple One-Line RAG**
- Quick setup with `build_index` and `airag` functions
- Default pipeline with semantic search and basic generation
- Seamless integration with PromptingTools.jl for various AI models

2. **Flexible Pipeline Components**
- Modular pipeline with consistent step names (`retrieve`, `rerank`, etc.)
- Each step dispatches on custom types (e.g., inherit from `AbstractRetriever`, `AbstractReranker`, etc.) 
- Easy to extend by implementing new types and the corresponding methods without changing core pipeline
- Dispatching kwarg & configuration always passed as first argument for maximum flexibility

#### Retrieval Options
- **Semantic Search**
  - Cosine similarity with dense embeddings
  - BM25 text similarity for keyword-based search
  - Binary embeddings with Hamming distance for efficiency
  - Bit-packed binary embeddings for maximum space efficiency
  - Hybrid indices combining multiple similarity methods

#### Advanced Retrieval Features
- **Query Enhancement**
  - HYDE (Hypothetical Document Embedding) for query rephrasing
  - Multiple query variations for better coverage
  
- **Ranking & Fusion**
  - Reciprocal Rank Fusion for combining multiple rankings
  - Multiple ranking models:
    - Local ranking with FlashRank.jl
    - RankGPT for LLM-based reranking
    - Cohere Rerank API integration
    - Custom ranking model support

#### Document Processing
- **Chunking & Embedding**
  - Multiple chunking strategies
  - Batched embedding for efficiency
  - Support for various embedding models
  - Binary and bit-packed embedding compression
  - Embedding dimension truncation

- **Tagging & Filtering**
  - Tag-based filtering system
  - Custom tag generation support
  - Flexible tag matching strategies

3. **Generation & Refinement**
- Multiple generation strategies
- Answer refinement steps
- Customizable post-processing
- Support for various AI models through PromptingTools.jl

4. **Quality & Analysis**
- **Answer Support Analysis**
  - Automatic source citation with `[source_id, score]` format
  - Support score calculation using trigram matching
  - Color-coded fact-checking visualization:
    - Uncolored: High confidence match with sources
    - Blue: Partial match with sources
    - Magenta: No source support (model-generated)
  - Sentence-level support analysis
  - Support threshold customization
  - Automated citation placement
  - Source document tracking
- **Visual Validation**
  - Pretty printing with color-coded support levels
  - HTML output for web applications
  - Interactive source exploration
  - Support score distribution analysis
- **Evaluation Tools**
  - Automated QA pair generation for evaluation
  - Support coverage metrics
  - Source utilization analysis
  - Answer consistency checking

5. **Integration & Observability**
- JSON logging of results and conversations
- Integration with Spehulak.jl for RAG performance analysis
- Cost tracking across API calls
- Performance metrics and timing

6. **Utility Features**
- Tokenization utilities
- Text splitting functions
- Pretty printing with support annotations
- Batch processing utilities
- Cost tracking and optimization tools

7. **Extensibility**
- Modular pipeline design
- Custom component support
- Multiple pre-built configurations:
  - `SimpleRetriever`
  - `SimpleBM25Retriever`
  - `AdvancedRetriever`
- Easy integration with vector databases

8. **Performance Optimization**
- In-memory operation for speed
- Efficient binary embedding compression
- Batched operations for API calls
- Multi-threading support
- Memory-efficient data structures
