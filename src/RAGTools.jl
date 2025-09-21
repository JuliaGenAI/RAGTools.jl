module RAGTools

# Base Dependencies
# ------------------ 
using Base: parent

# External Dependencies
# ----------------------
using AbstractTrees
using AbstractTrees: PreOrderDFS
using HTTP, JSON3
using JSON3: StructTypes
using LinearAlgebra, SparseArrays, Unicode, Snowball
using ProgressMeter

# PromptingTools Dependencies
# --------------------------
using PromptingTools
const PT = PromptingTools

# re-export PromptingTools
using PromptingTools: aigenerate, aiembed, aiclassify, aiextract, aiscan, aiimage, @ai_str,
                      @aai_str, @ai!_str, @aai!_str
export aigenerate, aiembed, aiclassify, aiextract, aiscan, aiimage, @ai_str, @aai_str,
       @ai!_str, @aai!_str

using PromptingTools: ConversationMemory, aitemplates, AITemplate, AICode, pprint
export ConversationMemory, aitemplates, AITemplate, AICode, pprint

using PromptingTools: AbstractMessage, UserMessage, SystemMessage, AIMessage,
                      UserMessageWithImages, DataMessage, AIToolRequest, ToolMessage
export UserMessage, SystemMessage, UserMessageWithImages, DataMessage, AIToolRequest,
       ToolMessage, AbstractMessage, AIMessage

using PromptingTools: create_template, recursive_splitter
export create_template, recursive_splitter

# Files inclusion 
# ----------------
## export trigrams, trigrams_hashed, text_to_trigrams, text_to_trigrams_hashed
## export STOPWORDS, tokenize, split_into_code_and_sentences
# export merge_kwargs_nested
export getpropertynested, setpropertynested
include("utils.jl")

include("api_services.jl") # eg, cohere_api, tavily_api, create_websearch

include("rag_interface.jl")

export ChunkIndex, ChunkKeywordsIndex, ChunkEmbeddingsIndex, CandidateChunks, RAGResult
export MultiIndex, SubChunkIndex, MultiCandidateChunks
include("types/candidate_chunks.jl") # contains CandidateChunks and MultiCandidateChunks types for storing retrieval results
include("types/document_term_matrix.jl") # contains DocumentTermMatrix type for BM25 similarity search
include("types/index.jl") # contains ChunkIndex, MultiIndex and related types for storing document chunks
include("types/rag_result.jl") # contains RAGResult type for storing RAG pipeline results

export build_index, get_chunks, get_embeddings, get_keywords, get_tags, SimpleIndexer,
       KeywordsIndexer
include("preparation.jl")

include("rank_gpt.jl")

include("bm25.jl")

export retrieve, SimpleRetriever, SimpleBM25Retriever, AdvancedRetriever
export find_closest, find_tags, rerank, rephrase
include("retrieval.jl")

export airag, build_context!, generate!, refine!, answer!, postprocess!
export SimpleGenerator, AdvancedGenerator, RAGConfig
include("generation.jl")

export annotate_support, TrigramAnnotater, print_html
include("annotation.jl")

export build_qa_evals, run_qa_evals
include("evaluation.jl")

end # end of module
