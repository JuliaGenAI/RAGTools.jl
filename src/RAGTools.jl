module RAGTools

using LinearAlgebra, SparseArrays, Unicode, Snowball
using PromptingTools
using HTTP, JSON3
using JSON3: StructTypes
using AbstractTrees
using AbstractTrees: PreOrderDFS
const PT = PromptingTools
using ProgressMeter

## Re-export PromptingTools
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

## export trigrams, trigrams_hashed, text_to_trigrams, text_to_trigrams_hashed
## export STOPWORDS, tokenize, split_into_code_and_sentences
# export merge_kwargs_nested
export getpropertynested, setpropertynested
include("utils.jl")

# eg, cohere_api, tavily_api, create_websearch
include("api_services.jl")

include("rag_interface.jl")

export ChunkIndex, ChunkKeywordsIndex, ChunkEmbeddingsIndex, CandidateChunks, RAGResult
export MultiIndex, SubChunkIndex, MultiCandidateChunks
include("types.jl")

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
