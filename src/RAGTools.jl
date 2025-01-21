module RAGTools

using LinearAlgebra, SparseArrays, Unicode, Snowball
using PromptingTools
using PromptingTools.Experimental.RAGTools
using FlashRank

## Re-export PromptingTools
export aigenerate, aiembed, aiclassify, aiextract, aiscan, aiimage
export AIMessage
export aitemplates, AITemplate
export AICode

## Export RAG specific functionality
export getpropertynested, setpropertynested
export ChunkIndex, ChunkKeywordsIndex, ChunkEmbeddingsIndex, CandidateChunks, RAGResult
export MultiIndex
export build_index, get_chunks, get_embeddings, get_keywords, get_tags, SimpleIndexer,
       KeywordsIndexer
export retrieve, SimpleRetriever, SimpleBM25Retriever, AdvancedRetriever
export find_closest, find_tags, rerank, rephrase
export airag, build_context!, generate!, refine!, answer!, postprocess!
export SimpleGenerator, AdvancedGenerator, RAGConfig
export annotate_support, TrigramAnnotater, print_html
export build_qa_evals, run_qa_evals

end
