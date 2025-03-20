using RAGTools: ChunkEmbeddingsIndex, 
                ChunkKeywordsIndex,
                MultiIndex,
                CandidateChunks,
                MultiCandidateChunks,
                AbstractCandidateChunks, 
                DocumentTermMatrix,
                SubDocumentTermMatrix,
                document_term_matrix, 
                HasEmbeddings,
                HasKeywords,
                ChunkKeywordsIndex, 
                AbstractChunkIndex,
                AbstractDocumentIndex
using RAGTools: embeddings, chunks, tags, tags_vocab, sources,
                extras, positions, scores, parent,
                RAGResult, chunkdata, preprocess_tokens, tf,
                vocab, vocab_lookup, idf, doc_rel_length
using RAGTools: SubChunkIndex, indexid, indexids,
                translate_positions_to_parent
using PromptingTools: last_message, last_output


include("chunk.jl")
include("document_matrix.jl")
include("index.jl")
include("rag_result.jl")
