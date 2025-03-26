using RAGTools: ChunkEmbeddingsIndex, ChunkKeywordsIndex,
	MultiIndex,
	CandidateChunks,
	MultiCandidateChunks,
	AbstractCandidateChunks, DocumentTermMatrix,
	SubDocumentTermMatrix,
	document_term_matrix, HasEmbeddings,
	HasKeywords,
	ChunkKeywordsIndex, AbstractChunkIndex,
	AbstractDocumentIndex
using RAGTools: embeddings, chunks, tags, tags_vocab, sources,
	extras, positions, scores, parent,
	RAGResult, chunkdata, preprocess_tokens, tf,
	vocab, vocab_lookup, idf, doc_rel_length
using RAGTools: SubChunkIndex, indexid, indexids,
	translate_positions_to_parent
using PromptingTools: last_message, last_output


# Include files 
# --------------
include("chunk.jl")
include("document_term_matrix.jl")
include("index.jl")
include("rag_result.jl")



# Shared tests 
# ------------

@testset "getindex-CandidateChunks" begin
	# Initialize a ChunkEmbeddingsIndex with test data
	chunks_data = ["First chunk", "Second chunk", "Third chunk"]
	embeddings_data = rand(3, 3)  # Random matrix with 3 embeddings
	tags_data = sparse(Bool[1 1; 0 1; 1 0])  # Some arbitrary sparse matrix representation
	tags_vocab_data = ["tag1", "tag2"]
	chunk_sym = Symbol("TestChunkEmbeddingsIndex")
	test_chunk_index = ChunkEmbeddingsIndex(chunks = chunks_data,
		embeddings = embeddings_data,
		tags = tags_data,
		tags_vocab = tags_vocab_data,
		sources = ["test_source$i" for i in 1:3],
		id = chunk_sym)

	# Test to get chunks based on valid CandidateChunks
	candidate_chunks = CandidateChunks(index_id = chunk_sym,
		positions = [1, 3],
		scores = [0.1, 0.2])
	@test collect(test_chunk_index[candidate_chunks]) == ["First chunk", "Third chunk"]
	@test collect(test_chunk_index[candidate_chunks, :chunks, sorted = true]) ==
		  ["Third chunk", "First chunk"]
	@test collect(test_chunk_index[candidate_chunks, :scores]) == [0.1, 0.2]
	@test collect(test_chunk_index[candidate_chunks, :sources]) ==
		  ["test_source1", "test_source3"]
	@test collect(test_chunk_index[candidate_chunks, :embeddings]) ==
		  embeddings_data[:, [1, 3]]
	@test collect(test_chunk_index[candidate_chunks, :chunkdata]) ==
		  embeddings_data[:, [1, 3]]

	# Test with empty positions, which should result in an empty array
	candidate_chunks_empty = CandidateChunks(index_id = chunk_sym,
		positions = Int[],
		scores = Float32[])
	@test isempty(test_chunk_index[candidate_chunks_empty])
	@test isempty(test_chunk_index[candidate_chunks_empty, :chunks])
	@test isempty(test_chunk_index[candidate_chunks_empty, :embeddings])
	@test isempty(test_chunk_index[candidate_chunks_empty, :chunkdata])
	@test isempty(test_chunk_index[candidate_chunks_empty, :sources])

	# Test with positions out of bounds, should handle gracefully without errors
	candidate_chunks_oob = CandidateChunks(index_id = chunk_sym,
		positions = [10, -1],
		scores = [0.5, 0.6])
	@test_throws BoundsError test_chunk_index[candidate_chunks_oob]

	# Test with an incorrect index_id, which should also result in an empty array
	wrong_sym = Symbol("InvalidIndex")
	candidate_chunks_wrong_id = CandidateChunks(index_id = wrong_sym,
		positions = [1, 2],
		scores = [0.3, 0.4])
	@test isempty(test_chunk_index[candidate_chunks_wrong_id])
	@test isempty(test_chunk_index[candidate_chunks_wrong_id, :chunks])
	@test isempty(test_chunk_index[candidate_chunks_wrong_id, :embeddings])
	@test isempty(test_chunk_index[candidate_chunks_wrong_id, :chunkdata])
	@test size(test_chunk_index[candidate_chunks_wrong_id, :chunkdata]) == (0, 0) # check that it's an array to maintain type
	@test isempty(test_chunk_index[candidate_chunks_wrong_id, :sources])
	@test isempty(test_chunk_index[candidate_chunks_wrong_id, :scores])

	# Test when chunks are requested from a MultiIndex, only chunks from the corresponding ChunkEmbeddingsIndex should be returned
	another_chunk_index = ChunkEmbeddingsIndex(chunks = chunks_data,
		embeddings = nothing,
		tags = nothing,
		tags_vocab = nothing,
		sources = repeat(["another_source"], 3),
		id = Symbol("AnotherChunkEmbeddingsIndex"))
	test_multi_index = MultiIndex(indexes = [
		test_chunk_index,
		another_chunk_index,
	])
	@test collect(test_multi_index[candidate_chunks]) == ["First chunk", "Third chunk"]

	# Test when wrong index_id is used with MultiIndex, resulting in an empty array
	@test isempty(test_multi_index[candidate_chunks_wrong_id])

	# Test error case when trying to use a non-chunks field, should assert error as only :chunks field is supported
	@test_throws AssertionError test_chunk_index[candidate_chunks, :nonexistent_field]

	# Multi-Candidate CandidateChunks
	cc = MultiCandidateChunks(; index_ids = [:TestChunkIndex2, :TestChunkIndex1],
		positions = [2, 2], scores = [0.1, 0.4])
	ci1 = ChunkEmbeddingsIndex(id = :TestChunkIndex1,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"])
	ci2 = ChunkEmbeddingsIndex(id = :TestChunkIndex2,
		chunks = ["chunk1", "chunk2x"],
		sources = ["source1", "source2"])
	@test ci1[cc, :chunks] == ["chunk2"]
	@test ci1[cc, :scores] == [0.4]
	@test ci2[cc] == ["chunk2x"]
	@test Base.getindex(ci1, cc, :chunks; sorted = true) == ["chunk2"]
	@test Base.getindex(ci1, cc, :scores; sorted = true) == [0.4]
	@test Base.getindex(ci1, cc, :chunks; sorted = false) == ["chunk2"]
	@test Base.getindex(ci1, cc, :scores; sorted = false) == [0.4]

	# Wrong index
	cc_wrong = MultiCandidateChunks(index_ids = [:TestChunkIndex2xxx, :TestChunkIndex1xxx],
		positions = [2, 2], scores = [0.1, 0.4])
	@test isempty(ci1[cc_wrong])
	@test isempty(ci1[cc_wrong, :chunks])
	@test isempty(ci1[cc_wrong, :scores])

	# with MultiIndex
	mi = MultiIndex(; id = :multi, indexes = [ci1, ci2])
	@test mi[cc] == ["chunk2", "chunk2x"]  # default is sorted=true
	@test Base.getindex(mi, cc, :chunks; sorted = true) == ["chunk2", "chunk2x"]
	@test Base.getindex(mi, cc, :chunks; sorted = false) == ["chunk2", "chunk2x"]

	# with MultiIndex -- flip the order of indices
	mi = MultiIndex(; id = :multi, indexes = [ci2, ci1])
	@test mi[cc] == ["chunk2", "chunk2x"] # default is sorted=true
	@test Base.getindex(mi, cc, :chunks; sorted = true) == ["chunk2", "chunk2x"]
	@test Base.getindex(mi, cc, :chunks; sorted = false) == ["chunk2x", "chunk2"]
end

@testset "getindex-MultiCandidateChunks" begin
	chunks_data = ["First chunk", "Second chunk", "Third chunk"]
	test_chunk_index = ChunkEmbeddingsIndex(chunks = chunks_data,
		embeddings = nothing,
		tags = nothing,
		tags_vocab = nothing,
		sources = ["test_source$i" for i in 1:3],
		id = Symbol("TestChunkIndex"))

	# Test with correct index_id and positions, expect correct chunks and scores
	multi_candidate_chunks = MultiCandidateChunks(
		index_ids = [Symbol("TestChunkIndex"), Symbol("TestChunkIndex")],
		positions = [1, 3],
		scores = [0.5, 0.6])
	@test test_chunk_index[multi_candidate_chunks] == ["First chunk", "Third chunk"]
	@test test_chunk_index[multi_candidate_chunks, :scores] == [0.5, 0.6]

	# Test with sorted option, expect chunks and scores sorted by scores in descending order
	@test Base.getindex(test_chunk_index, multi_candidate_chunks, :chunks; sorted = true) ==
		  ["Third chunk", "First chunk"]
	@test Base.getindex(test_chunk_index, multi_candidate_chunks, :scores; sorted = true) ==
		  [0.6, 0.5]
	@test Base.getindex(
		test_chunk_index, multi_candidate_chunks, :chunks; sorted = false) ==
		  ["First chunk", "Third chunk"]
	@test Base.getindex(
		test_chunk_index, multi_candidate_chunks, :scores; sorted = false) ==
		  [0.5, 0.6]

	# Test with incorrect index_id, expect empty array
	wrong_multi_candidate_chunks = MultiCandidateChunks(
		index_ids = [Symbol("WrongIndex"), Symbol("WrongIndex")],
		positions = [1, 3],
		scores = [0.5, 0.6])
	@test isempty(test_chunk_index[wrong_multi_candidate_chunks])
	@test isempty(test_chunk_index[wrong_multi_candidate_chunks, :scores])
	@test isempty(test_chunk_index[wrong_multi_candidate_chunks, :chunks])
	@test isempty(test_chunk_index[wrong_multi_candidate_chunks, :sources])

	# Test with a mix of correct and incorrect index_ids, expect only chunks and scores from correct index_id
	mixed_multi_candidate_chunks = MultiCandidateChunks(
		index_ids = [Symbol("TestChunkIndex"), Symbol("WrongIndex")],
		positions = [2, 3],
		scores = [0.5, 0.6])
	@test test_chunk_index[mixed_multi_candidate_chunks] == ["Second chunk"]
	@test test_chunk_index[mixed_multi_candidate_chunks, :scores] == [0.5]
	@test test_chunk_index[mixed_multi_candidate_chunks, :sources] == ["test_source2"]

	## MultiIndex
	ci2 = ChunkEmbeddingsIndex(chunks = ["4", "5", "6"],
		embeddings = nothing,
		tags = nothing,
		tags_vocab = nothing,
		sources = ["other_source$i" for i in 1:3],
		id = Symbol("TestChunkIndex2"))
	mi = MultiIndex(; id = :multi, indexes = [test_chunk_index, ci2])
	mc1 = MultiCandidateChunks(
		index_ids = [Symbol("TestChunkIndex"), Symbol("TestChunkIndex2")],
		positions = [1, 3],  # Assuming chunks_data has only 3 elements, position 4 is out of bounds
		scores = [0.5, 0.7])
	## sorted=false by default (Dict-like where order isn't guaranteed)
	## sorting follows index order
	@test mi[mc1] == ["6", "First chunk"]
	@test Base.getindex(mi, mc1, :chunks; sorted = true) == ["6", "First chunk"]
	@test Base.getindex(mi, mc1, :sources; sorted = true) ==
		  ["other_source3", "test_source1"]
	@test Base.getindex(mi, mc1, :chunks; sorted = false) == ["First chunk", "6"]
	@test Base.getindex(mi, mc1, :sources; sorted = false) ==
		  ["test_source1", "other_source3"]
	##
	@test Base.getindex(mi, mc1, :scores; sorted = true) == [0.7, 0.5]
	@test Base.getindex(mi, mc1, :scores; sorted = false) == [0.5, 0.7]
	@test Base.getindex(mi, mc1, :chunks; sorted = false) == ["First chunk", "6"]
	@test Base.getindex(mi, mc1, :sources; sorted = false) ==
		  ["test_source1", "other_source3"]
	@test Base.getindex(mi, mc1, :scores; sorted = false) == [0.5, 0.7]
end
