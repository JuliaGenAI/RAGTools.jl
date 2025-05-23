using PromptingTools: TestEchoOpenAISchema
using RAGTools: ChunkIndex
using RAGTools: ContextEnumerator, NoRephraser, SimpleRephraser,
	HyDERephraser,
	CosineSimilarity, BinaryCosineSimilarity,
	MultiFinder, BM25Similarity,
	NoTagFilter, AllTagFilter, AnyTagFilter,
	SimpleRetriever, AdvancedRetriever
using RAGTools: AbstractRephraser, AbstractTagFilter,
	AbstractSimilarityFinder, AbstractReranker,
	RankGPTReranker
using RAGTools: find_closest, hamming_distance, find_tags,
	rerank, rephrase,
	retrieve, HasEmbeddings, MultiCandidateChunks,
	CandidateChunks
using RAGTools: NoReranker, CohereReranker, ReciprocalRankFusionReranker,
	reciprocal_rank_fusion
using RAGTools: hamming_distance, BitPackedCosineSimilarity,
	pack_bits, unpack_bits
using RAGTools: bm25, max_bm25_score, document_term_matrix,
	DocumentTermMatrix

@testset "rephrase" begin
	# Test rephrase with NoRephraser, simple passthrough
	@test rephrase(NoRephraser(), "test") == ["test"]

	# Test rephrase with SimpleRephraser
	response = Dict(
		:choices => [
			Dict(:message => Dict(:content => "new question"), :finish_reason => "stop"),
		],
		:usage => Dict(:total_tokens => 3,
			:prompt_tokens => 2,
			:completion_tokens => 1))
	schema = TestEchoOpenAISchema(; response, status = 200)
	PT.register_model!(; name = "mock-gen", schema)
	output = rephrase(
		SimpleRephraser(), "old question", model = "mock-gen")
	@test output == ["old question", "new question"]

	output = rephrase(
		HyDERephraser(), "old question", model = "mock-gen")
	@test output == ["old question", "new question"]

	# with unknown rephraser
	struct UnknownRephraser123 <: AbstractRephraser end
	@test_throws ArgumentError rephrase(UnknownRephraser123(), "test question")
end

@testset "hamming_distance" begin

	## ORIGINAL TESTS
	# Test for matching number of rows
	@test_throws ArgumentError hamming_distance(
		[true false; false true], [true, false, true])

	# Test for correct calculation of distances
	@test hamming_distance([true false; false true], [true, false]) == [0, 2]
	@test hamming_distance([true false; false true], [false, true]) == [2, 0]
	@test hamming_distance([true false; false true], [true, true]) == [1, 1]
	@test hamming_distance([true false; false true], [false, false]) == [1, 1]

	## NEW TESTS
	# Test for Bool vectors
	vec1 = Bool[1, 0, 1, 0, 1, 0, 1, 0]
	vec2 = Bool[0, 1, 0, 1, 0, 1, 0, 1]
	# Basic functionality
	@test hamming_distance(vec1, vec2) == 8

	# Edge cases
	vec3 = Bool[1, 1, 1, 1, 1, 1, 1, 1]
	vec4 = Bool[0, 0, 0, 0, 0, 0, 0, 0]
	@test hamming_distance(vec3, vec4) == 8

	vec5 = Bool[1, 1, 1, 1, 1, 1, 1, 1]
	vec6 = Bool[1, 1, 1, 1, 1, 1, 1, 1]
	@test hamming_distance(vec5, vec6) == 0

	# Test for UInt64 (bitpacked) vectors
	vec7 = pack_bits(repeat(vec1, 8))
	vec8 = pack_bits(repeat(vec2, 8))
	@test hamming_distance(vec7, vec8) == 64

	vec9 = pack_bits(repeat(vec3, 8))
	vec10 = pack_bits(repeat(vec4, 8))
	@test hamming_distance(vec9, vec10) == 64

	vec11 = pack_bits(repeat(vec5, 8))
	vec12 = pack_bits(repeat(vec6, 8))
	@test hamming_distance(vec11, vec12) == 0

	# Test for Bool matrices
	mat1 = [vec1 vec2]
	mat2 = [vec3 vec4]
	@test hamming_distance(mat1, vec2) == [8, 0]
	@test hamming_distance(mat2, vec3) == [0, 8]

	# Test for UInt64 (bitpacked) matrices
	mat3 = pack_bits(repeat(mat1; outer = 8))
	mat4 = pack_bits(repeat(mat2; outer = 8))
	@test hamming_distance(mat3, vec8) == [64, 0]
	@test hamming_distance(mat4, vec9) == [0, 64]

	# Test for mismatched dimensions
	vec13 = Bool[1, 0, 1]
	@test_throws ArgumentError hamming_distance(mat1, vec13)

	# Additional edge cases
	# Empty vectors
	vec_empty1 = Bool[]
	vec_empty2 = Bool[]
	@test hamming_distance(vec_empty1, vec_empty2) == 0

	# Single element vectors
	vec_single1 = Bool[1]
	vec_single2 = Bool[0]
	@test hamming_distance(vec_single1, vec_single2) == 1

	# Large vectors
	vec_large1 = Bool[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
	vec_large2 = Bool[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
		0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
	@test hamming_distance(vec_large1, vec_large2) == 32

	# Large vectors with bitpacking
	vec_large_packed1 = pack_bits(repeat(vec_large1, 2))
	vec_large_packed2 = pack_bits(repeat(vec_large2, 2))
	@test hamming_distance(vec_large_packed1, vec_large_packed2) == 64

	## Compare packed vs binary results
	mat_rand1 = rand(Bool, 128, 10)
	q_rand2 = rand(Bool, 128)
	hamming_dist_binary = hamming_distance(mat_rand1, q_rand2)
	hamming_dist_packed = hamming_distance(pack_bits(mat_rand1), pack_bits(q_rand2))
	@test hamming_dist_binary == hamming_dist_packed
end

@testset "bm25" begin
	# Simple case
	documents = [["this", "is", "a", "test"],
		["this", "is", "another", "test"], ["foo", "bar", "baz"]]
	dtm = document_term_matrix(documents)
	query = ["this"]
	scores = bm25(dtm, query)
	idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
	tf = 1
	expected = idf * (tf * (1.2 + 1)) /
			   (tf + 1.2 * (1 - 0.75 + 0.75 * 4 / 3.666666666666667))
	@test scores[1] ≈ expected
	@test scores[2] ≈ expected
	@test scores[3] ≈ 0

	# Two words, both existing
	query = ["this", "test"]
	scores = bm25(dtm, query)
	@test scores[1] ≈ expected * 2
	@test scores[2] ≈ expected * 2
	@test scores[3] ≈ 0

	# Multiwords with no hits
	query = ["baz", "unknown", "words", "xyz"]
	scores = bm25(dtm, query)
	idf = log(1 + (3 - 1 + 0.5) / (1 + 0.5))
	tf = 1
	expected = idf * (tf * (1.2 + 1)) /
			   (tf + 1.2 * (1 - 0.75 + 0.75 * 3 / 3.666666666666667))
	@test scores[1] ≈ 0
	@test scores[2] ≈ 0
	@test scores[3] ≈ expected

	# Edge case: empty query
	@test bm25(dtm, String[]) == zeros(Float32, size(dtm.tf, 1))

	# Edge case: query with no matches
	query = ["food", "bard"]
	@test bm25(dtm, query) == zeros(Float32, size(dtm.tf, 1))

	# Edge case: query with multiple matches and repeats
	query = ["this", "is", "this", "this"]
	scores = bm25(dtm, query)
	idf = log(1 + (3 - 2 + 0.5) / (2 + 0.5))
	tf = 1
	expected = idf * (tf * (1.2 + 1)) /
			   (tf + 1.2 * (1 - 0.75 + 0.75 * 4 / 3.666666666666667))
	@test scores[1] ≈ expected * 4
	@test scores[2] ≈ expected * 4
	@test scores[3] ≈ 0

	## BM25 normalization
	# Basic test corpus
	documents = [
		["this", "is", "a", "test", "document"],
		["this", "is", "another", "test"],
		["completely", "different", "content"],
		["test", "test", "test", "test"],  # document with repeated terms
		["single"],  # shortest document
	]
	dtm = document_term_matrix(documents)

	# Test 1: Basic normalization - scores should be between 0 and 1
	query = ["test"]
	rel_len = RT.doc_rel_length(dtm)
	scores_norm = bm25(dtm, query; normalize = true, normalize_max_tf = 3,
		normalize_min_doc_rel_length = minimum(rel_len))
	@test all(0 .≤ scores_norm .≤ 1)

	# Test that document with most "test" occurrences gets highest score
	@test argmax(scores_norm) == 4

	# Test 2: Compare with manual normalization
	scores_raw = bm25(dtm, query; normalize = false)
	max_score = max_bm25_score(
		dtm, query; max_tf = 3, min_doc_rel_length = minimum(rel_len))
	scores_manual_norm = scores_raw ./ max_score
	@test scores_norm ≈ scores_manual_norm

	# Test 3: Parameter variations
	params = [
		(k1 = 1.2f0, b = 0.75f0, max_tf = 3, min_doc_len = 0.5f0),
		(k1 = 2.0f0, b = 0.5f0, max_tf = 10, min_doc_len = 1.0f0),
	]

	for p in params
		scores = bm25(dtm, query;
			normalize = true,
			k1 = p.k1,
			b = p.b,
			normalize_max_tf = p.max_tf,
			normalize_min_doc_rel_length = p.min_doc_len,
		)
		@test all(0 .≤ scores .≤ 1)

		# Verify against max_bm25_score
		max_theoretical = max_bm25_score(dtm, query;
			k1 = p.k1,
			b = p.b,
			max_tf = p.max_tf,
			min_doc_rel_length = p.min_doc_len,
		)
		scores_raw = bm25(dtm, query;
			normalize = false,
			k1 = p.k1,
			b = p.b,
		)
		@test maximum(scores_raw) ≤ max_theoretical
	end

	# Test 4: Edge cases
	# Empty query
	@test all(bm25(dtm, String[]; normalize = true) .== 0)

	# Query with non-existent words
	@test all(bm25(dtm, ["nonexistent"]; normalize = true) .== 0)

	# Multiple query terms
	multi_query = ["test", "document"]
	multi_scores = bm25(dtm, multi_query; normalize = true)
	@test all(0 .≤ multi_scores .≤ 1)
	# Document 1 should have highest score as it contains both terms
	@test argmax(multi_scores) == 1

	# Test 5: Repeated terms in query
	repeated_query = ["test", "test", "test"]
	rep_scores = bm25(dtm, repeated_query; normalize = true)
	@test all(0 .≤ rep_scores .≤ 1)

	# Test 6: Special cases - uniform document length
	uniform_docs = [["word", "test"] for _ in 1:3]
	uniform_dtm = document_term_matrix(uniform_docs)
	uniform_scores = bm25(uniform_dtm, ["test"]; normalize = true,
		normalize_max_tf = 1, normalize_min_doc_rel_length = 1.0f0)
	@test all(uniform_scores .≈ 1.0)

	# Test 7: Verify normalization with different max_tf values
	high_tf_docs = [
		["test", "test", "test"],  # tf = 3
		["test"],                  # tf = 1
		["other", "words"],         # tf = 0
	]
	high_tf_dtm = document_term_matrix(high_tf_docs)

	# With max_tf = 1 (matching actual tf in your dataset)
	scores_max1 = bm25(high_tf_dtm, ["test"]; normalize = true, normalize_max_tf = 1)
	# With max_tf = 3  (default)
	scores_max3 = bm25(high_tf_dtm, ["test"]; normalize = true, normalize_max_tf = 3)

	# The first document should get a lower relative score with max_tf=3 (max will be higher!)
	@test scores_max3[1] < scores_max1[1]
end

@testset "find_closest" begin
	finder = CosineSimilarity()
	test_embeddings = [1.0 2.0 -1.0; 3.0 4.0 -3.0; 5.0 6.0 -6.0] |>
					  x -> mapreduce(normalize, hcat, eachcol(x))
	query_embedding = [0.1, 0.35, 0.5] |> normalize
	positions, distances = find_closest(finder, test_embeddings, query_embedding, top_k = 2)
	# The query vector should be closer to the first embedding
	@test positions == [1, 2]
	@test isapprox(distances, [0.9975694083904584
			0.9939123761133188], atol = 1e-3)

	# Test when top_k is more than available embeddings
	positions, _ = find_closest(finder, test_embeddings, query_embedding, top_k = 5)
	@test length(positions) == size(test_embeddings, 2)

	# Test with minimum_similarity
	positions, _ = find_closest(finder, test_embeddings, query_embedding, top_k = 5,
		minimum_similarity = 0.995)
	@test length(positions) == 1

	# Test behavior with edge values (top_k == 0)
	@test find_closest(finder, test_embeddings, query_embedding, top_k = 0) == ([], [])

	## Test with ChunkIndex
	embeddings1 = ones(Float32, 2, 2)
	embeddings1[2, 2] = 5.0
	embeddings1 = mapreduce(normalize, hcat, eachcol(embeddings1))
	ci1 = ChunkIndex(id = :TestChunkIndex1,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"],
		embeddings = embeddings1)
	ci2 = ChunkIndex(id = :TestChunkIndex2,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"],
		embeddings = ones(Float32, 2, 2))
	ci3 = ChunkIndex(id = :TestChunkIndex3,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"],
		embeddings = nothing)

	## find_closest with ChunkIndex
	query_emb = [0.5, 0.5] # Example query embedding vector
	result = find_closest(finder, ci1, query_emb)
	@test result isa CandidateChunks
	@test result.positions == [1, 2]
	@test all(1.0 .>= result.scores .>= -1.0)   # Assuming default minimum_similarity

	## test with high minimum similarity
	result_high = find_closest(finder, ci1, query_emb; minimum_similarity = 0.99)
	@test isempty(result_high.positions)
	@test isempty(result_high.scores)
	@test result_high.index_id == :TestChunkIndex1

	## empty index
	query_emb = [0.5, 0.5] # Example query embedding vector
	result = find_closest(finder, ci3, query_emb)
	@test isempty(result)

	## Unknown type
	struct RandomSimilarityFinder123 <: AbstractSimilarityFinder end
	@test_throws ArgumentError find_closest(
		RandomSimilarityFinder123(), ones(5, 5), ones(5))

	## find_closest with multiple embeddings
	query_emb = [0.5 0.5; 0.5 1.0] |> x -> mapreduce(normalize, hcat, eachcol(x))
	result = find_closest(finder, ci1, query_emb; top_k = 2)
	@test result.positions == [1, 2]
	@test isapprox(result.scores, [1.0, 0.965], atol = 1e-2)

	# bad top_k -- too low, leads to 0 results
	result = find_closest(finder, ci1, query_emb; top_k = 1)
	@test isempty(result)
	# but it works in general, because 1/1 = 1 is a valid top_k
	result = find_closest(finder, ci1, query_emb[:, 1]; top_k = 1)
	@test result.positions == [1]
	@test result.scores == [1.0]

	### For Binary embeddings
	# Test for correct retrieval of closest positions and scores
	emb = [true false; false true]
	query_emb = [true, false]
	positions, scores = find_closest(BinaryCosineSimilarity(), emb, query_emb)
	@test positions == [1, 2]
	@test scores ≈ [1, 0] #query_emb' * emb[:, positions]

	query_emb = [0.5, -0.5]
	positions, scores = find_closest(BinaryCosineSimilarity(), emb, query_emb)
	@test positions == [1, 2]
	@test scores ≈ [0.5, -0.5] #query_emb' * emb[:, positions]

	# Test for custom top_k and minimum_similarity values
	positions, scores = find_closest(
		BinaryCosineSimilarity(), emb, query_emb; top_k = 1, minimum_similarity = 0.5)
	@test positions == [1]
	@test scores ≈ [0.5]

	positions, scores = find_closest(
		BinaryCosineSimilarity(), emb, query_emb; top_k = 1, minimum_similarity = 0.6)
	@test isempty(positions)
	@test isempty(scores)

	### Sense check for approximate methods

	# Generate random embeddings as a sense check
	Random.seed!(1234)  # For reproducibility
	emb = mapreduce(normalize, hcat, eachcol(randn(128, 1000)))
	query_emb = randn(128) |> normalize  # Normalize the query embedding

	# Calculate positions and scores using normal CosineSimilarity
	positions_cosine, scores_cosine = find_closest(
		CosineSimilarity(), emb, query_emb; top_k = 10)

	# Calculate positions and scores using BinaryCosineSimilarity
	binary_emb = map(>(0), emb)
	positions_binary, scores_binary = find_closest(
		BinaryCosineSimilarity(), binary_emb, query_emb; top_k = 10)
	@test length(intersect(positions_cosine, positions_binary)) >= 1

	# Calculate positions and scores using BinaryCosineSimilarity
	packed_emb = pack_bits(binary_emb)
	positions_packed, scores_packed = find_closest(
		BitPackedCosineSimilarity(), packed_emb, query_emb; top_k = 10)
	@test length(intersect(positions_cosine, positions_packed)) >= 1
end

## find_closest with MultiIndex
## mi = MultiIndex(id = :multi, indexes = [ci1, ci2])
## query_emb = [0.5, 0.5] # Example query embedding vector
## result = find_closest(mi, query_emb)
## @test result isa CandidateChunks
## @test result.positions == [1, 2]
## @test all(1.0 .>= result.distances .>= -1.0)   # Assuming default minimum_similarity

@testset "find_closest-MultiIndex" begin
	# Create mock data for testing
	emb1 = [0.1 0.2; 0.3 0.4; 0.5 0.6] |> x -> mapreduce(normalize, hcat, eachcol(x))
	emb2 = [0.7 0.8; 0.9 1.0; 1.1 1.2] |> x -> mapreduce(normalize, hcat, eachcol(x))
	query_emb = [0.1, 0.2, 0.3] |> normalize

	# Create ChunkIndex instances
	index1 = ChunkEmbeddingsIndex(id = :index1, chunks = ["chunk1", "chunk2"],
		embeddings = emb1, sources = ["source1", "source2"])
	index2 = ChunkEmbeddingsIndex(id = :index2, chunks = ["chunk3", "chunk4"],
		embeddings = emb2, sources = ["source3", "source4"])

	# Create MultiIndex instance
	multi_index = MultiIndex(id = :multi, indexes = [index1, index2])

	# Create MultiFinder instance
	multi_finder = MultiFinder([CosineSimilarity(), CosineSimilarity()])
	@test length(multi_finder) == 2

	# Perform find_closest with MultiFinder
	result = find_closest(multi_finder, multi_index, query_emb; top_k = 2)
	@test result isa MultiCandidateChunks
	@test result.index_ids == [:index1, :index2]
	@test result.positions == [2, 1]
	@test query_emb' * emb1[:, 2] ≈ result.scores[1]
	@test query_emb' * emb2[:, 1] ≈ result.scores[2]
	# Check that the positions and scores are sorted correctly
	@test result.scores[1] >= result.scores[2]

	## Get all results
	result = find_closest(multi_finder, multi_index, query_emb; top_k = 20)
	@test length(result.index_ids) == 4
	@test length(result.positions) == 4
	@test length(result.scores) == 4

	# Broadcast uni-finder without multi-finder
	result = find_closest(CosineSimilarity(), multi_index, query_emb; top_k = 20)
	@test length(result.index_ids) == 4
	@test length(result.positions) == 4
	@test length(result.scores) == 4

	## No embeddings
	index1 = ChunkEmbeddingsIndex(id = :index1, chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"])
	index2 = ChunkEmbeddingsIndex(id = :index2, chunks = ["chunk3", "chunk4"],
		sources = ["source3", "source4"])
	result = find_closest(MultiFinder([CosineSimilarity(), CosineSimilarity()]),
		MultiIndex(id = :multi, indexes = [index1, index2]), query_emb; top_k = 20)
	@test isempty(result.index_ids)
	@test isempty(result.positions)
	@test isempty(result.scores)

	### With mixed index types
	# Create mock data for testing
	emb1 = [0.1 0.2; 0.3 0.4; 0.5 0.6] |> x -> mapreduce(normalize, hcat, eachcol(x))
	query_emb = [0.1, 0.2, 0.3] |> normalize
	query_keywords = ["example", "query"]

	# Create ChunkIndex instances
	index1 = ChunkEmbeddingsIndex(id = :index1, chunks = ["chunk1", "chunk2"],
		embeddings = emb1, sources = ["source1", "source2"])
	index2 = ChunkKeywordsIndex(id = :index2, chunks = ["chunk3", "chunk4"],
		chunkdata = document_term_matrix([["example", "query"], ["random", "words"]]),
		sources = ["source3", "source4"])

	# Create MultiIndex instance
	multi_index = MultiIndex(id = :multi, indexes = [index1, index2])

	# Create MultiFinder instance
	multi_finder = MultiFinder([CosineSimilarity(), BM25Similarity()])

	# Perform find_closest with MultiFinder
	result = find_closest(multi_finder, multi_index, query_emb, query_keywords; top_k = 2)
	@test result isa MultiCandidateChunks
	@test result.index_ids == [:index2, :index1]
	@test result.positions == [1, 2]
	@test isapprox(result.scores, [1.387, 1.0], atol = 1e-1)
	# Check that the positions and scores are sorted correctly
	@test result.scores[1] >= result.scores[2]

	result = find_closest(multi_finder, multi_index, query_emb, query_keywords; top_k = 20)
	@test length(result.index_ids) == 4
	@test length(result.positions) == 4
	@test length(result.scores) == 4

	@test HasEmbeddings(index1)
	@test !HasEmbeddings(index2)
	@test HasEmbeddings(multi_index)

	## Test with high minimum similarity
	result = find_closest(multi_finder, multi_index, query_emb, query_keywords;
		top_k = 20, minimum_similarity = 100.0)
	@test isempty(result.index_ids)
	@test isempty(result.positions)
	@test isempty(result.scores)
end

@testset "find_tags" begin
	tagger = AnyTagFilter()
	test_embeddings = [1.0 2.0; 3.0 4.0; 5.0 6.0] |>
					  x -> mapreduce(normalize, hcat, eachcol(x))
	query_embedding = [0.1, 0.35, 0.5] |> normalize
	test_tags_vocab = ["julia", "python", "jr"]
	test_tags_matrix = sparse([1, 2], [1, 3], [true, true], 2, 3)
	index = ChunkIndex(;
		id = :indexX,
		sources = [".", "."],
		chunks = ["julia", "jr"],
		embeddings = test_embeddings,
		tags = test_tags_matrix,
		tags_vocab = test_tags_vocab)

	# Test for finding the correct positions of a specific tag
	@test find_tags(tagger, index, "julia").positions == [1]
	@test find_tags(tagger, index, "julia").scores == [1.0]

	# Test for no tag found // not in vocab
	@test find_tags(tagger, index, "python").positions |> isempty
	@test find_tags(tagger, index, "java").positions |> isempty

	# Test with regex matching
	@test find_tags(tagger, index, r"^j").positions == [1, 2]

	# Test with multiple tags in vocab
	@test find_tags(tagger, index, ["python", "jr", "x"]).positions == [2]

	## With AllTagFilter -- no difference for individual
	tagger2 = AllTagFilter()
	@test find_tags(tagger2, index, "julia").positions == [1]
	@test find_tags(tagger2, index, "julia").scores == [1.0]
	@test find_tags(tagger2, index, "python").positions |> isempty
	@test find_tags(tagger2, index, "java").positions |> isempty
	@test find_tags(tagger2, index, r"^j").positions |> isempty
	@test find_tags(tagger2, index, "jr").positions == [2]

	@test find_tags(tagger2, index, ["python", "jr", "x"]).positions |> isempty
	@test find_tags(tagger2, index, ["julia", "jr"]).positions |> isempty
	@test find_tags(tagger2, index, ["julia", "julia"]).positions == [1]
	@test find_tags(tagger2, index, ["julia", "julia"]).scores == [1.0]

	# No filter tag -- give everything
	cc = find_tags(NoTagFilter(), index, "julia")
	@test isnothing(cc)
	# @test cc.positions == [1, 2]
	# @test cc.scores == [0.0, 0.0]

	cc = find_tags(NoTagFilter(), index, nothing)
	@test isnothing(cc)
	# @test cc.positions == [1, 2]
	# @test cc.scores == [0.0, 0.0]

	# Unknown type
	struct RandomTagFilter123 <: AbstractTagFilter end
	@test_throws ArgumentError find_tags(RandomTagFilter123(), index, "hello")
	@test_throws ArgumentError find_tags(RandomTagFilter123(), index, ["hello"])

	## Multi-index implementation
	emb1 = [0.1 0.2; 0.3 0.4; 0.5 0.6] |> x -> mapreduce(normalize, hcat, eachcol(x))
	index1 = ChunkEmbeddingsIndex(id = :index1, chunks = ["chunk1", "chunk2"],
		embeddings = emb1, sources = ["source1", "source2"])
	index2 = ChunkKeywordsIndex(id = :index2, chunks = ["chunk3", "chunk4"],
		chunkdata = document_term_matrix([["example", "query"], ["random", "words"]]),
		sources = ["source3", "source4"])

	# Create MultiIndex instance
	multi_index = MultiIndex(id = :multi, indexes = [index1, index2])

	mcc = find_tags(NoTagFilter(), multi_index, "julia")
	@test mcc == nothing
	# @test mcc.positions == [1, 2, 3, 4]
	# @test mcc.scores == [0.0, 0.0, 0.0, 0.0]

	mcc = find_tags(NoTagFilter(), multi_index, nothing)
	@test mcc == nothing
	# @test mcc.positions == [1, 2, 3, 4]
	# @test mcc.scores == [0.0, 0.0, 0.0, 0.0]

	multi_index2 = MultiIndex(id = :multi2, indexes = [index, index2])
	mcc2 = find_tags(AnyTagFilter(), multi_index2, "julia")
	@test mcc2.index_ids == [:indexX]
	@test mcc2.positions == [1]
	@test mcc2.scores == [1.0]

	mcc3 = find_tags(AnyTagFilter(), multi_index2, ["julia", "python", "jr"])
	@test mcc3.index_ids == [:indexX, :indexX]
	@test mcc3.positions == [1, 2]
	@test mcc3.scores == [1.0, 1.0]

	mcc4 = find_tags(AnyTagFilter(), multi_index2, [r"^j"])
	@test mcc4.index_ids == [:indexX, :indexX]
	@test mcc4.positions == [1, 2]
	@test mcc4.scores == [1.0, 1.0]

	mcc5 = find_tags(AllTagFilter(), multi_index2, [r"^j"])
	@test mcc5.index_ids |> isempty
	@test mcc5.positions |> isempty
	@test mcc5.scores |> isempty
end

@testset "reciprocal_rank_fusion" begin
	# Test basic functionality with positions and scores
	positions1 = [1, 3, 5, 7, 9]
	scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
	positions2 = [2, 4, 6, 8, 5]
	scores2 = [0.5, 0.6, 0.7, 0.8, 0.9]

	# Test the base function that works with positions and scores
	merged_positions, scores_dict = reciprocal_rank_fusion(
		positions1, scores1, positions2, scores2; k = 60)
	@test merged_positions[1] == 5
	@test scores_dict[5] > 0.0

	# Check that all positions are included in the merged result
	@test sort(merged_positions) == sort(unique(vcat(positions1, positions2)))

	# Check that scores are properly calculated and stored in dictionary
	@test length(keys(scores_dict)) == length(merged_positions)
	@test all(0.0 .<= values(scores_dict) .<= 1.0)

	# Test with CandidateChunks
	cc1 = CandidateChunks(:index1, positions1, scores1)
	cc2 = CandidateChunks(:index2, positions2, scores2)

	merged_cc = reciprocal_rank_fusion(cc1, cc2; k = 60)

	# Check the merged CandidateChunks
	@test merged_cc.index_id == :index1
	@test length(merged_cc.positions) == 9
	@test length(merged_cc.scores) == 9
	@test all(0.0 .<= merged_cc.scores .<= 1.0)

	# Test with MultiCandidateChunks
	mcc = MultiCandidateChunks(
		vcat(fill(:index1a, length(positions1)), fill(:index1b, length(positions2))),
		vcat(positions1, positions2),
		vcat(scores1, scores2))

	mcc_merged = reciprocal_rank_fusion(mcc; k = 60)

	# Check the merged MultiCandidateChunks
	@test mcc_merged.index_ids[1] == :index1a
	@test all(id -> id == :index1a, mcc_merged.index_ids)
	@test length(mcc_merged.positions) == 9
	@test length(mcc_merged.scores) == 9
	@test all(0.0 .<= mcc_merged.scores .<= 1.0)

	# Test with different k value
	mcc_merged_k10 = reciprocal_rank_fusion(mcc; k = 10)
	@test length(mcc_merged_k10.positions) == 9

	# Test assertion for MultiCandidateChunks with more than two indices
	mcc_three_indices = MultiCandidateChunks(
		vcat(fill(:index1a, 2), fill(:index1b, 2), fill(:index1c, 2)),
		vcat([1, 2], [3, 4], [5, 6]),
		vcat([0.9, 0.8], [0.7, 0.6], [0.5, 0.4]))

	@test_throws AssertionError reciprocal_rank_fusion(mcc_three_indices)

	# Test with ReciprocalRankFusionReranker
	reranker = ReciprocalRankFusionReranker(k = 60)
	multi_index = MultiIndex(id = :multi,
		indexes = [
			ChunkIndex(
				id = :index1, chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"],
				sources = String[]),
			ChunkIndex(
				id = :index2, chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"],
				sources = String[]),
		])

	multi_candidates = MultiCandidateChunks(
		vcat(fill(:index1, length(positions1)), fill(:index2, length(positions2))),
		vcat(positions1, positions2),
		vcat(scores1, scores2))

	question = "test question"

	# Test assertions in rerank function
	@test_throws AssertionError rerank(reranker,
		ChunkIndex(id = :single, chunks = String[], sources = String[]), question, multi_candidates)
	@test_throws AssertionError rerank(reranker,
		multi_index, question, multi_candidates; top_n = 0)
	@test_throws AssertionError rerank(reranker,
		multi_index, question, cc1)

	# Test successful reranking
	reranked = rerank(reranker, multi_index, question, multi_candidates; top_n = 5)
	@test length(reranked.positions) == 5
	@test length(reranked.scores) == 5
	@test all(0.0 .<= reranked.scores .<= 1.0)
end

@testset "rerank" begin
	# Mock data for testing
	ci1 = ChunkIndex(id = :TestChunkIndex1,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"])
	question = "mock_question"
	cc1 = CandidateChunks(index_id = :TestChunkIndex1,
		positions = [1, 2],
		scores = [0.3, 0.4])

	# Passthrough Strategy
	ranker = NoReranker()
	reranked = rerank(ranker, ci1, question, cc1)
	@test reranked.positions == [2, 1] # gets resorted by score
	@test reranked.scores == [0.4, 0.3]

	reranked = rerank(ranker, ci1, question, cc1; top_n = 1)
	@test reranked.positions == [2] # gets resorted by score
	@test reranked.scores == [0.4]

	ci2 = ChunkIndex(id = :TestChunkIndex2,
		chunks = ["chunk1", "chunk2"],
		sources = ["source1", "source2"])
	mi = MultiIndex(; id = :multi, indexes = [ci1, ci2])
	reranked = rerank(NoReranker(),
		mi,
		question,
		cc1)
	@test reranked.positions == [2, 1] # gets resorted by score
	@test reranked.scores == [0.4, 0.3]

	# Cohere assertion
	## @test reranked isa MultiCandidateChunks

	# Bad top_n
	@test_throws AssertionError rerank(CohereReranker(),
		ci1,
		question,
		cc1; top_n = 0)

	# Bad index_id
	cc2 = CandidateChunks(index_id = :TestChunkIndex2,
		positions = [1, 2],
		scores = [0.3, 0.4])
	@test_throws AssertionError rerank(CohereReranker(),
		ci1,
		question,
		cc2; top_n = 1)

	## Unknown type
	struct RandomReranker123 <: AbstractReranker end
	@test_throws ArgumentError rerank(RandomReranker123(), ci1, "hello", cc2)

	## TODO: add testing of Cohere reranker API call -- not done yet
end

@testset "retrieve" begin
	# test with a mock server
	PORT = rand(20000:40001)
	PT.register_model!(; name = "mock-emb", schema = PT.CustomOpenAISchema())
	PT.register_model!(; name = "mock-emb2", schema = PT.CustomOpenAISchema())
	PT.register_model!(; name = "mock-meta", schema = PT.CustomOpenAISchema())
	PT.register_model!(; name = "mock-gen", schema = PT.CustomOpenAISchema())

	echo_server = HTTP.serve!(PORT; verbose = -1) do req
		content = JSON3.read(req.body)

		if content[:model] == "mock-gen"
			user_msg = last(content[:messages])
			response = Dict(
				:choices => [
					Dict(:message => user_msg, :finish_reason => "stop"),
				],
				:model => content[:model],
				:usage => Dict(:total_tokens => length(user_msg[:content]),
					:prompt_tokens => length(user_msg[:content]),
					:completion_tokens => 0))
		elseif content[:model] == "mock-emb"
			response = Dict(:data => [Dict(:embedding => ones(Float32, 10))],
				:usage => Dict(:total_tokens => length(content[:input]),
					:prompt_tokens => length(content[:input]),
					:completion_tokens => 0))
		elseif content[:model] == "mock-emb2"
			response = Dict(
				:data => [Dict(:embedding => ones(Float32, 10)),
					Dict(:embedding => ones(Float32, 10))],
				:usage => Dict(:total_tokens => length(content[:input]),
					:prompt_tokens => length(content[:input]),
					:completion_tokens => 0))
		elseif content[:model] == "mock-meta"
			user_msg = last(content[:messages])
			response = Dict(
				:choices => [
					Dict(:finish_reason => "stop",
						:message => Dict(
							:tool_calls => [
								Dict(:id => "1",
									:function => Dict(:arguments => JSON3.write(MaybeTags([
										Tag("yes", "category"),
									]))))],
							:name => "MaybeTags"))],
				:model => content[:model],
				:usage => Dict(:total_tokens => length(user_msg[:content]),
					:prompt_tokens => length(user_msg[:content]),
					:completion_tokens => 0))
		else
			@info content
		end
		return HTTP.Response(200, JSON3.write(response))
	end

	embeddings1 = ones(Float32, 10, 4)
	embeddings1[10, 3:4] .= 5.0
	embeddings1 = mapreduce(normalize, hcat, eachcol(embeddings1))
	index = ChunkIndex(id = :TestChunkIndex1,
		chunks = ["chunk1", "chunk2", "chunk3", "chunk4"],
		sources = ["source1", "source2", "source3", "source4"],
		embeddings = embeddings1)
	question = "test question"

	## Test with SimpleRetriever
	simple = SimpleRetriever()

	result = retrieve(simple, index, question;
		rephraser_kwargs = (; model = "mock-gen"),
		embedder_kwargs = (; model = "mock-emb"),
		tagger_kwargs = (; model = "mock-meta"), api_kwargs = (;
			url = "http://localhost:$(PORT)"))
	@test result.question == question
	@test result.rephrased_questions == [question]
	@test result.answer == nothing
	@test result.final_answer == nothing
	## there are two equivalent orderings
	@test Set(result.reranked_candidates.positions[1:2]) == Set([2, 1])
	@test Set(result.reranked_candidates.positions[3:4]) == Set([3, 4])
	@test result.reranked_candidates.scores[1:2] == ones(2)
	@test length(result.context) == 4
	@test length(unique(result.context)) == 4
	@test result.context[1] in ["chunk2", "chunk1"]
	@test result.context[2] in ["chunk2", "chunk1"]
	@test result.context[3] in ["chunk3", "chunk4"]
	@test result.context[4] in ["chunk3", "chunk4"]
	@test result.sources isa Vector{String}

	# Reduce number of candidates
	result = retrieve(simple, index, question;
		top_n = 2, top_k = 3,
		rephraser_kwargs = (; model = "mock-gen"),
		embedder_kwargs = (; model = "mock-emb"),
		tagger_kwargs = (; model = "mock-meta"), api_kwargs = (;
			url = "http://localhost:$(PORT)"))
	## the last item is 3 or 4
	@test result.emb_candidates.positions[3] in [3, 4]
	@test Set(result.reranked_candidates.positions[1:2]) == Set([2, 1])
	@test result.emb_candidates.scores[1:2] == ones(2)

	# with default dispatch
	result = retrieve(index, question;
		top_n = 2, top_k = 3,
		rephraser_kwargs = (; model = "mock-gen"),
		embedder_kwargs = (; model = "mock-emb"),
		tagger_kwargs = (; model = "mock-meta"), api_kwargs = (;
			url = "http://localhost:$(PORT)"))
	@test result.emb_candidates.positions[3] in [3, 4]
	@test result.emb_candidates.scores[1:2] == ones(2)
	@test Set(result.reranked_candidates.positions[1:2]) == Set([2, 1])

	## AdvancedRetriever
	adv = AdvancedRetriever()
	result = retrieve(adv, index, question;
		reranker = NoReranker(), # we need to disable cohere as we cannot test it
		rephraser_kwargs = (; model = "mock-gen"),
		embedder_kwargs = (; model = "mock-emb2"),
		tagger_kwargs = (; model = "mock-meta"), api_kwargs = (;
			url = "http://localhost:$(PORT)"))
	@test result.question == question
	@test result.rephrased_questions == [question, "Query: test question\n\nPassage:"] # from the template we use
	@test result.answer == nothing
	@test result.final_answer == nothing
	## there are two equivalent orderings
	@test Set(result.reranked_candidates.positions[1:2]) == Set([2, 1])
	@test Set(result.reranked_candidates.positions[3:4]) == Set([3, 4])
	@test result.reranked_candidates.scores[1:2] == ones(2)
	@test length(result.context) == 4
	@test length(unique(result.context)) == 4
	@test result.context[1] in ["chunk2", "chunk1"]
	@test result.context[2] in ["chunk2", "chunk1"]
	@test result.context[3] in ["chunk3", "chunk4"]
	@test result.context[4] in ["chunk3", "chunk4"]
	@test result.sources isa Vector{String}

	# Multi-index retriever
	index_keywords = ChunkKeywordsIndex(index, index_id = :TestChunkIndexX)
	index_keywords = ChunkIndex(; id = :AA, index.chunks, index.sources, index.embeddings)
	# Create MultiIndex instance
	multi_index = MultiIndex(id = :multi, indexes = [index, index_keywords])

	# Create MultiFinder instance
	finder = MultiFinder([RT.CosineSimilarity(), RT.BM25Similarity()])

	retriever = SimpleRetriever(; processor = RT.KeywordsProcessor(), finder)
	result = retrieve(SimpleRetriever(), multi_index, question;
		reranker = NoReranker(), # we need to disable cohere as we cannot test it
		rephraser_kwargs = (; model = "mock-gen"),
		embedder_kwargs = (; model = "mock-emb"),
		tagger_kwargs = (; model = "mock-meta"), api_kwargs = (;
			url = "http://localhost:$(PORT)"))
	@test result.question == question
	@test result.rephrased_questions == [question]
	@test result.answer == nothing
	@test result.final_answer == nothing
	## there are two equivalent orderings
	@test Set(result.reranked_candidates.positions[1:4]) == Set([2, 1])
	@test result.reranked_candidates.positions[5] in [3, 4]
	@test result.reranked_candidates.scores[1:4] == ones(4)
	@test length(result.context) == 5 # because the second index duplicates, so we have more
	@test length(unique(result.context)) == 3 # only 3 unique chunks because 1,2,1,2,3
	@test all([result.context[i] in ["chunk2", "chunk1"] for i in 1:4])
	@test result.context[5] in ["chunk3", "chunk4"]
	@test length(unique(result.sources)) == 3
	@test all([result.sources[i] in ["source2", "source1"] for i in 1:4])
	@test result.sources[5] in ["source3", "source4"]

	# clean up
	close(echo_server)
end
