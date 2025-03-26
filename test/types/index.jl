
@testset "ChunkEmbeddingsIndex" begin
	# Test constructors and basic accessors
	chunks_test = ["chunk1", "chunk2"]
	emb_test = ones(2, 2)
	tags_test = sparse([1, 2], [1, 2], [true, true], 2, 2)
	tags_vocab_test = ["vocab1", "vocab2"]
	sources_test = ["source1", "source2"]
	ci = ChunkEmbeddingsIndex(chunks = chunks_test,
		embeddings = emb_test,
		tags = tags_test,
		tags_vocab = tags_vocab_test,
		sources = sources_test)

	@test chunks(ci) == chunks_test
	@test (embeddings(ci)) == emb_test
	@test (chunkdata(ci)) == emb_test
	@test chunkdata(ci, [1]) == view(emb_test, :, [1])
	@test tags(ci) == tags_test
	@test tags_vocab(ci) == tags_vocab_test
	@test sources(ci) == sources_test
	@test length(ci) == 2
	@test translate_positions_to_parent(ci, [2, 1]) == [2, 1]
	@test translate_positions_to_parent(ci, [4, 6]) == [4, 6]

	# Test identity/equality
	ci1 = ChunkEmbeddingsIndex(
		chunks = ["chunk1", "chunk2"], sources = ["source1", "source2"])
	ci2 = ChunkEmbeddingsIndex(
		chunks = ["chunk1", "chunk2"], sources = ["source1", "source2"])
	@test ci1 == ci2

	# Test equality with different chunks and sources
	ci2 = ChunkEmbeddingsIndex(
		chunks = ["chunk3", "chunk4"], sources = ["source3", "source4"])
	@test ci1 != ci2

	# Test hcat with ChunkEmbeddingsIndex
	# Setup two different ChunkEmbeddingsIndex with different tags and then hcat them
	chunks1 = ["chunk1", "chunk2"]
	tags1 = sparse([1, 2], [1, 2], [true, true], 2, 3)
	tags_vocab1 = ["vocab1", "vocab2", "vocab3"]
	sources1 = ["source1", "source1"]
	ci1 = ChunkEmbeddingsIndex(chunks = chunks1,
		tags = tags1,
		tags_vocab = tags_vocab1,
		sources = sources1)

	chunks2 = ["chunk3", "chunk4"]
	tags2 = sparse([1, 2], [1, 3], [true, true], 2, 3)
	tags_vocab2 = ["vocab1", "vocab3", "vocab4"]
	sources2 = ["source2", "source2"]
	ci2 = ChunkEmbeddingsIndex(chunks = chunks2,
		tags = tags2,
		tags_vocab = tags_vocab2,
		sources = sources2)

	combined_ci = vcat(ci1, ci2)
	@test size(tags(combined_ci), 1) == 4
	@test size(tags(combined_ci), 2) == 4
	@test length(unique(vcat(tags_vocab(ci1), tags_vocab(ci2)))) ==
		  length(tags_vocab(combined_ci))
	@test sources(combined_ci) == vcat(sources(ci1), (sources(ci2)))
	@test length(combined_ci) == 4
	@test chunkdata(combined_ci) == nothing
	@test chunkdata(combined_ci, [1]) == nothing

	# Test base var"==" with ChunkEmbeddingsIndex
	ci1 = ChunkEmbeddingsIndex(chunks = ["chunk1"],
		id = :ci1,
		tags = trues(3, 1),
		tags_vocab = ["vocab1"],
		sources = ["source1"])
	ci2 = ChunkEmbeddingsIndex(chunks = ["chunk1"],
		tags = trues(3, 1),
		tags_vocab = ["vocab1"],
		sources = ["source1"])
	@test ci1 == ci2

	# HasEmbeddings
	@test HasEmbeddings(ci1) == true
	@test HasKeywords(ci1) == false

	# Getindex
	@test ci1[:ci1] == ci1
	@test ci1[:ci2] == nothing

	## Test general accessors
	@kwdef struct TestBadMultiIndex <: AbstractDocumentIndex
		indices::Vector{AbstractChunkIndex} = [ChunkEmbeddingsIndex(
			chunks = ["chunk1"], sources = ["source1"])]
	end
	bad_idx = TestBadMultiIndex()
	@test_throws ArgumentError chunkdata(bad_idx)
	@test_throws ArgumentError embeddings(bad_idx)
	@test_throws ArgumentError tags(bad_idx)
	@test_throws ArgumentError tags_vocab(bad_idx)
	@test_throws ArgumentError extras(bad_idx)

	@kwdef struct TestBadChunkIndex <: AbstractChunkIndex
		chunks::Vector{String}
		sources::Vector{String}
	end
	bad_chunk_idx = TestBadChunkIndex(chunks = ["chunk1"], sources = ["source1"])
	@test_throws ArgumentError embeddings(bad_chunk_idx)
end

@testset "ChunkKeywordsIndex" begin
	# Test creation of ChunkKeywordsIndex
	chunks_ = ["chunk1", "chunk2"]
	sources_ = ["source1", "source2"]
	ci = ChunkKeywordsIndex(chunks = chunks_, sources = sources_)
	@test chunks(ci) == chunks_
	@test sources(ci) == sources_
	@test chunkdata(ci) == nothing
	@test tags(ci) == nothing
	@test tags_vocab(ci) == nothing
	@test extras(ci) == nothing
	@test translate_positions_to_parent(ci, [1]) == [1]
	@test translate_positions_to_parent(ci, [2, 1]) == [2, 1]
	@test translate_positions_to_parent(ci, [4, 6]) == [4, 6]
	@test translate_positions_to_parent(ci, Int[]) == Int[]
	@test chunkdata(ci) == nothing
	@test chunkdata(ci, [1]) == nothing

	# Test equality of ChunkKeywordsIndex
	chunks_ = ["this is a test", "this is another test", "foo bar baz"]
	sources_ = ["source1", "source2", "source3"]
	dtm = document_term_matrix(chunks_)
	ci1 = ChunkKeywordsIndex(chunks = chunks_, sources = sources_, chunkdata = dtm)
	ci2 = ChunkKeywordsIndex(chunks = chunks_, sources = sources_, chunkdata = dtm)
	@test ci1 == ci2
	@test chunkdata(ci1) == dtm
	@test chunkdata(ci1, [1]) == view(dtm, [1], :)

	ci3 = ChunkKeywordsIndex(chunks = ["chunk2"], sources = ["source2"])
	@test ci1 != ci3

	# Test hcat with ChunkKeywordsIndex
	chunks1 = ["chunk1", "chunk2"]
	sources1 = ["source1", "source1"]
	ci1 = ChunkKeywordsIndex(
		chunks = chunks1, sources = sources1, chunkdata = document_term_matrix(chunks1))

	chunks2 = ["chunk3", "chunk4"]
	sources2 = ["source2", "source2"]
	ci2 = ChunkKeywordsIndex(
		chunks = chunks2, sources = sources2, chunkdata = document_term_matrix(chunks2))

	combined_ci = vcat(ci1, ci2)
	@test length(combined_ci.chunks) == 4
	@test length(combined_ci.sources) == 4
	@test combined_ci.chunks == ["chunk1", "chunk2", "chunk3", "chunk4"]
	@test combined_ci.sources == ["source1", "source1", "source2", "source2"]

	# HasEmbeddings
	@test HasEmbeddings(ci1) == false
	@test HasKeywords(ci1) == true
	@test_throws ArgumentError embeddings(ci1)
end


@testset "SubChunkIndex" begin
	ci1 = ChunkEmbeddingsIndex(chunks = ["chunk1", "chunk2", "chunk3"],
		embeddings = nothing,
		tags = nothing,
		tags_vocab = nothing,
		sources = ["source1", "source2", "source3"],
		id = Symbol("TestChunkIndex"))

	# Test creating a SubChunkIndex with CandidateChunks
	cc = CandidateChunks(ci1, 1:2)
	sub_index = view(ci1, cc)
	@test chunks(sub_index) == ["chunk1", "chunk2"]

	# Test creating a SubChunkIndex with different CandidateChunks
	cc = CandidateChunks(ci1, [2, 3])
	sub_index = view(ci1, cc)
	@test chunks(sub_index) == ["chunk2", "chunk3"]
	@test sources(sub_index) == ["source2", "source3"]
	@test translate_positions_to_parent(sub_index, [2, 1]) == [3, 2]

	# Test accessing chunks from SubChunkIndex
	cc = CandidateChunks(ci1, [2])
	sub_index = view(ci1, cc)
	@test sub_index[cc, :chunks] == ["chunk2"]
	@test sub_index[cc, :sources] == ["source2"]
	@test sub_index[cc, :embeddings] == nothing
	@test sub_index[cc, :chunkdata] == nothing
	@test parent(sub_index)[cc, :chunks] == ["chunk2"]
	@test chunkdata(sub_index) == nothing
	@test chunkdata(sub_index, [1]) == nothing

	# Wrong Index ID -> empty
	cc_wrongid = CandidateChunks(:bad_id, [2], [0.1f0])
	sub_index_wrongid = view(ci1, cc_wrongid)
	@test isempty(sub_index_wrongid)

	# Test creating a SubChunkIndex with out-of-bounds CandidateChunks
	cc = CandidateChunks(ci1, [4])
	@test_throws BoundsError view(ci1, cc)
	cc = CandidateChunks(ci1, 1:4)
	@test_throws BoundsError view(ci1, cc)

	chunks_test = ["chunk1", "chunk2", "chunk3"]
	emb_test = ones(2, 3) ./ (1:3)'
	tags_test = sparse([1, 2, 3], [1, 2, 3], [true, true, true], 3, 3)
	tags_vocab_test = ["vocab1", "vocab2", "vocab3"]
	sources_test = ["source1", "source2", "source3"]
	ci2 = ChunkEmbeddingsIndex(id = :TestChunkIndex2, chunks = chunks_test,
		embeddings = emb_test,
		tags = tags_test,
		tags_vocab = tags_vocab_test,
		sources = sources_test)

	# Create a SubChunkIndex for testing
	cc11 = CandidateChunks(ci2, [1, 2])
	sub_index11 = @view ci2[cc11]

	@test indexid(sub_index11) == indexid(ci2)
	@test positions(sub_index11) == [1, 2]
	@test parent(sub_index11) == ci2
	@test HasEmbeddings(sub_index11) == true
	@test HasKeywords(sub_index11) == false
	@test chunks(sub_index11) == ["chunk1", "chunk2"]
	@test sources(sub_index11) == ["source1", "source2"]
	@test chunkdata(sub_index11) ≈ [1.0 0.5; 1.0 0.5]
	@test chunkdata(sub_index11, [2]) ≈ [0.5, 0.5]
	@test embeddings(sub_index11) ≈ [1.0 0.5; 1.0 0.5]
	@test tags(sub_index11) == Bool[1 0 0; 0 1 0]
	@test tags_vocab(sub_index11) == tags_vocab_test
	@test extras(sub_index11) == nothing
	@test length(sub_index11) == 2
	@test unique(sub_index11) == sub_index11

	cc2 = CandidateChunks(ci2, [1, 2, 1, 2])
	sub_index2 = @view ci2[cc2]
	@test length(sub_index2) == 4
	@test chunks(sub_index2) == ["chunk1", "chunk2", "chunk1", "chunk2"]
	@test sources(sub_index2) == ["source1", "source2", "source1", "source2"]
	@test unique(sub_index2) == sub_index11
	@test positions(vcat(sub_index11, sub_index2)) == [1, 2, 1, 2, 1, 2]

	# Test vcat not implemented for different types
	ci3 = ChunkEmbeddingsIndex(chunks = ["chunk4", "chunk5"],
		embeddings = nothing,
		tags = nothing,
		tags_vocab = nothing,
		sources = ["source4", "source5"],
		id = Symbol("TestChunkIndex3"))
	cc3 = CandidateChunks(ci3, [1, 2])
	sub_index3 = view(ci3, cc3)
	@test_throws ArgumentError vcat(sub_index, sub_index3)

	# Test vcat for same parent
	cc = CandidateChunks(ci1, [1, 2])
	sub_index = view(ci1, cc)
	cc4 = CandidateChunks(ci1, [3])
	sub_index4 = view(ci1, cc4)
	vcat_index = vcat(sub_index, sub_index4)
	@test vcat_index == SubChunkIndex(ci1, [1, 2, 3])

	# Test edge cases
	# Empty positions
	cc_empty = CandidateChunks(ci1, Int[])
	sub_index_empty = view(ci1, cc_empty)
	@test length(sub_index_empty) == 0
	@test chunks(sub_index_empty) == String[]
	@test sources(sub_index_empty) == String[]
	@test isempty(sub_index_empty) == true

	# Out of bounds positions
	cc_oob = CandidateChunks(ci1, [10])
	@test_throws BoundsError view(ci1, cc_oob)

	# Duplicate positions
	cc_dup = CandidateChunks(ci1, [1, 1, 2])
	sub_index_dup = view(ci1, cc_dup)
	@test length(sub_index_dup) == 3
	@test chunks(sub_index_dup) == ["chunk1", "chunk1", "chunk2"]
	@test unique(sub_index_dup) == SubChunkIndex(ci1, [1, 2])

	# Test show method
	io = IOBuffer()
	show(io, sub_index)
	@test String(take!(io)) ==
		  "A view of ChunkEmbeddingsIndex (id: TestChunkIndex) with 2 chunks"

	## Nested SubChunkIndex
	# Test SubChunkIndex created from SubChunkIndex
	cc_sub = CandidateChunks(sub_index, [1])
	sub_sub_index = view(sub_index, cc_sub)
	@test length(sub_sub_index) == 1
	@test chunks(sub_sub_index) == ["chunk1"]
	@test sources(sub_sub_index) == ["source1"]
	@test parent(sub_sub_index) == ci1
	@test parent(@view sub_sub_index[cc_sub]) == ci1

	cc_oob = CandidateChunks(ci1, [10])
	@test_throws BoundsError view(ci1, cc_oob)

	## Nest deeper
	sub_sub_index = SubChunkIndex(sub_sub_index, cc_sub)
	@test parent(sub_sub_index) == ci1
	@test length(sub_sub_index) == 1
	@test chunks(sub_sub_index) == ["chunk1"]
	@test sources(sub_sub_index) == ["source1"]

	sub_oob = SubChunkIndex(sub_sub_index, [10])
	@test_throws BoundsError SubChunkIndex(sub_oob, cc_oob)

	# return empty if it's wrong index id
	cc_wrongid = CandidateChunks(:bad_id, [2], [0.1f0])
	sub_index_wrongid = SubChunkIndex(sub_sub_index, cc_wrongid)
	@test isempty(sub_index_wrongid)

	# views produce intersection, so if they don't match it becomes empty view
	cc_sub_notmatch = CandidateChunks(sub_sub_index, [2])
	@test view(sub_sub_index, cc_sub_notmatch) |> isempty

	# Test edge cases for SubChunkIndex created from SubChunkIndex
	# Empty positions
	cc_empty_sub = CandidateChunks(sub_index, Int[])
	sub_index_empty_sub = view(sub_index, cc_empty_sub)
	@test length(sub_index_empty_sub) == 0
	@test chunks(sub_index_empty_sub) == String[]
	@test sources(sub_index_empty_sub) == String[]
	@test isempty(sub_index_empty_sub) == true

	# Out of bounds positions
	cc_oob_sub = CandidateChunks(ci1, [10])
	@test_throws BoundsError view(ci1, cc_oob_sub)

	# Duplicate positions
	cc_dup_sub = CandidateChunks(ci1, [1, 1, 2])
	sub_index_dup_sub = view(ci1, cc_dup_sub)
	@test length(sub_index_dup_sub) == 3
	@test chunks(sub_index_dup_sub) == ["chunk1", "chunk1", "chunk2"]
	@test unique(sub_index_dup_sub) == SubChunkIndex(ci1, [1, 2])

	# Test show method for SubChunkIndex created from SubChunkIndex
	io_sub = IOBuffer()
	show(io_sub, sub_sub_index)
	@test String(take!(io_sub)) ==
		  "A view of ChunkEmbeddingsIndex (id: TestChunkIndex) with 1 chunks"

	## MultiCandidateChunks
	# Test SubChunkIndex with MultiCandidateChunks
	mcc = MultiCandidateChunks(ci2, [2, 3])
	sub_index_mcc = view(ci2, mcc)
	@test length(sub_index_mcc) == 2
	@test chunks(sub_index_mcc) == ["chunk2", "chunk3"]
	@test sources(sub_index_mcc) == ["source2", "source3"]
	@test chunkdata(sub_index_mcc) ≈ [0.5 0.3333333333333333; 0.5 0.3333333333333333]
	@test embeddings(sub_index_mcc) ≈ [0.5 0.3333333333333333; 0.5 0.3333333333333333]
	@test tags(sub_index_mcc) == Bool[0 1 0; 0 0 1]
	@test tags_vocab(sub_index_mcc) == tags_vocab_test
	@test extras(sub_index_mcc) == nothing

	## Nested sub-chunk index
	sub_sub_index = @view sub_index_mcc[mcc]
	@test length(sub_sub_index) == 2
	@test chunks(sub_sub_index) == ["chunk2", "chunk3"]
	@test sources(sub_sub_index) == ["source2", "source3"]
	mcc_oob = MultiCandidateChunks(ci2, [10])
	@test_throws BoundsError view(ci2, mcc_oob)

	## Nest deeper
	sub_sub_index = SubChunkIndex(sub_sub_index, mcc)
	@test parent(sub_sub_index) == ci2
	@test length(sub_sub_index) == 2
	@test chunks(sub_sub_index) == ["chunk2", "chunk3"]
	@test sources(sub_sub_index) == ["source2", "source3"]

	sub_oob = SubChunkIndex(ci2, [10])
	@test_throws BoundsError SubChunkIndex(sub_oob, mcc_oob)

	# views produce intersection, so if they don't match it becomes empty view
	mcc_notmatch = MultiCandidateChunks(sub_sub_index, [1])
	@test view(sub_sub_index, mcc_notmatch) |> isempty

	## With keyword index
	chunks_ = ["chunk1", "chunk2"]
	sources_ = ["source1", "source2"]
	cki = ChunkKeywordsIndex(chunks = chunks_, sources = sources_)
	cck = CandidateChunks(cki, [2])
	sub_cki = @view cki[cck]
	@test length(cki) == 2
	@test length(cck) == 1
	@test length(sub_cki) == 1
	@test chunks(sub_cki) == ["chunk2"]
	@test sources(sub_cki) == ["source2"]
	@test parent(sub_cki) == cki
	@test chunkdata(sub_cki) == nothing
	@test HasEmbeddings(sub_cki) == false
	@test HasKeywords(sub_cki) == true
	@test_throws ArgumentError embeddings(sub_cki)
	@test tags(sub_cki) == nothing
	@test tags_vocab(sub_cki) == nothing
	@test extras(sub_cki) == nothing

	## MultiIndex not implemented yet
	mi = MultiIndex(indexes = [ci1, cki])
	mccx = MultiCandidateChunks(index_ids = [:TestChunkIndex1, :TestChunkIndex2],
		positions = [1, 2], scores = [0.1, 0.2])
	@test_throws ArgumentError @view mi[mccx]
end


@testset "MultiIndex" begin
	# Test constructors/accessors
	# MultiIndex behaves as a container for ChunkEmbeddingsIndexes
	cin1 = ChunkEmbeddingsIndex(chunks = ["chunk1"], sources = ["source1"])
	cin2 = ChunkEmbeddingsIndex(chunks = ["chunk2"], sources = ["source2"])
	multi_index = MultiIndex(indexes = [cin1, cin2])
	@test length(multi_index.indexes) == 2
	@test cin1 in multi_index.indexes
	@test cin2 in multi_index.indexes

	# Test base var"==" with MultiIndex
	# Case where MultiIndexes are equal
	cin1 = ChunkEmbeddingsIndex(chunks = ["chunk1"], sources = ["source1"])
	cin2 = ChunkEmbeddingsIndex(chunks = ["chunk2"], sources = ["source2"])
	mi1 = MultiIndex(indexes = [cin1, cin2])
	mi2 = MultiIndex(indexes = [cin1, cin2])
	@test mi1 == mi2

	# Test equality with different ChunkEmbeddingsIndexes inside
	cin1 = ChunkEmbeddingsIndex(chunks = ["chunk1"], sources = ["source1"])
	cin2 = ChunkEmbeddingsIndex(chunks = ["chunk2"], sources = ["source2"])
	mi1 = MultiIndex([cin1])
	mi2 = MultiIndex(cin2)
	@test mi1 != mi2

	# HasEmbeddings
	@test HasEmbeddings(mi1) == true
	@test HasKeywords(mi1) == false

	ci = ChunkKeywordsIndex(chunks = ["chunk1"], sources = ["source1"])
	mi2 = MultiIndex(indexes = [ci])
	@test HasEmbeddings(mi2) == false

	cin1 = ChunkEmbeddingsIndex(chunks = ["chunk1"], sources = ["source1"], id = :cin1)
	cin2 = ChunkKeywordsIndex(chunks = ["chunk1"], sources = ["source1"], id = :cin2)
	mi3 = MultiIndex(indexes = [cin1, cin2], id = :mi3)
	@test HasEmbeddings(mi3) == true
	@test HasKeywords(mi3) == true

	## not implemented
	@test_throws ArgumentError vcat(mi1, mi2)

	# Get index
	@test mi3[:cin1] == cin1
	@test mi3[:cin2] == cin2
	@test mi3[:xyz] == nothing
	@test mi3[:mi3] == mi3
end