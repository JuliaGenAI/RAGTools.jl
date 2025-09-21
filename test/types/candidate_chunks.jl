# Local Dependencies
# ------------------
using RAGTools: AbstractCandidateChunks, CandidateChunks, MultiCandidateChunks
using RAGTools: indexid, indexids, positions, scores

# Tests
# ------

@testset "CandidateChunks" begin
    chunk_sym = Symbol("TestChunkEmbeddingsIndex")
    cc1 = CandidateChunks(index_id = chunk_sym,
        positions = [1, 3],
        scores = [0.1, 0.2])
    @test Base.length(cc1) == 2
    out = Base.first(cc1, 1)
    @test out.positions == [3]
    @test out.scores == [0.2]
    @test indexid(cc1) == chunk_sym
    @test indexids(cc1) == [chunk_sym, chunk_sym]

    # Test intersection &
    cc2 = CandidateChunks(index_id = chunk_sym,
        positions = [2, 4],
        scores = [0.3, 0.4])
    @test isempty((cc1 & cc2).positions)
    cc3 = CandidateChunks(index_id = chunk_sym,
        positions = [1, 4],
        scores = [0.3, 0.5])
    joint = (cc1 & cc3)
    @test joint.positions == [1]
    @test joint.scores == [0.3]
    joint2 = (cc2 & cc3)
    @test joint2.positions == [4]
    @test joint2.scores == [0.5]

    # long positions intersection
    cc5 = CandidateChunks(index_id = chunk_sym,
        positions = [5, 6, 7, 8, 9, 10, 4],
        scores = 0.1 * ones(7))
    joint5 = (cc2 & cc5)
    @test joint5.positions == [4]
    @test joint5.scores == [0.4]

    # wrong index
    cc4 = CandidateChunks(index_id = :xyz,
        positions = [2, 4],
        scores = [0.3, 0.4])
    joint4 = (cc2 & cc4)
    @test isempty(joint4.positions)
    @test isempty(joint4.scores)
    @test isempty(joint4) == true

    # Test unknown type
    struct RandomCandidateChunks123 <: AbstractCandidateChunks end
    @test_throws ArgumentError (cc1 & RandomCandidateChunks123())

    # Test vcat
    vcat1 = vcat(cc1, cc2)
    @test Base.length(vcat1) == 4
    vcat2 = vcat(cc1, cc3)
    @test vcat2.positions == [4, 1, 3]
    @test vcat2.scores == [0.5, 0.3, 0.2]
    # wrong index
    @test_throws ArgumentError vcat(cc1, cc4)
    # uknown type
    @test_throws ArgumentError vcat(cc1, RandomCandidateChunks123())

    # Test copy
    cc1_copy = copy(cc1)
    @test cc1 == cc1_copy
    @test cc1.positions !== cc1_copy.positions # not the same array

    # Serialization
    tmp, _ = mktemp()
    JSON3.write(tmp, cc1)
    cc1x = JSON3.read(tmp, CandidateChunks)
    @test cc1x.index_id == cc1.index_id
    @test cc1x.positions == cc1.positions
    @test cc1x.scores ≈ cc1.scores
end

@testset "MultiCandidateChunks" begin
    chunk_sym1 = Symbol("TestChunkEmbeddingsIndex1")
    chunk_sym2 = Symbol("TestChunkEmbeddingsIndex2")
    mcc1 = MultiCandidateChunks(index_ids = [chunk_sym1, chunk_sym2],
        positions = [1, 3],
        scores = [0.1, 0.2])
    @test Base.length(mcc1) == 2
    out = Base.first(mcc1, 1)
    @test out.positions == [3]
    @test out.scores == [0.2]
    @test indexids(mcc1) == [chunk_sym1, chunk_sym2]

    # Test vcat
    mcc2 = MultiCandidateChunks(index_ids = [chunk_sym1, chunk_sym2],
        positions = [2, 4],
        scores = [0.3, 0.4])
    vcat1 = vcat(mcc1, mcc2)
    @test Base.length(vcat1) == 4
    vcat2 = vcat(mcc1,
        MultiCandidateChunks(index_ids = [chunk_sym1, chunk_sym2],
            positions = [1, 4],
            scores = [0.3, 0.5]))
    @test vcat2.positions == [4, 1, 3]
    @test vcat2.scores == [0.5, 0.3, 0.2]

    # Test copy
    mcc1_copy = copy(mcc1)
    @test mcc1 == mcc1_copy
    @test mcc1.positions !== mcc1_copy.positions # not the same array

    chunk_sym1 = Symbol("TestChunkEmbeddingsIndex1")
    chunk_sym2 = Symbol("TestChunkEmbeddingsIndex2")
    # Test intersection with overlapping positions
    mcc3 = MultiCandidateChunks(index_ids = [chunk_sym1, chunk_sym2],
        positions = [1, 4],
        scores = [0.3, 0.5])
    joint = (mcc1 & mcc3)
    @test joint.positions == [1]
    @test joint.scores == [0.3]
    joint2 = (mcc2 & mcc3)
    @test joint2.positions == [4]
    @test joint2.scores == [0.5]

    # Test intersection with no overlapping positions
    mcc4 = MultiCandidateChunks(index_ids = [chunk_sym1, chunk_sym2],
        positions = [6, 7],
        scores = [0.6, 0.7])
    joint3 = (mcc1 & mcc4)
    @test isempty(joint3.positions)
    @test isempty(joint3.scores)
    @test isempty(joint3) == true

    # Test intersection with long positions
    mcc5 = MultiCandidateChunks(index_ids = fill(chunk_sym2, 7),
        positions = [5, 6, 7, 8, 9, 10, 4],
        scores = 0.1 * ones(7))
    joint4 = (mcc2 & mcc5)
    @test joint4.positions == [4]
    @test joint4.scores == [0.4]

    # Test intersection with wrong index
    mcc6 = MultiCandidateChunks(index_ids = [:xyz, :abc],
        positions = [2, 4],
        scores = [0.3, 0.4])
    joint5 = (mcc2 & mcc6)
    @test isempty(joint5.positions)
    @test isempty(joint5.scores)
    @test isempty(joint5) == true

    # Test intersection with unknown type
    struct RandomMultiCandidateChunks123 <: AbstractCandidateChunks end
    @test_throws ArgumentError (mcc1 & RandomMultiCandidateChunks123())
end
