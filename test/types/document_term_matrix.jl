# Local Dependencies
# ------------------
using RAGTools: DocumentTermMatrix, SubDocumentTermMatrix, document_term_matrix
using RAGTools: parent, positions, tf, vocab, vocab_lookup, idf, doc_rel_length

# Tests 
# ------

@testset "DocumentTermMatrix" begin
    # Simple case
    documents = [["this", "is", "a", "test"],
        ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
    dtm = document_term_matrix(documents)
    @test size(dtm.tf) == (3, 8)
    @test Set(dtm.vocab) == Set(["a", "another", "bar", "baz", "foo", "is", "test", "this"])
    avgdl = 3.666666666666667
    @test all(dtm.doc_rel_length .≈ [4 / avgdl, 4 / avgdl, 3 / avgdl])
    @test length(dtm.idf) == 8

    # Edge case: single document
    documents = [["this", "is", "a", "test"]]
    dtm = document_term_matrix(documents)
    @test size(dtm.tf) == (1, 4)
    @test Set(dtm.vocab) == Set(["a", "is", "test", "this"])
    @test dtm.doc_rel_length == ones(1)
    @test length(dtm.idf) == 4

    # Edge case: duplicate tokens
    documents = [["this", "is", "this", "test"],
        ["this", "is", "another", "test"], ["this", "bar", "baz"]]
    dtm = document_term_matrix(documents)
    @test size(dtm.tf) == (3, 6)
    @test Set(dtm.vocab) == Set(["another", "bar", "baz", "is", "test", "this"])
    avgdl = 3.666666666666667
    @test all(dtm.doc_rel_length .≈ [4 / avgdl, 4 / avgdl, 3 / avgdl])
    @test length(dtm.idf) == 6

    # Edge case: no tokens
    documents = [String[], String[], String[]]
    dtm = document_term_matrix(documents)
    @test size(dtm.tf) == (3, 0)
    @test isempty(dtm.vocab)
    @test isempty(dtm.vocab_lookup)
    @test isempty(dtm.idf)
    @test dtm.doc_rel_length == zeros(3)

    ## Methods - hcat
    documents = [["this", "is", "a", "test"],
        ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
    dtm1 = document_term_matrix(documents)
    documents = [["this", "is", "a", "test"],
        ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
    dtm2 = document_term_matrix(documents)
    dtm = hcat(dtm1, dtm2)
    @test size(dtm.tf) == (6, 8)
    @test length(dtm.vocab) == 8
    @test length(dtm.idf) == 8
    @test isapprox(dtm.doc_rel_length,
        [4 / 3.666666666666667, 4 / 3.666666666666667, 3 / 3.666666666666667,
            4 / 3.666666666666667, 4 / 3.666666666666667, 3 / 3.666666666666667])

    ## SubDocumentTermMatrix
    # Create a parent DocumentTermMatrix
    documents = [["this", "is", "a", "test"], ["another", "test", "document"]]
    dtm = document_term_matrix(documents)

    # Create a SubDocumentTermMatrix
    sub_dtm = view(dtm, [1], :)

    # Test parent method
    @test parent(sub_dtm) == dtm

    # Test positions method
    @test positions(sub_dtm) == [1]

    # Test tf method
    @test tf(sub_dtm) == dtm.tf[1:1, :]

    # Test vocab method
    @test vocab(sub_dtm) == vocab(dtm)

    # Test vocab_lookup method
    @test vocab_lookup(sub_dtm) == vocab_lookup(dtm)

    # Test idf method
    @test idf(sub_dtm) == idf(dtm)

    # Test doc_rel_length method
    @test doc_rel_length(sub_dtm) == doc_rel_length(dtm)[1:1]

    # Test view method for SubDocumentTermMatrix
    sub_dtm_view = view(sub_dtm, [1], :)
    @test parent(sub_dtm_view) == dtm
    @test positions(sub_dtm_view) == [1]
    @test tf(sub_dtm_view) == dtm.tf[1:1, :]

    # Nested view // no intersection
    sub_sub_dtm_view = view(sub_dtm_view, [2], :)
    @test parent(sub_sub_dtm_view) == dtm
    @test isempty(positions(sub_sub_dtm_view))
    @test tf(sub_sub_dtm_view) |> isempty

    # Test view method with out of bounds positions
    @test_throws BoundsError view(sub_dtm, [10], :)

    # Test view method with intersecting positions
    sub_dtm_intersect = view(dtm, [1, 2], :)
    sub_dtm_view_intersect = view(sub_dtm_intersect, [2], :)
    @test parent(sub_dtm_view_intersect) == dtm
    @test positions(sub_dtm_view_intersect) == [2]
    @test tf(sub_dtm_view_intersect) == dtm.tf[2:2, :]

    ### Test hcat for DocumentTermMatrix
    # Create two DocumentTermMatrix instances
    documents1 = [["this", "is", "a", "test"], ["another", "test", "document"]]
    dtm1 = document_term_matrix(documents1)

    documents2 = [["new", "document"], ["with", "different", "words"]]
    dtm2 = document_term_matrix(documents2)

    # Perform hcat
    combined_dtm = hcat(dtm1, dtm2)

    # Test the resulting DocumentTermMatrix
    @test size(combined_dtm.tf, 1) == size(dtm1.tf, 1) + size(dtm2.tf, 1)
    @test length(combined_dtm.vocab) == length(unique(vcat(dtm1.vocab, dtm2.vocab)))
    @test all(word in combined_dtm.vocab for word in dtm1.vocab)
    @test all(word in combined_dtm.vocab for word in dtm2.vocab)

    # Check if the tf matrix is correctly combined
    @test size(combined_dtm.tf, 2) == length(combined_dtm.vocab)
    @test sum(combined_dtm.tf) ≈ sum(dtm1.tf) + sum(dtm2.tf)

    # Test vocab_lookup
    @test all(haskey(combined_dtm.vocab_lookup, word) for word in combined_dtm.vocab)

    # Test idf
    @test length(combined_dtm.idf) == length(combined_dtm.vocab)

    # Test doc_rel_length
    @test length(combined_dtm.doc_rel_length) == size(combined_dtm.tf, 1)

    # Test with empty DocumentTermMatrix
    empty_dtm = document_term_matrix(Vector{Vector{String}}())
    combined_with_empty = hcat(dtm1, empty_dtm)
    @test combined_with_empty == dtm1

    # Test associativity
    dtm3 = document_term_matrix([["third", "set", "of", "documents"]])
    @test hcat(hcat(dtm1, dtm2), dtm3) == hcat(dtm1, hcat(dtm2, dtm3))

    # Test with dense matrix
    ddtm1 = DocumentTermMatrix(
        Matrix(tf(dtm1)), vocab(dtm1), vocab_lookup(dtm1), idf(dtm1), doc_rel_length(dtm1))
    ddtm2 = DocumentTermMatrix(
        Matrix(tf(dtm2)), vocab(dtm2), vocab_lookup(dtm2), idf(dtm2), doc_rel_length(dtm2))
    combined_ddtm = hcat(ddtm1, ddtm2)
    @test size(combined_ddtm.tf, 1) == size(ddtm1.tf, 1) + size(ddtm2.tf, 1)
    @test length(combined_ddtm.vocab) == length(unique(vcat(ddtm1.vocab, ddtm2.vocab)))
    @test all(word in combined_ddtm.vocab for word in ddtm1.vocab)
    @test all(word in combined_ddtm.vocab for word in ddtm2.vocab)
    @test size(combined_ddtm.tf, 2) == length(combined_ddtm.vocab)
    @test sum(combined_ddtm.tf) ≈ sum(ddtm1.tf) + sum(ddtm2.tf)
end
