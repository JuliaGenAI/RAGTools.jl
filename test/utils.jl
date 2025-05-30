using RAGTools: _check_aiextract_capability,
	vcat_labeled_matrices, hcat_labeled_matrices
using RAGTools: tokenize, trigrams, trigrams_hashed
using RAGTools: token_with_boundaries, text_to_trigrams,
	text_to_trigrams_hashed
using RAGTools: split_into_code_and_sentences
using RAGTools: getpropertynested, setpropertynested,
	merge_kwargs_nested
using RAGTools: pack_bits, unpack_bits, preprocess_tokens,
	reciprocal_rank_fusion, score_to_unit_scale,
	hcat_truncate

@testset "_check_aiextract_capability" begin
	@test _check_aiextract_capability("gpt-3.5-turbo") == nothing
	@test_throws AssertionError _check_aiextract_capability("llama2")
end

@testset "vcat_labeled_matrices" begin
	# Test with dense matrices and overlapping vocabulary
	mat1 = [1 2; 3 4]
	vocab1 = ["word1", "word2"]
	mat2 = [5 6; 7 8]
	vocab2 = ["word2", "word3"]

	merged_mat, combined_vocab = vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test size(merged_mat) == (4, 3)
	@test combined_vocab == ["word1", "word2", "word3"]
	@test merged_mat == [1 2 0; 3 4 0; 0 5 6; 0 7 8]

	# Test with sparse matrices and disjoint vocabulary
	mat1 = sparse([1 0; 0 2])
	vocab1 = ["word1", "word2"]
	mat2 = sparse([3 0; 0 4])
	vocab2 = ["word3", "word4"]

	merged_mat, combined_vocab = vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test size(merged_mat) == (4, 4)
	@test combined_vocab == ["word1", "word2", "word3", "word4"]
	@test merged_mat == sparse([1 0 0 0; 0 2 0 0; 0 0 3 0; 0 0 0 4])

	# Test with different data types
	mat1 = [1.0 2.0; 3.0 4.0]
	vocab1 = ["word1", "word2"]
	mat2 = [5 6; 7 8]
	vocab2 = ["word2", "word3"]

	merged_mat, combined_vocab = vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test eltype(merged_mat) == Float64
	@test size(merged_mat) == (4, 3)
	@test combined_vocab == ["word1", "word2", "word3"]
	@test merged_mat ≈ [1.0 2.0 0.0; 3.0 4.0 0.0; 0.0 5.0 6.0; 0.0 7.0 8.0]

	### Test cases with sparse matrices
	# Test case 1: Basic functionality with non-overlapping vocabularies
	mat1 = sparse([1 0; 0 2])
	vocab1 = ["word1", "word2"]
	mat2 = sparse([3 0; 0 4])
	vocab2 = ["word3", "word4"]

	result_mat, result_vocab = RT.vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)
	@test result_mat isa SparseMatrixCSC
	@test size(result_mat) == (4, 4)
	@test result_vocab == ["word1", "word2", "word3", "word4"]
	@test Array(result_mat) == [1 0 0 0; 0 2 0 0; 0 0 3 0; 0 0 0 4]

	# Test case 2: Overlapping vocabularies
	mat1 = sparse([1 2; 3 0])
	vocab1 = ["word1", "word2"]
	mat2 = sparse([0 4; 5 6])
	vocab2 = ["word2", "word3"]

	result_mat, result_vocab = RT.vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)
	@test result_mat isa SparseMatrixCSC
	@test size(result_mat) == (4, 3)
	@test result_vocab == ["word1", "word2", "word3"]
	@test Array(result_mat) == [1 2 0; 3 0 0; 0 0 4; 0 5 6]

	# Test case 3: Different data types
	mat1 = sparse([1.0 0.0; 0.0 2.0])
	vocab1 = ["word1", "word2"]
	mat2 = sparse(Float32[3 0; 0 4])
	vocab2 = ["word3", "word4"]

	result_mat, result_vocab = RT.vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)
	@test result_mat isa SparseMatrixCSC{Float64}
	@test size(result_mat) == (4, 4)
	@test result_vocab == ["word1", "word2", "word3", "word4"]
	@test Array(result_mat) ==
		  [1.0 0.0 0.0 0.0; 0.0 2.0 0.0 0.0; 0.0 0.0 3.0 0.0; 0.0 0.0 0.0 4.0]

	# Test case 4: Empty matrices
	mat1 = sparse(Int[], Int[], Int[], 0, 0)
	vocab1 = String[]
	mat2 = sparse(Int[], Int[], Int[], 0, 0)
	vocab2 = String[]

	result_mat, result_vocab = RT.vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test result_mat isa SparseMatrixCSC
	@test size(result_mat) == (0, 0)
	@test result_vocab == String[]

	# Test case 5: Large sparse matrices
	n = 1000
	m = 500
	mat1 = sprand(Float32, n, m, 0.01)
	vocab1 = ["word$i" for i in 1:m]
	mat2 = sprand(Float32, n, m, 0.01)
	vocab2 = ["word$(i+m÷2)" for i in 1:m]

	result_mat, result_vocab = RT.vcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test result_mat isa SparseMatrixCSC
	@test size(result_mat) == (2n, length(unique([vocab1; vocab2])))
	@test length(result_vocab) == length(unique([vocab1; vocab2]))
	@test nnz(result_mat) ≈ nnz(mat1) + nnz(mat2) atol = 10  # Allow for some numerical imprecision
end

@testset "hcat_labeled_matrices" begin
	# Test with dense matrices and overlapping vocabulary
	mat1 = [1 2; 3 4]
	vocab1 = ["word1", "word2"]
	mat2 = [5 6; 7 8]
	vocab2 = ["word2", "word3"]

	merged_mat, combined_vocab = hcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test size(merged_mat) == (3, 4)
	@test combined_vocab == ["word1", "word2", "word3"]
	@test merged_mat == [1 2 0 0; 3 4 5 6; 0 0 7 8]

	# Test with sparse matrices and disjoint vocabulary
	mat1 = sparse([1 0; 0 2])
	vocab1 = ["word1", "word2"]
	mat2 = sparse([3 0; 0 4])
	vocab2 = ["word3", "word4"]

	merged_mat, combined_vocab = hcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test size(merged_mat) == (4, 4)
	@test combined_vocab == ["word1", "word2", "word3", "word4"]
	@test merged_mat == sparse([1 0 0 0; 0 2 0 0; 0 0 3 0; 0 0 0 4])

	# Test with different data types
	mat1 = [1.0 2.0; 3.0 4.0]
	vocab1 = ["word1", "word2"]
	mat2 = [5 6; 7 8]
	vocab2 = ["word2", "word3"]

	merged_mat, combined_vocab = hcat_labeled_matrices(mat1, vocab1, mat2, vocab2)

	@test eltype(merged_mat) == Float64
	@test size(merged_mat) == (3, 4)
	@test combined_vocab == ["word1", "word2", "word3"]
	@test merged_mat ≈ [1.0 2.0 0.0 0.0; 3.0 4.0 5.0 6.0; 0.0 0.0 7.0 8.0]
end

@testset "hcat_truncate" begin
	# Test basic functionality with no truncation
	m1 = Float32[1 2; 3 4; 5 6]
	m2 = Float32[7 8; 9 10; 11 12]
	result = hcat_truncate([m1, m2])
	@test size(result) == (3, 4)
	@test result == Float32[1 2 7 8; 3 4 9 10; 5 6 11 12]

	# Test with truncation
	result_truncated = hcat_truncate([m1, m2], 2)
	@test size(result_truncated) == (2, 4)

	# Test normalization after truncation
	expected_col1 = Float32[1, 3] / sqrt(1^2 + 3^2)
	@test result_truncated[:, 1] ≈ expected_col1

	# Test with single matrix input
	single_result = hcat_truncate([m1])
	@test single_result == m1

	# Test with empty input
	@test_throws Exception hcat_truncate([])

	# Test with matrices of different row counts
	m3 = Float32[1 2; 3 4]
	@test_throws AssertionError hcat_truncate([m1, m3])

	# Test with truncation dimension larger than input
	@test_throws AssertionError hcat_truncate([m1, m2], 4)

	# Test with truncate_dimension set to 0
	zero_truncate = hcat_truncate([m1, m2], 0)
	@test zero_truncate == Float32[1 2 7 8; 3 4 9 10; 5 6 11 12]

	# Test with large matrices to ensure performance
	large_m1 = rand(Float32, 1000, 1000)
	large_m2 = rand(Float32, 1000, 1000)
	@test size(hcat_truncate([large_m1, large_m2], 500)) == (500, 2000)

	# Test with different types (should convert to Float32)
	m4 = [1.0 2.0; 3.0 4.0; 5.0 6.0]
	result_type_conversion = hcat_truncate([m4])
	@test eltype(result_type_conversion) == Float32

	# Test with truncate=nothing (should behave the same as no truncation)
	result_nothing = hcat_truncate([m1, m2], nothing)
	@test result_nothing == Float32[1 2 7 8; 3 4 9 10; 5 6 11 12]

	# Test with truncate=-1 (should behave the same as no truncation)
	result_negative = hcat_truncate([m1, m2], -1)
	@test result_negative == Float32[1 2 7 8; 3 4 9 10; 5 6 11 12]

	## Test for Vectors
	# Test basic functionality
	v1 = [1.0, 2.0, 3.0]
	v2 = [4.0, 5.0, 6.0]
	result = hcat_truncate([v1, v2])
	@test size(result) == (3, 2)
	@test result == [1.0 4.0; 2.0 5.0; 3.0 6.0]

	# Test with truncation
	result_truncated = hcat_truncate([v1, v2], 2)
	@test size(result_truncated) == (2, 2)
	@test result_truncated ≈ mapreduce(normalize, hcat, eachcol([1.0 4.0; 2.0 5.0]))

	# Test with single vector input
	single_result = hcat_truncate([v1])
	@test single_result == reshape(v1, :, 1)

	# Test with empty input
	@test_throws Exception hcat_truncate(Vector{Float64}[])

	# Test with vectors of different lengths
	v3 = [1.0, 2.0]
	@test_throws AssertionError hcat_truncate([v1, v3])

	# Test with truncation dimension larger than input
	@test_throws AssertionError hcat_truncate([v1, v2], 4)

	# Test with truncate_dimension set to 0
	zero_truncate = hcat_truncate([v1, v2], 0)
	@test zero_truncate == [1.0 4.0; 2.0 5.0; 3.0 6.0]

	# Test with large vectors to ensure performance
	large_v1 = rand(1000)
	large_v2 = rand(1000)
	@test size(hcat_truncate([large_v1, large_v2], 500)) == (500, 2)

	# Test with different types (should convert to Float32)
	v4 = [1, 2, 3]
	result_type_conversion = hcat_truncate([v4])
	@test eltype(result_type_conversion) == Float32

	# Test with truncate=nothing (should behave the same as no truncation)
	result_nothing = hcat_truncate([v1, v2], nothing)
	@test result_nothing == [1.0 4.0; 2.0 5.0; 3.0 6.0]

	# Test with truncate=-1 (should behave the same as no truncation)
	result_negative = hcat_truncate([v1, v2], -1)
	@test result_negative == [1.0 4.0; 2.0 5.0; 3.0 6.0]
end

### Text-manipulation utilities

@testset "tokenize" begin
	# Test basic tokenization with common delimiters
	@test tokenize("Hello, world!") == ["Hello", ",", " ", "world", "!"]

	# Test tokenization with various whitespace characters
	@test tokenize("New\nLine\tTab") == ["New", "\n", "Line", "\t", "Tab"]

	# Test tokenization with a mix of punctuation and words
	@test tokenize("Yes! This works.") == ["Yes", "!", " ", "This", " ", "works", "."]

	# Test tokenization of a string with no delimiters, i.e., a single word
	@test tokenize("SingleWord") == ["SingleWord"]

	# Test tokenization of an empty string
	@test tokenize("") == []

	# multi-space
	@test tokenize("   ") == ["   "]

	# Special characters for Julia code
	@test tokenize("α β γ δ") == ["α", " ", "β", " ", "γ", " ", "δ"]
	@test tokenize("a = (; a=1)") == ["a", " ", "=", " ", "(;", " ", "a", "=", "1", ")"]
	@test tokenize("replace(s, \"abc\"=>\"ABC\")") ==
		  ["replace", "(", "s", ",", " ", "\"", "abc", "\"", "=>", "\"", "ABC", "\"", ")"]
end

@testset "trigrams" begin
	# Test generating trigrams from a string of sufficient length
	@test trigrams("hello") == ["hel", "ell", "llo"]

	# Test generating trigrams from a string with exactly 3 characters
	@test trigrams("cat") == ["cat"]

	# Test with a string of length less than 3, expecting an empty array
	@test trigrams("no") == []

	# Test with an empty string, also expecting an empty array
	@test trigrams("") == []

	# Test a case with special characters and spaces
	@test trigrams("a b c") == ["a b", " b ", "b c"]

	# With boundaries
	@test trigrams(" (cat=") == [" (c", "(ca", "cat", "at="]

	# Add the token itself
	@test trigrams("hello"; add_word = "hello") == ["hel", "ell", "llo", "hello"]

	# non-standard chars
	s = "α β γ δ"
	@test trigrams(s) == ["α β", " β ", "β γ", " γ ", "γ δ"]
end

@testset "trigrams_hashed" begin
	# Test hashing trigrams from a string of sufficient length
	# Since hashing produces unique UInt64 values, we test for the set's length instead of specific values
	@test trigrams_hashed("hello") == hash.(["hel", "ell", "llo"]) |> Set

	# Test hashing a string with exactly 3 characters
	@test trigrams_hashed("cat") == Set(hash("cat"))

	# Test with a string of length less than 3, expecting a set with 1 hash value
	@test trigrams_hashed("no") == Set()

	# Test with an empty string, expecting a set with 1 hash value because the empty string itself is hashed
	@test (trigrams_hashed("")) == Set()

	# Test to ensure no duplicate hash values in case of repeating trigrams
	# "ababab" will generate "aba", "bab", "aba", "bab" - only two unique trigrams when hashed
	@test trigrams_hashed("ababab") == Set([hash("aba"), hash("bab")])

	# Test a unique case with special characters to ensure hashing works across different character sets
	@test trigrams_hashed("a!@") == Set(hash("a!@"))

	# Add the token itself
	@test trigrams_hashed("hello"; add_word = "hello") ==
		  hash.(["hel", "ell", "llo", "hello"]) |> Set

	# special chars
	s = "α β γ δ"
	@test trigrams_hashed(s) == Set(hash.(["α β", " β ", "β γ", " γ ", "γ δ"]))
end

@testset "token_with_boundaries" begin
	# Test with no surrounding tokens
	@test token_with_boundaries(nothing, "current", nothing) == "current"

	# Test with both surrounding tokens being single characters (should concatenate all)
	@test token_with_boundaries("a", "current", "b") == "acurrentb"

	# Test with only previous token being a single character (should prepend it)
	@test token_with_boundaries("a", "current", nothing) == "acurrent"

	# Test with only next token being a single character (should append it)
	@test token_with_boundaries(nothing, "current", "b") == "currentb"

	# Test with both surrounding tokens but only next token being a single character (should append next token)
	@test token_with_boundaries("previous", "current", "b") == "currentb"

	# Test with both surrounding tokens but only previous token being a single character (should prepend previous token)
	@test token_with_boundaries("a", "current", "next") == "acurrent"

	# Test with neither surrounding tokens being single characters (should return the current token unchanged)
	@test token_with_boundaries("previous", "current", "next") == "current"

	# Test with single character current token and no surrounding tokens (should return the current token unchanged)
	@test token_with_boundaries(nothing, "c", nothing) == "c"
end

@testset "text_to_trigrams" begin
	# Test converting basic text into trigrams
	exp_output = [
		"Thi", "his", "is ", "This", " is", "is ", "is", " te", "tes", "est", "st.", "test"]
	@test text_to_trigrams("This is a test."; add_word = true) == exp_output

	# Test converting without adding the word itself
	exp_output = ["Thi", "his", "is ", " is", "is ", " te", "tes", "est", "st."]
	@test text_to_trigrams("This is a test."; add_word = false) == exp_output

	# Test that spaces and punctuation are treated as separate tokens
	exp_output = ["Hel", "ell", "llo", "lo,", " wo", "wor", "orl", "rld", "ld!"]
	@test text_to_trigrams("Hello, world!"; add_word = false) == exp_output

	# Test with a string that includes single-character tokens affecting neighboring tokens
	# Expecting the single-character tokens to not produce separate trigrams but to influence surrounding tokens
	@test text_to_trigrams("A cat."; add_word = false) == [" ca", "cat", "at."]

	# Test with an empty string, expecting an empty array
	@test text_to_trigrams("") == []

	# Test a complex case with special characters, spaces, and punctuation
	# This checks that the function handles various types of tokens correctly
	@test text_to_trigrams("It's rain-ing!"; add_word = false) ==
		  ["It'", " ra", "rai", "ain", "in-", "-in", "ing", "ng!"]

	# Test to ensure correct handling of multiple adjacent spaces and punctuation
	# Spaces and punctuation should be treated as tokens but not produce trigrams
	@test text_to_trigrams("Wow...  That's amazing!"; add_word = false) ==
		  ["Wow", "ow.", ".  ", "Tha", "hat", "at'", " am",
		"ama", "maz", "azi", "zin", "ing", "ng!"]

	# Special characters
	text_to_trigrams("a!@ #\$%^"; add_word = false) == []
end

@testset "text_to_trigrams_hashed" begin
	# Test basic text conversion to hashed trigrams
	exp_output = [
		"Thi", "his", "is ", "This", " is", "is ", "is", " te", "tes", "est", "st.", "test"]
	@test text_to_trigrams_hashed("This is a test."; add_word = true) ==
		  Set(hash.(exp_output))

	# Test converting without adding the word itself
	exp_output = ["Thi", "his", "is ", " is", "is ", " te", "tes", "est", "st."]
	@test text_to_trigrams_hashed("This is a test."; add_word = false) ==
		  Set(hash.(exp_output))

	# Test that unique trigrams produce a set of unique hashes
	# "hello" produces 3 unique trigrams, expecting 3 unique hash values
	@test length(text_to_trigrams_hashed("hello"; add_word = false)) == 3

	# Test with a string of repeating characters, which should still produce unique trigrams
	@test text_to_trigrams_hashed("A cat."; add_word = false) ==
		  Set(hash.([" ca", "cat", "at."]))

	# Test handling of special characters and spaces -- nothing produces (too short)
	text_to_trigrams_hashed("a!@ #\$%^"; add_word = false) == Set()

	# Test with an empty string, it's empty
	@test text_to_trigrams_hashed("") == Set()

	# Test to ensure no duplicate hash values in case of repeating patterns within the input string
	# For a pattern that repeats, like "ababab", the number of unique trigrams should be 2
	@test text_to_trigrams_hashed("ababab"; add_word = false) == Set(hash.(["aba", "bab"]))

	# Test a complex sentence with various characters, expecting a mix of unique hashes
	# The exact number of unique hashes is less important than ensuring we're getting a non-zero, plausible count
	@test text_to_trigrams_hashed("Complex sentence: 123!") ==
		  Set(hash.(text_to_trigrams("Complex sentence: 123!")))
end

@testset "split_into_code_and_sentences" begin
	# Test basic sentence splitting
	input = "This is a test. This is another test."
	sentences, group_ids = split_into_code_and_sentences(input)
	@test sentences == ["This is a test.", " This is another test."]
	@test join(sentences, "") == input # lossless
	@test group_ids == [1, 2]

	# Test handling of code blocks and inline code
	input = """Here is a code block: 
	  ```julia
	  code here
	  ```
	  and `inline code`."""
	sentences, group_ids = split_into_code_and_sentences(input)
	@test sentences == ["Here is a code block: ", "\n", "```julia", "\n",
		"code here", "\n", "```", "\n", "and ", "`inline code`", "."]
	@test join(sentences, "") == input
	@test group_ids == [1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7]

	## Multi-faceted code
	input = """Here is a code block: 
	```julia
	code here
	```
	and `inline code`.
	Sentences here.
	Bullets:
	- I like this
	- But does it work?
	```julia
	another code
	```
	1. Tester
	Third sentence - but what happened.
	"""
	sentences, group_ids = split_into_code_and_sentences(input)
	@test sentences ==
		  [
		"Here is a code block: ", "\n", "```julia", "\n", "code here", "\n", "```", "\n",
		"and ", "`inline code`", ".", "\n", "Sentences here.", "\n", "Bullets:", "\n", "- ",
		"I like this", "\n", "- ", "But does it work?", "\n", "```julia", "\n", "another code",
		"\n", "```", "\n", "1. ", "Tester", "\n", "Third sentence - but what happened.", "\n"]
	@test join(sentences, "") == input
	@test group_ids == [1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
		15, 16, 17, 18, 19, 19, 19, 19, 19, 20, 21, 22, 23, 24, 25]
end

@testset "getpropertynested" begin
	# Direct Match Tests
	kw = (; abc = (; def = "x"))
	@test getpropertynested(kw, [:abc], :def) == "x"
	@test getpropertynested(kw, [:abc], :ghi, "default") == "default"

	# Nested Match Tests
	kw = (; abc = (; def = (; ghi = "y")))
	@test getpropertynested(kw, [:abc, :def], :ghi) == "y"
	@test getpropertynested(kw, [:abc, :def], :xyz, "default") == "default"

	# No Match Tests
	kw = (; abc = (; def = "x"))
	@test getpropertynested(kw, [:xyz], :def, "default") == "default"
	@test getpropertynested(kw, [:abc, :def], :ghi, "default") == "default"
	# Complex Nested Match Tests
	kw = (; abc = (; def = (; ghi = (; jkl = "z"))))
	@test getpropertynested(kw, [:abc, :def, :ghi], :jkl) == "z"
	@test getpropertynested(kw, [:abc, :def, :ghi], :mno, "default") == "default"
end

@testset "setpropertynested" begin
	# Direct Set Tests
	kw = (; abc = (; def = "x"))
	modified_kw = setpropertynested(kw, [:abc], :def, "y")
	@test modified_kw == (; abc = (; def = "y"))

	# Nested Set Tests
	kw = (; abc = (; def = (; ghi = "x")))
	modified_kw = setpropertynested(kw, [:abc, :def], :ghi, "y")
	@test modified_kw == (; abc = (; def = (; ghi = "y"), ghi = "y"))

	# New Key Set Tests
	kw = (; abc = (; def = "x"))
	modified_kw = setpropertynested(kw, [:abc], :ghi, "y")
	@test modified_kw == (; abc = (; def = "x", ghi = "y"))

	# Complex Nested Set Tests
	kw = (; abc = (; def = (; ghi = (; jkl = "x"))))
	modified_kw = setpropertynested(kw, [:abc, :def, :ghi], :jkl, "y")
	@test modified_kw == (; abc = (; jkl = "y", def = (; jkl = "y", ghi = (; jkl = "y"))))

	# Set In Non-Existent Nested Key
	kw = (; abc = (; def = "x"))
	modified_kw = setpropertynested(kw, [:xyz], :ghi, "y")
	@test modified_kw == (; abc = (; def = "x"))
end

@testset "merge_kwargs_nested" begin
	# Basic Merge
	nt1 = (; a = 1, b = 2)
	nt2 = (; b = 3, c = 4)
	expected = (; a = 1, b = 3, c = 4)
	@test merge_kwargs_nested(nt1, nt2) == expected

	# Nested Merge
	nt1 = (; a = (; x = 1), b = 2)
	nt2 = (; a = (; y = 2), c = 3)
	expected = (; a = (; y = 2, x = 1), b = 2, c = 3)
	@test merge_kwargs_nested(nt1, nt2) == expected

	# Deep Nested Merge
	nt1 = (; a = (; x = (; i = 1)), b = 2)
	nt2 = (; a = (; x = (; j = 2)), c = 3)
	expected = (; a = (; x = (; j = 2, i = 1)), b = 2, c = 3)
	@test merge_kwargs_nested(nt1, nt2) == expected

	# Override with Non-NamedTuple
	nt1 = (; a = (; x = 1), b = 2)
	nt2 = (; a = "Not a NamedTuple", c = 3)
	expected = (; a = "Not a NamedTuple", b = 2, c = 3)
	@test merge_kwargs_nested(nt1, nt2) == expected

	# Merge with Empty NamedTuple
	nt1 = NamedTuple()
	nt2 = (; a = 1, b = (; c = 2))
	expected = (; a = 1, b = (; c = 2))
	@test merge_kwargs_nested(nt1, nt2) == expected

	nt1 = (; a = 1, b = (; c = 2))
	nt2 = NamedTuple()
	expected = (; a = 1, b = (; c = 2))
	@test merge_kwargs_nested(nt1, nt2) == expected
end

@testset "pack_bits,unpack_bits" begin
	### Test for vectors
	# Basic functionality
	bin = rand(Bool, 128)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Edge cases
	# Test with all true values
	bin = trues(128)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Test with all false values
	bin = falses(128)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Test with alternating true and false values
	bin = Bool[mod(i, 2) == 0 for i in 1:128]
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# empty vector
	bin_empty = Bool[]
	binint_empty = pack_bits(bin_empty)
	binx_empty = unpack_bits(binint_empty)
	@test bin_empty == binx_empty

	# Invalid input
	# Test with length not divisible by 64
	bin = rand(Bool, 130)
	@test_throws AssertionError pack_bits(bin)
	@test_throws ArgumentError pack_bits(rand(Float32, 128))
	@test_throws ArgumentError unpack_bits(rand(Float32, 128))

	### Test for matrices
	# Basic functionality
	bin = rand(Bool, 128, 10)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Edge cases
	# Test with all true values
	bin = trues(128, 10)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Test with all false values
	bin = falses(128, 10)
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Test with alternating true and false values
	bin = Bool[mod(i, 2) == 0 for i in 1:128, j in 1:10]
	binint = pack_bits(bin)
	binx = unpack_bits(binint)
	@test bin == binx

	# Invalid input
	# Test with number of rows not divisible by 64
	bin = rand(Bool, 130, 10)
	@test_throws AssertionError pack_bits(bin)
	# Wrong number type
	@test_throws ArgumentError pack_bits(rand(Float32, 128, 10))
	@test_throws ArgumentError unpack_bits(rand(Float32, 128, 10))
end

@testset "preprocess_tokens" begin
	stemmer = Snowball.Stemmer("english")
	stopwords = Set([
		"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in",
		"into", "is", "it", "no", "not", "of", "on", "or", "such", "some", "that", "the",
		"their", "then", "there", "these", "they", "this", "to", "was", "will", "with"])
	# Empty string
	@test preprocess_tokens("") == []

	# Simple case
	@test preprocess_tokens("This is a test."; stopwords) == ["test"]

	# Case insensitive
	@test preprocess_tokens("This Is A Test."; stopwords) == ["test"]

	# Punctuation and numbers
	@test preprocess_tokens(
		"This is a test, with punctuation and 123 numbers!", stemmer; stopwords) ==
		  ["test", "punctuat", "number"]

	# Unicode and accents
	@test preprocess_tokens(
		"Thís is à tést wîth Ünïcôdë and áccênts.", stemmer; stopwords) ==
		  ["test", "unicod", "accent"]

	# Multiple spaces
	@test preprocess_tokens(
		"This  is a   test with   multiple    spaces.", stemmer; stopwords) ==
		  ["test", "multipl", "space"]

	# Stopwords
	@test preprocess_tokens(
		"This is a test with some stopwords like the and is.", stemmer; stopwords) ==
		  ["test", "stopword", "like"]

	# Stemming
	@test preprocess_tokens(
		"This is a test with some words for stemming like testing and tested.",
		stemmer; stopwords) == ["test", "word", "stem", "like", "test", "test"]

	# Long text
	long_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl. Sed euismod, nulla sit amet aliquam lacinia, nisl nisl aliquam nisl, nec aliquam nisl nisl sit amet nisl."
	@test preprocess_tokens(long_text, stemmer; stopwords) ==
		  ["lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipisc", "elit",
		"sed", "euismod", "nulla", "sit", "amet", "aliquam", "lacinia", "nisl", "nisl",
		"aliquam", "nisl", "nec", "aliquam", "nisl", "nisl", "sit", "amet", "nisl",
		"sed", "euismod", "nulla", "sit", "amet", "aliquam", "lacinia", "nisl", "nisl",
		"aliquam", "nisl", "nec", "aliquam", "nisl", "nisl", "sit", "amet", "nisl",
		"sed", "euismod", "nulla", "sit", "amet", "aliquam", "lacinia", "nisl", "nisl",
		"aliquam", "nisl", "nec", "aliquam", "nisl", "nisl", "sit", "amet", "nisl"]

	# Edge case: non-English text
	@test preprocess_tokens("Ceci n'est pas une pipe.", stemmer; stopwords) ==
		  ["ceci", "est", "pas", "une", "pipe"]

	# Vector of inputs
	@test preprocess_tokens(
		["This is a test, with punctuation and 123 numbers!",
			"This is a test, with punctuation and 123 numbers!"],
		stemmer;
		stopwords) == [["test", "punctuat", "number"], ["test", "punctuat", "number"]]
end

@testset "reciprocal_rank_fusion" begin
	# Test with two simple lists
	positions, scores = reciprocal_rank_fusion([1, 2, 3], [4, 5, 6]; k = 0)
	@test Set(positions) == Set([1, 2, 3, 4, 5, 6])
	@test Set(positions[1:2]) == Set([1, 4])
	@test Set(positions[3:4]) == Set([2, 5])
	@test Set(positions[5:6]) == Set([3, 6])
	@test scores == Dict(1 => 1.0, 2 => 0.5, 3 => 0.3333333333333333,
		4 => 1.0, 5 => 0.5, 6 => 0.3333333333333333)

	# Test with overlapping lists
	positions, scores = reciprocal_rank_fusion([1, 2, 3], [2, 3, 4]; k = 0)
	@test Set(positions) == Set([2, 3, 1, 4])
	@test positions[1] == 2
	@test positions[2] == 1
	@test positions[3] == 3
	@test positions[4] == 4

	# Higher discount to reward more appearances
	positions, scores = reciprocal_rank_fusion([1, 2, 3], [2, 3, 4]; k = 60)
	@test Set(positions) == Set([2, 3, 1, 4])
	@test positions[1] == 2
	@test positions[2] == 3
	@test positions[3] == 1
	@test positions[4] == 4

	# Test with three lists
	positions, scores = reciprocal_rank_fusion([1, 2, 3], [2, 3, 4], [3, 4, 5]; k = 0)
	@test Set(positions) == Set([3, 2, 4, 1, 5])
	@test positions[1] == 3
	@test positions[2] == 2
	@test positions[3] == 1
	@test positions[4] == 4
	@test positions[5] == 5

	# Test with empty list
	@test reciprocal_rank_fusion([]; k = 0) == ([], Dict{Int, Float64}())

	# Test with one empty and one non-empty list
	@test reciprocal_rank_fusion([], [1, 2, 3]; k = 0) ==
		  ([1, 2, 3], Dict(1 => 1.0, 2 => 0.5, 3 => 0.3333333333333333))

	# Test with different lengths of lists
	positions, scores = reciprocal_rank_fusion([1, 2], [3, 4, 5]; k = 0)
	@test Set(positions) == Set([1, 2, 3, 4, 5])
	@test Set(positions[1:2]) == Set([1, 3])
	@test Set(positions[3:4]) == Set([2, 4])
	@test positions[5] == 5

	## Paired reciprocal rank
	positions1 = [1, 2, 3, 4, 5]
	scores1 = [0.9, 0.8, 0.7, 0.6, 0.5]
	positions2 = [3, 4, 5, 6, 7]
	scores2 = [0.5, 0.6, 0.7, 0.9, 0.9]

	merged, scores = reciprocal_rank_fusion(positions1, scores1, positions2, scores2; k = 0)
	@test length(merged) == 7
	@test Set(merged) == Set(1:7)
	@test merged[1] == 1
	@test scores[1] == 0.9
	@test merged[2] == 3
	@test scores[3] == 0.7 / 3 + 0.5
	@test merged[end] == 7
	@test scores[7] == 0.9 / 5

	merged, scores = reciprocal_rank_fusion(
		positions1, scores1, positions2, scores2; k = 60)
	@test length(merged) == 7
	@test merged[1] == 3
	@test merged[2] == 4
	@test merged[3] == 5
	@test scores[3] > scores[4]
	@test scores[4] > scores[5]
	@test scores[5] > scores[6]
	@test scores[6] > scores[7]
end

@testset "score_to_unit_scale" begin
	# Test with a normal range of values
	x = [1.0, 2.0, 3.0, 4.0, 5.0]
	scaled_x = score_to_unit_scale(x)
	@test extrema(scaled_x) == (0.0, 1.0)

	# Test with all values the same
	y = [2.0, 2.0, 2.0, 2.0, 2.0]
	scaled_y = score_to_unit_scale(y)
	@test all(scaled_y .== 1.0)

	# Test with a single value
	z = [3.0]
	scaled_z = score_to_unit_scale(z)
	@test scaled_z == [1.0]

	# Test with a range of negative values
	w = [-5.0, -4.0, -3.0, -2.0, -1.0]
	scaled_w = score_to_unit_scale(w)
	@test extrema(scaled_w) == (0.0, 1.0)

	# Test with a mix of positive and negative values
	v = [-1.0, 0.0, 1.0]
	scaled_v = score_to_unit_scale(v)
	@test extrema(scaled_v) == (0.0, 1.0)

	# Test with empty vector
	@test score_to_unit_scale(Float32[]) |> isempty
end
