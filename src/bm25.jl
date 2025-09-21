function Base.hcat(
        d1::DocumentTermMatrix{<:AbstractSparseMatrix},
        d2::DocumentTermMatrix{<:AbstractSparseMatrix}
)
    tf_, vocab_ = vcat_labeled_matrices(tf(d1), vocab(d1), tf(d2), vocab(d2))
    vocab_lookup_ = Dict(t => i for (i, t) in enumerate(vocab_))

    ## decompose tf for efficient ops
    N, M = size(tf_)
    I, J, V = findnz(tf_)
    doc_freq = zeros(Int, M)
    @inbounds for j in eachindex(J, V)
        if V[j] > 0
            doc_freq[J[j]] += 1
        end
    end
    idf = @. log(1.0f0 + (N - doc_freq + 0.5f0) / (doc_freq + 0.5f0))
    doc_lengths = zeros(Float32, N)
    @inbounds for i in eachindex(I, V)
        if V[i] > 0
            doc_lengths[I[i]] += V[i]
        end
    end
    sumdl = sum(doc_lengths)
    doc_rel_length_ = sumdl == 0 ? zeros(Float32, N) :
                      convert(Vector{Float32}, (doc_lengths ./ (sumdl / N)))
    return DocumentTermMatrix(tf_, vocab_, vocab_lookup_, idf, doc_rel_length_)
end

"""
document_term_matrix(
	documents::AbstractVector{<:AbstractVector{T}};
	min_term_freq::Int = 1, max_terms::Int = typemax(Int)) where {T <: AbstractString}

Builds a sparse matrix of term frequencies and document lengths from the given vector of documents wrapped in type `DocumentTermMatrix`.

Expects a vector of preprocessed (tokenized) documents, where each document is a vector of strings (clean tokens).

Returns: `DocumentTermMatrix`

# Arguments
- `documents`: A vector of documents, where each document is a vector of terms (clean tokens).
- `min_term_freq`: The minimum frequency a term must have to be included in the vocabulary, eg, `min_term_freq = 2` means only terms that appear at least twice will be included.
- `max_terms`: The maximum number of terms to include in the vocabulary, eg, `max_terms = 100` means only the 100 most frequent terms will be included.

# Example
```
documents = [["this", "is", "a", "test"], ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
dtm = document_term_matrix(documents)
```
"""
function document_term_matrix(
        documents::AbstractVector{<:AbstractVector{T}};
        min_term_freq::Int = 1,
        max_terms::Int = typemax(Int)
) where {T <: AbstractString}
    ## Calculate term frequencies, sort descending
    counts = Dict{T, Int}()
    @inbounds for doc in documents
        for term in doc
            counts[term] = get(counts, term, 0) + 1
        end
    end
    counts = sort(collect(counts), by = x -> -x[2]) |> Base.Fix2(first, max_terms) |>
             Base.Fix1(filter!, x -> x[2] >= min_term_freq)
    ## Create vocabulary
    vocab = convert(Vector{T}, getindex.(counts, 1))
    vocab_lookup = Dict{T, Int}(term => i for (i, term) in enumerate(vocab))
    N = length(documents)
    doc_freq = zeros(Int, length(vocab))
    doc_lengths = zeros(Float32, N)
    ## Term frequency matrix to be recorded via its sparse entries: I, J, V
    # term_freq = spzeros(Float32, N, length(vocab))
    I, J, V = Int[], Int[], Float32[]

    unique_terms = Set{eltype(vocab)}()
    sizehint!(unique_terms, 1000)
    for di in eachindex(documents)
        empty!(unique_terms)
        doc = documents[di]
        @inbounds for t in doc
            doc_lengths[di] += 1
            tid = get(vocab_lookup, t, nothing)
            tid === nothing && continue
            push!(I, di)
            push!(J, tid)
            push!(V, 1.0f0)
            if !(t in unique_terms)
                doc_freq[tid] += 1
                push!(unique_terms, t)
            end
        end
    end
    ## combine repeated terms with `+`
    term_freq = sparse(I, J, V, N, length(vocab), +)
    idf = @. log(1.0f0 + (N - doc_freq + 0.5f0) / (doc_freq + 0.5f0))
    sumdl = sum(doc_lengths)
    doc_rel_length = sumdl == 0 ? zeros(Float32, N) : doc_lengths ./ (sumdl / N)
    DocumentTermMatrix(term_freq, vocab, vocab_lookup, idf, doc_rel_length)
end

function document_term_matrix(documents::AbstractVector{<:AbstractString})
    document_term_matrix(preprocess_tokens(documents))
end

"""
bm25(
	dtm::AbstractDocumentTermMatrix, query::AbstractVector{<:AbstractString};
	k1::Float32 = 1.2f0, b::Float32 = 0.75f0, normalize::Bool = false, normalize_max_tf::Real = 3,
	normalize_min_doc_rel_length::Float32 = 1.0f0, kwargs...)

Scores all documents in `dtm` based on the `query`.

References: https://opensourceconnections.com/blog/2015/10/16/bm25-the-next-generation-of-lucene-relevation/

# Arguments
- `dtm`: A `DocumentTermMatrix` object.
- `query`: A vector of query tokens.
- `k1`: The k1 parameter for BM25.
- `b`: The b parameter for BM25.
- `normalize`: Whether to normalize the scores (returns scores between 0 and 1). 
Theoretically, if you choose `normalize_max_tf` and `normalize_min_doc_rel_length` to be too low, you could get scores greater than 1.
- `normalize_max_tf`: The maximum term frequency to normalize to. 3 is a good default (assumes max 3 hits per document).
- `normalize_min_doc_rel_length`: The minimum document relative length to normalize to. 0.5 is a good default.
Ideally, pick the minimum document relative length of the corpus that is non-zero
`min_doc_rel_length = minimum(x for x in doc_rel_length(chunkdata(key_index)) if x > 0) |> Float32`

# Example
```
documents = [["this", "is", "a", "test"], ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
dtm = document_term_matrix(documents)
query = ["this"]
scores = bm25(dtm, query)
# Returns array with 3 scores (one for each document)
```

Normalization is done by dividing the score by the maximum possible score (given some assumptions).
It's useful to be get results in the same range as cosine similarity scores and when comparing different queries or documents.

```
documents = [["this", "is", "a", "test"], ["this", "is", "another", "test"], ["foo", "bar", "baz"]]
dtm = document_term_matrix(documents)
query = ["this"]
scores = bm25(dtm, query)
scores_norm = bm25(dtm, query; normalize = true)

## Make it more accurate for your dataset/index
normalize_max_tf = 3 # assume max term frequency is 3 (what is likely for your dataset? depends on chunk size, preprocessing, etc.)
normalize_min_doc_rel_length = minimum([x for x in doc_rel_length(dtm) if x > 0]) |> Float32
scores_norm = bm25(dtm, query; normalize = true, normalize_max_tf, normalize_min_doc_rel_length)
```
"""
function bm25(
        dtm::AbstractDocumentTermMatrix,
        query::AbstractVector{<:AbstractString};
        k1::Float32 = 1.2f0,
        b::Float32 = 0.75f0,
        normalize::Bool = false,
        normalize_max_tf::Real = 3,
        normalize_min_doc_rel_length::Float32 = 0.5f0,
        kwargs...
)
    @assert normalize_max_tf > 0 "normalize_max_tf term frequency must be positive (got $normalize_max_tf)"
    @assert normalize_min_doc_rel_length > 0 "normalize_min_doc_rel_length must be positive (got $normalize_min_doc_rel_length)"

    scores = zeros(Float32, size(tf(dtm), 1))
    ## Identify non-zero items to leverage the sparsity
    nz_rows = rowvals(tf(dtm))
    nz_vals = nonzeros(tf(dtm))
    max_score = 0.0f0
    for i in eachindex(query)
        t = query[i]
        t_id = get(vocab_lookup(dtm), t, nothing)
        t_id === nothing && continue
        idf_ = idf(dtm)[t_id]
        # Scan only documents that have this token
        @inbounds @simd for j in nzrange(tf(dtm), t_id)
            ## index into the sparse matrix
            di, tf_ = nz_rows[j], nz_vals[j]
            doc_len = doc_rel_length(dtm)[di]
            tf_top = (tf_ * (k1 + 1.0f0))
            tf_bottom = (tf_ + k1 * (1.0f0 - b + b * doc_len))
            score = idf_ * tf_top / tf_bottom
            ## @info "di: $di, tf: $tf, doc_len: $doc_len, idf: $idf, tf_top: $tf_top, tf_bottom: $tf_bottom, score: $score"
            scores[di] += score
        end
        ## Once per token, calculate max score
        ## assumes max term frequency is `normalize_max_tf` and min document relative length is `normalize_min_doc_rel_length`
        if normalize
            max_score += idf_ * (normalize_max_tf * (k1 + 1.0f0)) / (normalize_max_tf +
                          k1 * (1.0f0 - b + b * normalize_min_doc_rel_length))
        end
    end
    if normalize && !iszero(max_score)
        scores ./= max_score
    elseif normalize && iszero(max_score)
        ## happens only with empty queries, so scores is zero anyway
        @warn "BM25: `max_score` is zero, so scores are not normalized. Returning unnormalized scores (all zero)."
    end

    return scores
end

"""
max_bm25_score(
	dtm::AbstractDocumentTermMatrix, query_tokens::AbstractVector{<:AbstractString};
	k1::Float32 = 1.2f0, b::Float32 = 0.75f0, max_tf::Real = 3,
	min_doc_rel_length::Float32 = 0.5f0)

Returns the maximum BM25 score that can be achieved for a given query (assuming the `max_tf` matches and the `min_doc_rel_length` being the smallest document relative length).
Good for normalizing BM25 scores.

# Example
```
max_score = max_bm25_score(chunkdata(key_index), query_tokens)
```
"""
function max_bm25_score(
        dtm::AbstractDocumentTermMatrix,
        query_tokens::AbstractVector{<:AbstractString};
        k1::Float32 = 1.2f0,
        b::Float32 = 0.75f0,
        max_tf::Real = 3,
        min_doc_rel_length::Float32 = 0.5f0
)
    max_score = 0.0f0
    @inbounds for t in query_tokens
        t_id = get(vocab_lookup(dtm), t, nothing)
        t_id === nothing && continue

        idf_ = idf(dtm)[t_id]

        # Find maximum tf (term frequency) for this term in any document - pre-set in kwargs!
        # eg, `max_tf = maximum(@view(tf(dtm)[:, t_id]))` but that would be a bit extreme and slow

        # Find first non-zero element in doc lengths -- pre-set in kwargs!
        # eg, `min_doc_rel_length = minimum(x for x in doc_rel_length(chunkdata(key_index)) if x > 0) |> Float32`

        # Maximum tf component assuming perfect match
        tf_top = (max_tf * (k1 + 1.0f0))
        tf_bottom = (max_tf + k1 * (1.0f0 - b + b * min_doc_rel_length))
        max_score += idf_ * tf_top / tf_bottom
    end
    return max_score
end
