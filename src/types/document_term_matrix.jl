
""" 
	This file contains the definition of the types:
	- `DocumentTermMatrix`
	- `SubDocumentTermMatrix`
	and related functions, including those for `AbstractDocumentTermMatrix`.
"""

""" 
	AbstractDocumentTermMatrix
"""

""" 
	Base.parent(dtm::AbstractDocumentTermMatrix)

The parent of an `AbstractDocumentTermMatrix` is itself.
"""
function Base.parent(dtm::AbstractDocumentTermMatrix)
    dtm
end

""" 
	tf(dtm::AbstractDocumentTermMatrix)

Get the term frequency matrix of an `AbstractDocumentTermMatrix`.
"""
function tf(dtm::AbstractDocumentTermMatrix)
    dtm.tf
end

""" 
	vocab(dtm::AbstractDocumentTermMatrix)

Get the vocabulary vector of an `AbstractDocumentTermMatrix`, defined in `rag_interface.jl`.
"""
function vocab(dtm::AbstractDocumentTermMatrix)
    dtm.vocab
end

""" 
	vocab_lookup(dtm::AbstractDocumentTermMatrix)

Get the vocabulary lookup dictionary of an `AbstractDocumentTermMatrix`.
"""
function vocab_lookup(dtm::AbstractDocumentTermMatrix)
    dtm.vocab_lookup
end

""" 
	idf(dtm::AbstractDocumentTermMatrix)

Get the inverse document frequency vector of an `AbstractDocumentTermMatrix`.
"""
function idf(dtm::AbstractDocumentTermMatrix)
    dtm.idf
end

""" 
	doc_rel_length(dtm::AbstractDocumentTermMatrix)

Get the document relative length vector of an `AbstractDocumentTermMatrix`.
"""
function doc_rel_length(dtm::AbstractDocumentTermMatrix)
    dtm.doc_rel_length
end

""" 
	Base.var"=="(dtm1::AbstractDocumentTermMatrix, dtm2::AbstractDocumentTermMatrix) = false

Check if two `AbstractDocumentTermMatrix` objects are equal.
"""
Base.var"=="(dtm1::AbstractDocumentTermMatrix, dtm2::AbstractDocumentTermMatrix) = false

function Base.var"=="(dtm1::T, dtm2::T) where {T <: AbstractDocumentTermMatrix}
    tf(dtm1) == tf(dtm2) &&
        vocab(dtm1) == vocab(dtm2) &&
        vocab_lookup(dtm1) == vocab_lookup(dtm2) &&
        idf(dtm1) == idf(dtm2) &&
        doc_rel_length(dtm1) == doc_rel_length(dtm2)
end

""" 
	Base.hcat(d1::AbstractDocumentTermMatrix, d2::AbstractDocumentTermMatrix)

Concatenate two `AbstractDocumentTermMatrix` objects horizontally.
"""
function Base.hcat(d1::AbstractDocumentTermMatrix, d2::AbstractDocumentTermMatrix)
    throw(ArgumentError("A hcat not implemented for DTMs of type $(typeof(d1)) and $(typeof(d2))"))
end

function Base.view(
        dtm::AbstractDocumentTermMatrix, doc_idx::AbstractVector{<:Integer}, token_idx
)
    throw(ArgumentError("A view not implemented for type $(typeof(dtm)) across docs: $(typeof(doc_idx)) and tokens: $(typeof(token_idx))"))
end

Base.@propagate_inbounds function Base.view(
        dtm::AbstractDocumentTermMatrix,
        doc_idx::AbstractVector{<:Integer},
        token_idx::Colon
)
    tf_mat = tf(parent(dtm))
    @boundscheck if !checkbounds(Bool, axes(tf_mat, 1), doc_idx)
        ## Avoid printing huge position arrays, show the extremas of the attempted range
        max_pos = extrema(doc_idx)
        throw(BoundsError(tf_mat, max_pos))
    end
    ## computations on top of views of sparse arrays are expensive, materialize the view
    ## Moreover, nonzeros and rowvals accessors for SparseCSCMatrix are not defined for views
    tf_ = tf_mat[doc_idx, :]
    SubDocumentTermMatrix(dtm, tf_, collect(doc_idx))
end

"""
	DocumentTermMatrix{T<:AbstractString}

A sparse matrix of term frequencies and document lengths to allow calculation of BM25 similarity scores.
"""
struct DocumentTermMatrix{
    T1 <: AbstractMatrix{<:Real},
    T2 <: AbstractString
} <: AbstractDocumentTermMatrix
    tf::T1 # term frequency matrix, assumed to be SparseMatrixCSC{Float32, Int64}
    vocab::Vector{T2} # vocabulary
    vocab_lookup::Dict{T2, Int} # lookup table for vocabulary
    idf::Vector{Float32} # inverse document frequency
    doc_rel_length::Vector{Float32} # document relative length |d|/avgDl
end

function Base.hcat(d1::DocumentTermMatrix, d2::DocumentTermMatrix)
    tf_, vocab_ = vcat_labeled_matrices(tf(d1), vocab(d1), tf(d2), vocab(d2))
    vocab_lookup_ = Dict(t => i for (i, t) in enumerate(vocab_))

    N, _ = size(tf_)
    doc_freq = [count(x -> x > 0, col) for col in eachcol(tf_)]
    idf = @. log(1.0f0 + (N - doc_freq + 0.5f0) / (doc_freq + 0.5f0))
    doc_lengths = [count(x -> x > 0, row) for row in eachrow(tf_)]
    sumdl = sum(doc_lengths)
    doc_rel_length_ = sumdl == 0 ? zeros(Float32, N) : (doc_lengths ./ (sumdl / N))

    return DocumentTermMatrix(
        tf_, vocab_, vocab_lookup_, idf, convert(Vector{Float32}, doc_rel_length_)
    )
end

"""
	SubDocumentTermMatrix

A partial view of a DocumentTermMatrix, `tf` is MATERIALIZED for performance and fewer allocations."
"""
struct SubDocumentTermMatrix{
    T <: DocumentTermMatrix,
    T1 <: AbstractMatrix{<:Real}
} <: AbstractDocumentTermMatrix
    parent::T
    tf::T1 ## Materialize the sub-matrix, because it's too expensive to use otherwise (row-view of SparseMatrixCSC)
    positions::Vector{Int}
end

Base.parent(dtm::SubDocumentTermMatrix) = dtm.parent
positions(dtm::SubDocumentTermMatrix) = dtm.positions
tf(dtm::SubDocumentTermMatrix) = dtm.tf
vocab(dtm::SubDocumentTermMatrix) = Base.parent(dtm) |> vocab
vocab_lookup(dtm::SubDocumentTermMatrix) = Base.parent(dtm) |> vocab_lookup
idf(dtm::SubDocumentTermMatrix) = Base.parent(dtm) |> idf

Base.@propagate_inbounds function doc_rel_length(dtm::SubDocumentTermMatrix)
    view(doc_rel_length(Base.parent(dtm)), positions(dtm))
end

# hcat for SubDocumentTermMatrix does not make sense -> the vocabulary is the same / shared

""" 
	Base.view(dtm::SubDocumentTermMatrix, doc_idx::AbstractVector{<:Integer}, token_idx::Colon)

Create a view of a `SubDocumentTermMatrix` for a specific document index and all tokens.
"""
function Base.view(
        dtm::SubDocumentTermMatrix,
        doc_idx::AbstractVector{<:Integer},
        token_idx::Colon
)
    tf_mat = tf(parent(dtm))
    @boundscheck if !checkbounds(Bool, axes(tf_mat, 1), doc_idx)
        ## Avoid printing huge position arrays, show the extremas of the attempted range
        max_pos = extrema(doc_idx)
        throw(BoundsError(tf_mat, max_pos))
    end
    intersect_pos = intersect(positions(dtm), doc_idx)
    return SubDocumentTermMatrix(parent(dtm), tf_mat[intersect_pos, :], intersect_pos)
end
