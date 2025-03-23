# Base Dependencies
# ------------------
using Base: parent
using StructTypes, JSON3

# Files inclusion 
# ---------------
include("chunk.jl")
include("document_term_matrix.jl")
include("index.jl") 
include("rag_result.jl")


# Shared functions 
# ----------------
"""
    Base.getindex
"""
function Base.getindex(
    ci::AbstractDocumentIndex,
    candidate::AbstractCandidateChunks,
    field::Symbol
)
    throw(ArgumentError("Not implemented"))
end

function Base.getindex(index::AbstractChunkIndex, id::Symbol)
    id == indexid(index) ? index : nothing
end

function Base.getindex(index::AbstractMultiIndex, id::Symbol)
    id == indexid(index) && return index
    idx = findfirst(x -> indexid(x) == id, indexes(index))
    isnothing(idx) ? nothing : indexes(index)[idx]
end

function Base.getindex(
    ci::AbstractChunkIndex,
    candidate::CandidateChunks{TP, TD},
    field::Symbol = :chunks; 
    sorted::Bool = false
) where {TP <: Integer, TD <: Real}
    @assert field in [:chunks, :embeddings, :chunkdata, :sources, :scores] "Only `chunks`, `embeddings`, `chunkdata`, `sources`, `scores` fields are supported for now"
    ## embeddings is a compatibility alias, use chunkdata
    field = field == :embeddings ? :chunkdata : field

    if indexid(ci) == indexid(candidate)
        # Sort if requested
        sorted_idx = sorted ? sortperm(scores(candidate), rev = true) :
                     eachindex(scores(candidate))
        sub_index = view(ci, candidate)
        if field == :chunks
            chunks(sub_index)[sorted_idx]
        elseif field == :chunkdata
            ## If embeddings, chunks are columns
            ## If keywords (DTM), chunks are rows
            chkdata = chunkdata(sub_index, sorted_idx)
        elseif field == :sources
            sources(sub_index)[sorted_idx]
        elseif field == :scores
            scores(candidate)[sorted_idx]
        end
    else
        if field == :chunks
            eltype(chunks(ci))[]
        elseif field == :chunkdata
            chkdata = chunkdata(ci)
            isnothing(chkdata) && return nothing
            TypeItem = typeof(chkdata)
            init_dim = ntuple(i -> 0, ndims(chkdata))
            TypeItem(undef, init_dim)
        elseif field == :sources
            eltype(sources(ci))[]
        elseif field == :scores
            TD[]
        end
    end
end

function Base.getindex(
    mi::MultiIndex,
    candidate::CandidateChunks{TP, TD},
    field::Symbol = :chunks; sorted::Bool = false
) where {TP <: Integer, TD <: Real}
    ## Always sorted!
    @assert field in [:chunks, :sources, :scores] "Only `chunks`, `sources`, `scores` fields are supported for now"
    valid_index = findfirst(x -> indexid(x) == indexid(candidate), indexes(mi))
    if isnothing(valid_index) && field == :chunks
        String[]
    elseif isnothing(valid_index) && field == :sources
        String[]
    elseif isnothing(valid_index) && field == :scores
        TD[]
    else
        getindex(indexes(mi)[valid_index], candidate, field)
    end
end

function Base.getindex(
    ci::AbstractChunkIndex,
    candidate::MultiCandidateChunks{TP, TD},
    field::Symbol = :chunks; 
    sorted::Bool = false
) where {TP <: Integer, TD <: Real}
    @assert field in [:chunks, :embeddings, :chunkdata, :sources, :scores] "Only `chunks`, `embeddings`, `chunkdata`, `sources`, `scores` fields are supported for now"

    index_pos = findall(==(indexid(ci)), indexids(candidate))
    ## Convert to CandidateChunks and re-use method above
    cc = CandidateChunks(
        indexid(ci), positions(candidate)[index_pos], scores(candidate)[index_pos]
    )
    getindex(ci, cc, field; sorted)
end

function Base.getindex(
    mi::MultiIndex,
    candidate::MultiCandidateChunks{TP, TD},
    field::Symbol = :chunks; 
    sorted::Bool = true
) where {TP <: Integer, TD <: Real}
    @assert field in [:chunks, :sources, :scores] "Only `chunks`, `sources`, and `scores` fields are supported for now"
    if sorted
        # values can be either of chunks or sources
        # ineffective but easier to implement
        # TODO: remove the duplication later
        values = mapreduce(idxs -> getindex(idxs, candidate, field, sorted = false),
            vcat, indexes(mi))
        scores_ = mapreduce(
            idxs -> getindex(idxs, candidate, :scores, sorted = false),
            vcat, indexes(mi))
        sorted_idx = sortperm(scores_, rev = true)
        values[sorted_idx]
    else
        mapreduce(idxs -> getindex(idxs, candidate, field, sorted = false),
            vcat, indexes(mi))
    end
end

""" 
    Base.view
"""
function Base.view(index::AbstractDocumentIndex, cc::AbstractCandidateChunks)
    throw(ArgumentError("Not implemented for type $(typeof(index)) and $(typeof(cc))"))
end

Base.@propagate_inbounds function Base.view(
    index::AbstractChunkIndex, 
    cc::CandidateChunks
)
    @boundscheck let chk_vector = chunks(parent(index))
        if !checkbounds(Bool, axes(chk_vector, 1), positions(cc))
            ## Avoid printing huge position arrays, show the extremas of the attempted range
            max_pos = extrema(positions(cc))
            throw(BoundsError(chk_vector, max_pos))
        end
    end
    pos = indexid(index) == indexid(cc) ? positions(cc) : Int[]
    return SubChunkIndex(parent(index), pos)
end

Base.@propagate_inbounds function Base.view(index::SubChunkIndex, cc::CandidateChunks)
    SubChunkIndex(index, cc)
end

Base.@propagate_inbounds function Base.view(
    index::AbstractChunkIndex, 
    cc::MultiCandidateChunks
)
    valid_items = findall(==(indexid(index)), indexids(cc))
    valid_positions = positions(cc)[valid_items]
    @boundscheck let chk_vector = chunks(parent(index))
        if !checkbounds(Bool, axes(chk_vector, 1), valid_positions)
            ## Avoid printing huge position arrays, show the extremas of the attempted range
            max_pos = extrema(valid_positions)
            throw(BoundsError(chk_vector, max_pos))
        end
    end
    return SubChunkIndex(parent(index), valid_positions)
end

Base.@propagate_inbounds function Base.view(index::SubChunkIndex, cc::MultiCandidateChunks)
    SubChunkIndex(index, cc)
end
