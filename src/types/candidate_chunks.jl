""" 
	This file contains the definition of the types:
	- `CandidateChunks`
	- `MultiCandidateChunks`
	and related functions.
"""

"""
	AbstractCandidateChunks
"""

function Base.vcat(cc1::AbstractCandidateChunks, cc2::AbstractCandidateChunks)
    throw(ArgumentError("Not implemented for type $(typeof(cc1)) and $(typeof(cc2))"))
end

# combine/intersect two candidate chunks. take the maximum of the score if available
function Base.var"&"(cc1::AbstractCandidateChunks, cc2::AbstractCandidateChunks)
    throw(ArgumentError("Not implemented for type $(typeof(cc1)) and $(typeof(cc2))"))
end

""" 
	CandidateChunks

A struct for storing references to chunks in the given index (identified by `index_id`) 
called `positions` and `scores` holding the strength of similarity (=1 is the highest, 
most similar). It's the result of the retrieval stage of RAG.

# Fields
- `index_id::Symbol`: the id of the index from which the candidates are drawn
- `positions::Vector{Int}`: the positions of the candidates in the index (ie, `5` 
	refers to the 5th chunk in the index - `chunks(index)[5]`)
- `scores::Vector{Float32}`: the similarity scores of the candidates from the query (higher is better)
"""
@kwdef struct CandidateChunks{TP <: Integer, TD <: Real} <: AbstractCandidateChunks
    index_id::Symbol
    positions::Vector{TP} = Int[] # if TP is Int, then positions are indices into the index
    scores::Vector{TD} = Float32[]
end

function CandidateChunks(
        index::AbstractChunkIndex,
        positions::AbstractVector{<:Integer},
        scores::AbstractVector{<:Real} = fill(0.0f0, length(positions))
)
    CandidateChunks(
        indexid(index),
        convert(Vector{Int}, positions),
        convert(Vector{Float32}, scores)
    )
end

indexid(cc::CandidateChunks) = cc.index_id
indexids(cc::CandidateChunks) = fill(indexid(cc), length(positions(cc))) # for compatibility
positions(cc::CandidateChunks) = cc.positions
scores(cc::CandidateChunks) = cc.scores
Base.length(cc::CandidateChunks) = length(cc.positions)
StructTypes.StructType(::Type{CandidateChunks}) = StructTypes.Struct()

function Base.first(cc::CandidateChunks, k::Integer)
    sorted_idxs = sortperm(scores(cc), rev = true) |> x -> first(x, k)
    CandidateChunks(indexid(cc), positions(cc)[sorted_idxs], scores(cc)[sorted_idxs])
end

function Base.copy(cc::CandidateChunks{TP, TD}) where {TP <: Integer, TD <: Real}
    CandidateChunks{TP, TD}(indexid(cc), copy(positions(cc)), copy(scores(cc)))
end

function Base.isempty(cc::CandidateChunks)
    isempty(positions(cc))
end

function Base.var"=="(cc1::CandidateChunks, cc2::CandidateChunks)
    all(map(f -> getfield(cc1, f) == getfield(cc2, f), fieldnames(CandidateChunks)))
end

function Base.vcat(
        cc1::CandidateChunks{TP1, TD1},
        cc2::CandidateChunks{TP2, TD2}
) where {TP1 <: Integer, TP2 <: Integer, TD1 <: Real, TD2 <: Real}
    ## Check validity
    indexid(cc1) != indexid(cc2) &&
        throw(ArgumentError("Index ids must match (provided: $(indexid(cc1)) and $(indexid(cc2)))"))

    positions_ = vcat(positions(cc1), positions(cc2))
    # operates on maximum similarity principle, ie, take the max similarity
    scores_ = if !isempty(scores(cc1)) && !isempty(scores(cc2))
        vcat(scores(cc1), scores(cc2))
    else
        TD1[]
    end
    if !isempty(scores_)
        ## Get sorted by maximum similarity (scores are similarity)
        sorted_idxs = sortperm(scores_, rev = true)
        positions_sorted = view(positions_, sorted_idxs)
        ## get the positions of unique elements
        unique_idxs = unique(i -> positions_sorted[i], eachindex(positions_sorted))
        positions_ = positions_sorted[unique_idxs]
        ## apply the sorting and then the filtering
        scores_ = view(scores_, sorted_idxs)[unique_idxs]
    else
        positions_ = unique(positions_)
    end
    CandidateChunks(indexid(cc1), positions_, scores_)
end

function Base.var"&"(
        cc1::CandidateChunks{TP1, TD1},
        cc2::CandidateChunks{TP2, TD2}
) where {TP1 <: Integer, TP2 <: Integer, TD1 <: Real, TD2 <: Real}
    ##
    indexid(cc1) != indexid(cc2) && return CandidateChunks(; index_id = indexid(cc1))

    positions_ = intersect(positions(cc1), positions(cc2))

    scores_ = if !isempty(scores(cc1)) && !isempty(scores(cc2))
        # identify maximum scores from each CC
        scores_dict = Dict(pos => -Inf for pos in positions_)
        # scan the first CC
        for i in eachindex(positions(cc1), scores(cc1))
            pos = positions(cc1)[i]
            if haskey(scores_dict, pos)
                scores_dict[pos] = max(scores_dict[pos], scores(cc1)[i])
            end
        end
        # scan the second CC
        for i in eachindex(positions(cc2), scores(cc2))
            pos = positions(cc2)[i]
            if haskey(scores_dict, pos)
                scores_dict[pos] = max(scores_dict[pos], scores(cc2)[i])
            end
        end
        [scores_dict[pos] for pos in positions_]
    else
        TD1[]
    end
    ## Sort by maximum similarity
    if !isempty(scores_)
        sorted_idxs = sortperm(scores_, rev = true)
        positions_ = positions_[sorted_idxs]
        scores_ = scores_[sorted_idxs]
    end

    CandidateChunks(indexid(cc1), positions_, scores_)
end

"""
	MultiCandidateChunks

A struct for storing references to multiple sets of chunks across different indices. 
Each set of chunks is identified by an `index_id` in `index_ids`, with corresponding 
`positions` in the index and `scores` indicating the strength of similarity.

This struct is useful for scenarios where candidates are drawn from multiple indices, 
and there is a need to keep track of which candidates came from which index.

# Fields
- `index_ids::Vector{Symbol}`: the ids of the indices from which the candidates are drawn
- `positions::Vector{TP}`: the positions of the candidates in their respective indices
- `scores::Vector{TD}`: the similarity scores of the candidates from the query
"""
@kwdef struct MultiCandidateChunks{TP, TD} <: AbstractCandidateChunks
    # Records the indices that the candidate chunks are from
    index_ids::Vector{Symbol}
    # Records the positions of the candidate chunks in the index
    positions::Vector{TP} = Int[]
    scores::Vector{TD} = Float32[]
end

function MultiCandidateChunks(
        index::AbstractChunkIndex,
        positions::AbstractVector{<:Integer},
        scores::AbstractVector{<:Real} = fill(0.0f0, length(positions))
)
    index_ids = fill(indexid(index), length(positions))
    MultiCandidateChunks(
        index_ids, convert(Vector{Int}, positions), convert(Vector{Float32}, scores)
    )
end

indexids(cc::MultiCandidateChunks) = cc.index_ids
positions(cc::MultiCandidateChunks) = cc.positions
scores(cc::MultiCandidateChunks) = cc.scores
Base.length(cc::MultiCandidateChunks) = length(positions(cc))
StructTypes.StructType(::Type{MultiCandidateChunks}) = StructTypes.Struct()

function Base.first(cc::MultiCandidateChunks, k::Integer)
    sorted_idxs = sortperm(scores(cc), rev = true) |> x -> first(x, k)
    MultiCandidateChunks(
        indexids(cc)[sorted_idxs], positions(cc)[sorted_idxs], scores(cc)[sorted_idxs])
end

function Base.copy(cc::MultiCandidateChunks{TP, TD}) where {TP <: Integer, TD <: Real}
    MultiCandidateChunks{TP, TD}(copy(indexids(cc)), copy(positions(cc)), copy(scores(cc)))
end

function Base.isempty(cc::MultiCandidateChunks)
    isempty(positions(cc))
end

function Base.var"=="(cc1::MultiCandidateChunks, cc2::MultiCandidateChunks)
    all(
        getfield(cc1, f) == getfield(cc2, f) for f in fieldnames(MultiCandidateChunks))
end

function Base.vcat(
        cc1::MultiCandidateChunks{TP1, TD1},
        cc2::MultiCandidateChunks{TP2, TD2}
) where {TP1 <: Integer, TP2 <: Integer, TD1 <: Real, TD2 <: Real}
    # operates on maximum similarity principle, ie, take the max similarity
    scores_ = if !isempty(scores(cc1)) && !isempty(scores(cc2))
        vcat(scores(cc1), scores(cc2))
    else
        TD1[]
    end
    positions_ = vcat(positions(cc1), positions(cc2))
    # pool the index ids
    index_ids = vcat(indexids(cc1), indexids(cc2))

    if !isempty(scores_)
        ## Get sorted by maximum similarity (scores are similarity)
        sorted_idxs = sortperm(scores_, rev = true)
        view_positions = view(positions_, sorted_idxs)
        view_indices = view(index_ids, sorted_idxs)
        ## get the positions of unique elements
        unique_idxs = unique(
            i -> (view_indices[i], view_positions[i]), eachindex(
                view_positions, view_indices))
        positions_ = view_positions[unique_idxs]
        index_ids = view_indices[unique_idxs]
        ## apply the sorting and then the filtering
        scores_ = view(scores_, sorted_idxs)[unique_idxs]
    else
        unique_idxs = unique(
            i -> (positions_[i], index_ids[i]), eachindex(positions_, index_ids))
        positions_ = positions_[unique_idxs]
        index_ids = index_ids[unique_idxs]
    end
    MultiCandidateChunks(index_ids, positions_, scores_)
end

function Base.var"&"(
        mc1::MultiCandidateChunks{TP1, TD1},
        mc2::MultiCandidateChunks{TP2, TD2}
) where {TP1 <: Integer, TP2 <: Integer, TD1 <: Real, TD2 <: Real}
    ## if empty, skip the work
    if isempty(scores(mc1)) || isempty(scores(mc2))
        return MultiCandidateChunks(;
            index_ids = Symbol[], positions = TP1[], scores = TD1[])
    end

    keep_indexes = intersect(indexids(mc1), indexids(mc2))

    ## Build the scores dict from first candidates
    ## Structure: id=>position=>max_score
    scores_dict = Dict()
    for id in keep_indexes
        inner_dict = Dict()
        for (pos, score, id_) in zip(positions(mc1), scores(mc1), indexids(mc1))
            if id_ == id
                inner_dict[pos] = score
            end
        end
        scores_dict[id] = inner_dict
    end

    ## Iterate the second candidate set and directly save to output arrays
    index_ids = Symbol[]
    positions_ = TP1[]
    scores_ = TD1[]
    for i in eachindex(positions(mc2), indexids(mc2), scores(mc2))
        pos, score, id = positions(mc2)[i], scores(mc2)[i], indexids(mc2)[i]
        if haskey(scores_dict, id)
            index_dict = scores_dict[id]
            if haskey(index_dict, pos)
                ## This item was found in both -> set to true as intersection
                push!(index_ids, id)
                push!(positions_, pos)
                push!(scores_, max(index_dict[pos], score))
            end
        end
    end

    ## Sort by maximum similarity
    if !isempty(scores_)
        sorted_idxs = sortperm(scores_, rev = true)
        positions_ = positions_[sorted_idxs]
        index_ids = index_ids[sorted_idxs]
        scores_ = scores_[sorted_idxs]
    else
        ## take as is
        index_ids = Symbol[]
        positions_ = TP1[]
    end

    return MultiCandidateChunks(index_ids, positions_, scores_)
end

# Serialization for JSON3
# ------------------------

"""
	StructTypes.constructfrom(
		::Type{T},
		obj::Union{Dict, JSON3.Object}
	) where {T <: Union{CandidateChunks, MultiCandidateChunks}}

Constructor for serialization - opinionated for abstract types!
"""
function StructTypes.constructfrom(
        ::Type{T},
        obj::Union{Dict, JSON3.Object}
) where {T <: Union{CandidateChunks, MultiCandidateChunks}}
    obj = copy(obj)
    haskey(obj, :index_id) && (obj[:index_id] = Symbol(obj[:index_id]))
    haskey(obj, :index_ids) && (obj[:index_ids] = convert(Vector{Symbol}, obj[:index_ids]))
    haskey(obj, :positions) && (obj[:positions] = convert(Vector{Int}, obj[:positions]))
    haskey(obj, :scores) && (obj[:scores] = convert(Vector{Float32}, obj[:scores]))
    T(; obj...)
end

## function StructTypes.constructfrom(::Type{CandidateChunks}, obj::JSON3.Object)
##     obj = copy(obj)
##     haskey(obj, :positions) && (obj[:positions] = convert(Vector{Int}, obj[:positions]))
##     haskey(obj, :scores) && (obj[:scores] = convert(Vector{Float32}, obj[:scores]))
##     CandidateChunks(; obj...)
## end

function JSON3.read(path::AbstractString, ::Type{T}) where {T <: Union{
        CandidateChunks, MultiCandidateChunks}}
    StructTypes.constructfrom(T, JSON3.read(path))
end
