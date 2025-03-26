"""
	This file contains the definition of the types:
	- `ChunkEmbeddingsIndex`
	- `ChunkKeywordsIndex`
	- `MultiIndex`
	- `SubChunkIndex`
	and related functions, including those for `AbstractDocumentIndex`.
"""


""" 
	AbstractDocumentIndex
"""

Base.parent(index::AbstractDocumentIndex) = index
indexid(index::AbstractDocumentIndex) = index.id


function chunkdata(index::AbstractDocumentIndex)
	throw(ArgumentError("`chunkdata` not implemented for $(typeof(index))"))
end
function chunkdata(index::AbstractDocumentIndex, chunk_idx::AbstractVector{<:Integer})
	throw(ArgumentError("`chunkdata` not implemented for $(typeof(index)) and chunk indices: $(typeof(chunk_idx))"))
end

function embeddings(index::AbstractDocumentIndex)
	throw(ArgumentError("`embeddings` not implemented for $(typeof(index))"))
end

function tags(index::AbstractDocumentIndex)
	throw(ArgumentError("`tags` not implemented for $(typeof(index))"))
end

function tags_vocab(index::AbstractDocumentIndex)
	throw(ArgumentError("`tags_vocab` not implemented for $(typeof(index))"))
end

function extras(index::AbstractDocumentIndex)
	throw(ArgumentError("`extras` not implemented for $(typeof(index))"))
end

function Base.vcat(i1::AbstractDocumentIndex, i2::AbstractDocumentIndex)
	throw(ArgumentError("Not implemented"))
end

""" 
	AbstractChunkIndex
"""

chunkdata(index::AbstractChunkIndex) = index.chunkdata
Base.@propagate_inbounds function chunkdata(
	index::AbstractChunkIndex, chunk_idx::AbstractVector{<:Integer},
)
	## We need this accessor because different chunk indices can have chunks in different dimensions!!
	chkdata = chunkdata(index)
	if isnothing(chkdata)
		return nothing
	end
	return view(chkdata, :, chunk_idx)
end

HasEmbeddings(::AbstractChunkIndex) = false
HasKeywords(::AbstractChunkIndex) = false
chunks(index::AbstractChunkIndex) = index.chunks
Base.length(index::AbstractChunkIndex) = length(chunks(index))
tags(index::AbstractChunkIndex) = index.tags
tags_vocab(index::AbstractChunkIndex) = index.tags_vocab
sources(index::AbstractChunkIndex) = index.sources
extras(index::AbstractChunkIndex) = index.extras

"""
	translate_positions_to_parent(index::AbstractChunkIndex, positions::AbstractVector{<:Integer})

Translate positions to the parent index. Useful to convert between positions in a view and the original index.

Used whenever a `chunkdata()` is used to re-align positions in case index is a view.
"""
function translate_positions_to_parent(index::AbstractChunkIndex, positions::AbstractVector{<:Integer})
	return positions
end

Base.var"=="(i1::AbstractChunkIndex, i2::AbstractChunkIndex) = false

function Base.var"=="(i1::T, i2::T) where {T <: AbstractChunkIndex}
	((sources(i1) == sources(i2)) && (tags_vocab(i1) == tags_vocab(i2)) &&
	 (chunkdata(i1) == chunkdata(i2)) && (chunks(i1) == chunks(i2)) &&
	 (tags(i1) == tags(i2)) && (extras(i1) == extras(i2)))
end


function Base.vcat(i1::AbstractChunkIndex, i2::AbstractChunkIndex)
	throw(ArgumentError("Not implemented"))
end

function Base.vcat(i1::T, i2::T) where {T <: AbstractChunkIndex}
	tags_, tags_vocab_ = if (isnothing(tags(i1)) || isnothing(tags(i2)))
		nothing, nothing
	elseif tags_vocab(i1) == tags_vocab(i2)
		vcat(tags(i1), tags(i2)), tags_vocab(i1)
	else
		vcat_labeled_matrices(tags(i1), tags_vocab(i1), tags(i2), tags_vocab(i2))
	end
	chunkdata_ = (isnothing(chunkdata(i1)) || isnothing(chunkdata(i2))) ? nothing :
				 hcat(chunkdata(i1), chunkdata(i2))
	extras_ = if isnothing(extras(i1)) || isnothing(extras(i2))
		nothing
	else
		vcat(extras(i1), extras(i2))
	end
	T(indexid(i1), vcat(chunks(i1), chunks(i2)),
		chunkdata_,
		tags_,
		tags_vocab_,
		vcat(sources(i1), sources(i2)),
		extras_)
end


"""
	ChunkEmbeddingsIndex

Main struct for storing document chunks and their embeddings. It also stores tags 
and sources for each chunk.

Previously, this struct was called `ChunkIndex`.

# Fields
- `id::Symbol`: unique identifier of each index (to ensure we're using the right index with `CandidateChunks`)
- `chunks::Vector{<:AbstractString}`: underlying document chunks / snippets
- `embeddings::Union{Nothing, Matrix{<:Real}}`: for semantic search
- `tags::Union{Nothing, AbstractMatrix{<:Bool}}`: for exact search, filtering, etc. 
    This is often a sparse matrix indicating which chunks have the given `tag` 
    (see `tag_vocab` for the position lookup)
- `tags_vocab::Union{Nothing, Vector{<:AbstractString}}`: vocabulary for the `tags` 
    matrix (each column in `tags` is one item in `tags_vocab` and rows are the chunks)
- `sources::Vector{<:AbstractString}`: sources of the chunks
- `extras::Union{Nothing, AbstractVector}`: additional data, eg, metadata, source code, etc.
"""
@kwdef struct ChunkEmbeddingsIndex{
	T1 <: AbstractString,
	T2 <: Union{Nothing, AbstractMatrix{<:Real}},
	T3 <: Union{Nothing, AbstractMatrix{<:Bool}},
	T4 <: Union{Nothing, AbstractVector},
} <: AbstractChunkIndex
	id::Symbol = gensym("ChunkEmbeddingsIndex")
	# underlying document chunks / snippets
	chunks::Vector{T1}
	# for semantic search
	embeddings::T2 = nothing
	# for exact search, filtering, etc.
	# expected to be some sparse structure, eg, sparse matrix or nothing
	# column oriented, ie, each column is one item in `tags_vocab` and rows are the chunks
	tags::T3 = nothing
	tags_vocab::Union{Nothing, Vector{<:AbstractString}} = nothing
	sources::Vector{<:AbstractString}
	extras::T4 = nothing
end

const ChunkIndex = ChunkEmbeddingsIndex # for backward compatibility

embeddings(index::ChunkEmbeddingsIndex) = index.embeddings
HasEmbeddings(::ChunkEmbeddingsIndex) = true
chunkdata(index::ChunkEmbeddingsIndex) = embeddings(index)
# It's column aligned so we don't have to re-define `chunkdata(index, chunk_idx)`


"""
	ChunkKeywordsIndex

Struct for storing chunks of text and associated keywords for BM25 similarity search.

# Fields
- `id::Symbol`: unique identifier of each index (to ensure we're using the right 
    index with `CandidateChunks`)
- `chunks::Vector{<:AbstractString}`: underlying document chunks / snippets
- `chunkdata::Union{Nothing, AbstractMatrix{<:Real}}`: for similarity search, 
    assumed to be `DocumentTermMatrix`
- `tags::Union{Nothing, AbstractMatrix{<:Bool}}`: for exact search, filtering, etc. 
    This is often a sparse matrix indicating which chunks have the given `tag` 
    (see `tag_vocab` for the position lookup)
- `tags_vocab::Union{Nothing, Vector{<:AbstractString}}`: vocabulary for the `tags` 
    matrix (each column in `tags` is one item in `tags_vocab` and rows are the chunks)
- `sources::Vector{<:AbstractString}`: sources of the chunks
- `extras::Union{Nothing, AbstractVector}`: additional data, eg, metadata, source code, etc.

# Example

We can easily create a keywords-based index from a standard embeddings-based index.

```julia

# Let's assume we have a standard embeddings-based index
index = build_index(SimpleIndexer(), texts; chunker_kwargs = (; max_length=10))

# Creating an additional index for keyword-based search (BM25), is as simple as
index_keywords = ChunkKeywordsIndex(index)

# We can immediately create a MultiIndex (a hybrid index holding both indices)
multi_index = MultiIndex([index, index_keywords])

```

You can also build the index via build_index
```julia
# given some sentences and sources
index_keywords = build_index(KeywordsIndexer(), sentences; chunker_kwargs=(; sources))

# Retrive closest chunks with
retriever = SimpleBM25Retriever()
result = retrieve(retriever, index_keywords, "What are the best practices for parallel computing in Julia?")
result.context
```

If you want to use airag, don't forget to specify the config to make sure keywords 
are processed (ie, tokenized) and that BM25 is used for searching candidates
```julia
cfg = RAGConfig(; retriever = SimpleBM25Retriever());
airag(cfg, index_keywords;
	question = "What are the best practices for parallel computing in Julia?")
```
"""
@kwdef struct ChunkKeywordsIndex{
	T1 <: AbstractString,
	T2 <: Union{Nothing, DocumentTermMatrix},
	T3 <: Union{Nothing, AbstractMatrix{<:Bool}},
	T4 <: Union{Nothing, AbstractVector},
} <: AbstractChunkIndex
	id::Symbol = gensym("ChunkKeywordsIndex")
	# underlying document chunks / snippets
	chunks::Vector{T1}
	# for similarity search
	chunkdata::T2 = nothing
	# for exact search, filtering, etc.
	# expected to be some sparse structure, eg, sparse matrix or nothing
	# column oriented, ie, each column is one item in `tags_vocab` and rows are the chunks
	tags::T3 = nothing
	tags_vocab::Union{Nothing, Vector{<:AbstractString}} = nothing
	sources::Vector{<:AbstractString}
	extras::T4 = nothing
end

HasKeywords(::ChunkKeywordsIndex) = true


""" 
    chunkdata(index::ChunkKeywordsIndex, chunk_idx::AbstractVector{<:Integer})

Access chunkdata for a subset of chunks.
 
# Arguments 
- `index::ChunkKeywordsIndex`: the index to access
- `chunk_idx::AbstractVector{<:Integer}`: the indices of the chunks to access
"""
Base.@propagate_inbounds function chunkdata(
	index::ChunkKeywordsIndex, chunk_idx::AbstractVector{<:Integer},
)
	chkdata = index.chunkdata
	if isnothing(chkdata)
		return nothing
	end
	## Keyword index is row-oriented, ie, chunks are rows, tokens are columns 
	return view(chkdata, chunk_idx, :)
end


"""
	SubChunkIndex

A view of the parent index with respect to the `chunks` (and chunk-aligned fields). 
All methods and accessors working for `AbstractChunkIndex` also work for `SubChunkIndex`.
It does not yet work for `MultiIndex`.

# Fields
- `parent::AbstractChunkIndex`: the parent index from which the chunks are drawn 
    (always the original index, never a view)
- `positions::Vector{Int}`: the positions of the chunks in the parent index (always 
    refers to original PARENT index, even if we create a view of the view)

# Example
```julia
cc = CandidateChunks(index.id, 1:10)
sub_index = @view(index[cc])
```

You can use `SubChunkIndex` to access chunks or sources (and other fields) from a parent index, eg,
```julia
RT.chunks(sub_index)
RT.sources(sub_index)
RT.chunkdata(sub_index) # slice of embeddings
RT.embeddings(sub_index) # slice of embeddings
RT.tags(sub_index) # slice of tags
RT.tags_vocab(sub_index) # unchanged, identical to parent version
RT.extras(sub_index) # slice of extras
```

Access the parent index that the `positions` correspond to
```julia
parent(sub_index)
RT.positions(sub_index)
```
"""
@kwdef struct SubChunkIndex{T <: AbstractChunkIndex} <: AbstractChunkIndex
	parent::T
	positions::Vector{Int}
end

Base.@propagate_inbounds function SubChunkIndex(index::SubChunkIndex, cc::CandidateChunks)
	pos = indexid(index) == indexid(cc) ? positions(cc) : Int[]
	intersect_pos = intersect(pos, positions(index))
	@boundscheck let chk_vector = chunks(parent(index))
		if !checkbounds(Bool, axes(chk_vector, 1), intersect_pos)
			## Avoid printing huge position arrays, show the extremas of the attempted range
			max_pos = extrema(intersect_pos)
			throw(BoundsError(chk_vector, max_pos))
		end
	end
	return SubChunkIndex(parent(index), intersect_pos)
end

Base.@propagate_inbounds function SubChunkIndex(
	index::SubChunkIndex,
	cc::MultiCandidateChunks,
)
	valid_items = findall(==(indexid(index)), indexids(cc))
	valid_positions = positions(cc)[valid_items]
	intersect_pos = intersect(valid_positions, positions(index))
	@boundscheck let chk_vector = chunks(parent(index))
		if !checkbounds(Bool, axes(chk_vector, 1), intersect_pos)
			## Avoid printing huge position arrays, show the extremas of the attempted range
			max_pos = extrema(intersect_pos)
			throw(BoundsError(chk_vector, max_pos))
		end
	end
	return SubChunkIndex(parent(index), intersect_pos)
end

indexid(index::SubChunkIndex) = parent(index) |> indexid
positions(index::SubChunkIndex) = index.positions
Base.parent(index::SubChunkIndex) = index.parent
HasEmbeddings(index::SubChunkIndex) = HasEmbeddings(parent(index))
HasKeywords(index::SubChunkIndex) = HasKeywords(parent(index))

Base.@propagate_inbounds function chunks(index::SubChunkIndex)
	view(chunks(parent(index)), positions(index))
end

Base.@propagate_inbounds function sources(index::SubChunkIndex)
	view(sources(parent(index)), positions(index))
end

Base.@propagate_inbounds function chunkdata(index::SubChunkIndex)
	chunkdata(parent(index), positions(index))
end
"Access chunkdata for a subset of chunks, `chunk_idx` is a vector of chunk indices in the index"
Base.@propagate_inbounds function chunkdata(
	index::SubChunkIndex, chunk_idx::AbstractVector{<:Integer})
	## We need this accessor because different chunk indices can have chunks in different dimensions!!
	index_chunk_idx = translate_positions_to_parent(index, chunk_idx)
	pos = intersect(positions(index), index_chunk_idx)
	chkdata = chunkdata(parent(index), pos)
end

function embeddings(index::SubChunkIndex)
	if HasEmbeddings(index)
		view(embeddings(parent(index)), :, positions(index))
	else
		throw(ArgumentError("`embeddings` not implemented for $(typeof(index))"))
	end
end

function tags(index::SubChunkIndex)
	tagsdata = tags(parent(index))
	isnothing(tagsdata) && return nothing
	view(tagsdata, positions(index), :)
end

function tags_vocab(index::SubChunkIndex)
	tags_vocab(parent(index))
end

function extras(index::SubChunkIndex)
	extrasdata = extras(parent(index))
	isnothing(extrasdata) && return nothing
	view(extrasdata, positions(index))
end

function Base.vcat(i1::SubChunkIndex, i2::SubChunkIndex)
	throw(ArgumentError("vcat not implemented for type $(typeof(i1)) and $(typeof(i2))"))
end

function Base.vcat(i1::T, i2::T) where {T <: SubChunkIndex}
	## Check if can be merged
	if indexid(parent(i1)) != indexid(parent(i2))
		throw(ArgumentError("Parent indices must be the same (provided: $(indexid(parent(i1))) and $(indexid(parent(i2))))"))
	end
	return SubChunkIndex(parent(i1), vcat(positions(i1), positions(i2)))
end

function Base.unique(index::SubChunkIndex)
	return SubChunkIndex(parent(index), unique(positions(index)))
end

function Base.length(index::SubChunkIndex)
	return length(positions(index))
end

function Base.isempty(index::SubChunkIndex)
	return isempty(positions(index))
end

function Base.show(io::IO, index::SubChunkIndex)
	print(io,
		"A view of $(typeof(parent(index))|>nameof) (id: $(indexid(parent(index)))) with $(length(index)) chunks")
end

"""
	translate_positions_to_parent(
		index::SubChunkIndex, pos::AbstractVector{<:Integer})

Translate positions to the parent index. Useful to convert between positions in a view and the original index.

Used whenever a `chunkdata()` or `tags()` are used to re-align positions to the "parent" index.
"""
Base.@propagate_inbounds function translate_positions_to_parent(
	index::SubChunkIndex, pos::AbstractVector{<:Integer},
)
	sub_positions = positions(index)
	return sub_positions[pos]
end


"""
	MultiIndex

Composite index that stores multiple ChunkIndex objects and their embeddings.

# Fields
- `id::Symbol`: unique identifier of each index (to ensure we're using the right index with `CandidateChunks`)
- `indexes::Vector{<:AbstractChunkIndex}`: the indexes to be combined

Use accesor `indexes` to access the individual indexes.

# Examples

We can create a `MultiIndex` from a vector of `AbstractChunkIndex` objects.
```julia
index = build_index(SimpleIndexer(), texts; chunker_kwargs = (; sources))
index_keywords = ChunkKeywordsIndex(index) # same chunks as above but adds BM25 instead of embeddings

multi_index = MultiIndex([index, index_keywords])
```

To use `airag` with different types of indices, we need to specify how to find the closest items for each index
```julia
# Cosine similarity for embeddings and BM25 for keywords, same order as indexes in MultiIndex
finder = RT.MultiFinder([RT.CosineSimilarity(), RT.BM25Similarity()])

# Notice that we add `processor` to make sure keywords are processed (ie, tokenized) as well
cfg = RAGConfig(; retriever = SimpleRetriever(; processor = RT.KeywordsProcessor(), finder))

# Ask questions
msg = airag(cfg, multi_index; question = "What are the best practices for parallel computing in Julia?")
pprint(msg) # prettify the answer
```

"""
@kwdef struct MultiIndex <: AbstractMultiIndex
	id::Symbol = gensym("MultiIndex")
	indexes::Vector{<:AbstractChunkIndex} = AbstractChunkIndex[]
end

function MultiIndex(indexes::AbstractChunkIndex...)
	MultiIndex(; indexes = collect(indexes))
end

function MultiIndex(indexes::AbstractVector{<:AbstractChunkIndex})
	MultiIndex(; indexes = indexes)
end

indexes(index::MultiIndex) = index.indexes
HasEmbeddings(index::AbstractMultiIndex) = any(HasEmbeddings, indexes(index))
HasKeywords(index::AbstractMultiIndex) = any(HasKeywords, indexes(index))

"""
	Base.var"=="(i1::MultiIndex, i2::MultiIndex)

Check that each index has a counterpart in the other MultiIndex.
"""
function Base.var"=="(i1::MultiIndex, i2::MultiIndex)
	length(indexes(i1)) != length(indexes(i2)) && return false
	for i in i1.indexes
		if !(i in i2.indexes)
			return false
		end
	end
	for i in i2.indexes
		if !(i in i1.indexes)
			return false
		end
	end
	return true
end
