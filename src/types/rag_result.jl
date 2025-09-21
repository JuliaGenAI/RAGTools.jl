"""
	This file contains the definition of the types:
	- `RAGResult`
	and related functions.
"""

"""
	RAGResult

A struct for debugging RAG answers. It contains the question, answer, context, 
and the candidate chunks at each step of the RAG pipeline.

Think of the flow as `question` -> `rephrased_questions` -> `answer` -> `final_answer` 
with the context and candidate chunks helping along the way.

# Fields
- `question::AbstractString`: the original question
- `rephrased_questions::Vector{<:AbstractString}`: a vector of rephrased questions (eg, HyDe, Multihop, etc.)
- `answer::AbstractString`: the generated answer
- `final_answer::AbstractString`: the refined final answer (eg, after CorrectiveRAG), 
	also considered the FINAL answer (it must be always available)
- `context::Vector{<:AbstractString}`: the context used for retrieval (ie, the vector 
	of chunks and their surrounding window if applicable)
- `sources::Vector{<:AbstractString}`: the sources of the context (for the original matched chunks)
- `emb_candidates::CandidateChunks`: the candidate chunks from the embedding index (from `find_closest`)
- `tag_candidates::Union{Nothing, CandidateChunks}`: the candidate chunks from the tag index (from `find_tags`)
- `filtered_candidates::CandidateChunks`: the filtered candidate chunks (intersection 
	of `emb_candidates` and `tag_candidates`)
- `reranked_candidates::CandidateChunks`: the reranked candidate chunks (from `rerank`)
- `conversations::Dict{Symbol,Vector{<:AbstractMessage}}`: the conversation history for 
	AI steps of the RAG pipeline, use keys that correspond to the function names, eg, `:answer` or `:refine`

See also: `pprint` (pretty printing), `annotate_support` (for annotating the answer)
"""
@kwdef mutable struct RAGResult <: AbstractRAGResult
    question::AbstractString
    rephrased_questions::AbstractVector{<:AbstractString} = [question]
    answer::Union{Nothing, AbstractString} = nothing
    final_answer::Union{Nothing, AbstractString} = nothing
    context::Vector{<:AbstractString} = String[]
    sources::Vector{<:AbstractString} = String[]
    emb_candidates::Union{CandidateChunks,
        MultiCandidateChunks} = CandidateChunks(
        index_id = :NOTINDEX, positions = Int[], scores = Float32[]
    )
    tag_candidates::Union{Nothing,
        CandidateChunks,
        MultiCandidateChunks} = CandidateChunks(
        index_id = :NOTINDEX, positions = Int[], scores = Float32[]
    )
    filtered_candidates::Union{CandidateChunks,
        MultiCandidateChunks} = CandidateChunks(
        index_id = :NOTINDEX, positions = Int[], scores = Float32[]
    )
    reranked_candidates::Union{CandidateChunks,
        MultiCandidateChunks} = CandidateChunks(
        index_id = :NOTINDEX, positions = Int[], scores = Float32[]
    )
    conversations::Dict{
        Symbol, Vector{<:AbstractMessage}} = Dict{Symbol, Vector{<:AbstractMessage}}()
end

"""
	Base.var"=="(r1::T, r2::T) where {T <: AbstractRAGResult}

Two RAGResult objects are equal if all their fields are equal.
"""
function Base.var"=="(r1::T, r2::T) where {T <: AbstractRAGResult}
    all(f -> getfield(r1, f) == getfield(r2, f), fieldnames(T))
end

"""
	Base.copy(r::T) where {T <: AbstractRAGResult}

Copy a RAGResult object by deep copying all its fields.
"""
function Base.copy(r::T) where {T <: AbstractRAGResult}
    T(map(f -> deepcopy(getfield(r, f)), fieldnames(T))...)
end

"""
	Base.show(io::IO,
		t::Union{AbstractDocumentIndex, AbstractCandidateChunks, AbstractRAGResult})

Structured show method for easier reading (each kwarg on a new line)
"""
function Base.show(io::IO,
        t::Union{AbstractDocumentIndex, AbstractCandidateChunks, AbstractRAGResult})
    dump(IOContext(io, :limit => true), t, maxdepth = 1)
end

"""
	PT.last_message(result::RAGResult)

Extract the last message from the RAGResult for consistency with AICall / Message vectors. 
It looks for `final_answer` first, then `answer` fields in the `conversations` dictionary. 
Returns `nothing` if not found.
"""
function PT.last_message(result::RAGResult)
    (; conversations) = result
    if haskey(conversations, :final_answer) &&
       !isempty(conversations[:final_answer])
        conversations[:final_answer][end]
    elseif haskey(conversations, :answer) &&
           !isempty(conversations[:answer])
        conversations[:answer][end]
    else
        nothing
    end
end

""" 
	PT.last_output(result::RAGResult)

Extracts the last output (generated text answer) from the RAGResult for consistency 
with AICall / Message vectors.

See also: `PT.last_message`
"""
function PT.last_output(result::RAGResult)
    msg = PT.last_message(result)
    isnothing(msg) ? result.final_answer : msg.content
end

# TODO: add more customizations, eg, context itself
"""
	PT.pprint(
		io::IO, r::AbstractRAGResult; add_context::Bool = false,
		text_width::Int = displaysize(io)[2], annotater_kwargs...)

Pretty print the RAG result `r` to the given `io` stream. 

If `add_context` is `true`, the context will be printed as well. The `text_width` 
parameter can be used to control the width of the output.

You can provide additional keyword arguments to the annotater, eg, `add_sources`, 
`add_scores`, `min_score`, etc. See `annotate_support` for more details.
"""
function PT.pprint(
        io::IO, r::AbstractRAGResult;
        add_context::Bool = false,
        text_width::Int = displaysize(io)[2],
        annotater_kwargs...
)
    if !isempty(r.rephrased_questions)
        content = PT.wrap_string("- " * join(r.rephrased_questions, "\n- "), text_width)
        print(io, "-"^20, "\n")
        printstyled(io, "QUESTION(s)", color = :blue, bold = true)
        print(io, "\n", "-"^20, "\n")
        print(io, content, "\n\n")
    end

    if !isnothing(r.final_answer) && !isempty(r.final_answer)
        annotater = TrigramAnnotater()
        root = annotate_support(annotater, r; annotater_kwargs...)
        print(io, "-"^20, "\n")
        printstyled(io, "ANSWER", color = :blue, bold = true)
        print(io, "\n", "-"^20, "\n")
        pprint(io, root; text_width)
    end

    if add_context && !isempty(r.context)
        print(io, "\n" * "-"^20, "\n")
        printstyled(io, "CONTEXT", color = :blue, bold = true)
        print(io, "\n", "-"^20, "\n")
        for (i, ctx) in enumerate(r.context)
            print(io, PT.wrap_string(ctx, text_width))
            print(io, "\n", "-"^20, "\n")
        end
    end
end

# Serialization for JSON3
# ------------------------
"""
	StructTypes.StructType(::Type{RAGResult})
"""
StructTypes.StructType(::Type{RAGResult}) = StructTypes.Struct()

"""
	StructTypes.constructfrom(RAGResult, JSON3.read(tmp)) 

Use as: `StructTypes.constructfrom(RAGResult, JSON3.read(tmp))`
"""
function StructTypes.constructfrom(::Type{RAGResult}, obj::Union{Dict, JSON3.Object})
    obj = copy(obj)
    if haskey(obj, :conversations)
        obj[:conversations] = Dict(k => StructTypes.constructfrom(
                                       Vector{PT.AbstractMessage}, v)
        for (k, v) in pairs(obj[:conversations]))
    end
    ## Retype where necessary
    for f in [
        :emb_candidates, :tag_candidates, :filtered_candidates, :reranked_candidates]
        ## Check for nothing value, because tag_candidates can be empty
        if haskey(obj, f) && !isnothing(obj[f]) && haskey(obj[f], :index_ids)
            obj[f] = StructTypes.constructfrom(MultiCandidateChunks, obj[f])
        elseif haskey(obj, f) && !isnothing(obj[f])
            obj[f] = StructTypes.constructfrom(CandidateChunks, obj[f])
        end
    end
    obj[:context] = convert(Vector{String}, get(obj, :context, String[]))
    obj[:sources] = convert(Vector{String}, get(obj, :sources, String[]))
    RAGResult(; obj...)
end

"""
	JSON3.read(path::AbstractString, ::Type{RAGResult})

Read a RAGResult object from a JSON file.
"""
function JSON3.read(path::AbstractString, ::Type{RAGResult})
    StructTypes.constructfrom(RAGResult, JSON3.read(path))
end
