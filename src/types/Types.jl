# Base Dependencies
# ------------------
using Base: parent
using StructTypes, JSON3

# Files inclusion 
# ---------------

include("chunk.jl")
include("document_matrix.jl")
include("index.jl") 
include("rag_result.jl")