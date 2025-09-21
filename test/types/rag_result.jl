# PromptingTools Dependencies
# ----------------------------
using PromptingTools: last_message, last_output

# Local Dependencies
# ------------------
using RAGTools: RAGResult

# Tests 
# ------
@testset "RAGResult" begin
    result = RAGResult(; question = "a", answer = "b", final_answer = "c")
    result2 = RAGResult(; question = "a", answer = "b", final_answer = "c")
    @test result == result2

    result3 = copy(result)
    @test result == result3
    @test result !== result3

    ## pprint checks - empty context fails
    io = IOBuffer()
    @test_throws AssertionError PT.pprint(io, result)

    ## RAG Details dispatch
    answer = "This is a test answer."
    sources_ = ["Source 1", "Source 2", "Source 3"]
    result = RAGResult(;
        question = "?", final_answer = answer, context = sources_, sources = sources_)
    io = IOBuffer()
    PT.pprint(io, result; add_context = true)
    output = String(take!(io))
    @test occursin("This is a test answer.", output)
    @test occursin("\nQUESTION", output)
    @test occursin("\nSOURCES\n", output)
    @test occursin("\nCONTEXT\n", output)
    @test occursin("1. Source 1", output)

    ## last_message, last_output
    result = RAGResult(; question = "a", answer = "b", final_answer = "c")
    @test isnothing(last_message(result))
    @test last_output(result) == "c"

    result = RAGResult(; question = "a", answer = "b", final_answer = "c",
        conversations = Dict(:final_answer => [PT.UserMessage("c")]))
    @test last_message(result) == PT.UserMessage("c")
    @test last_output(result) == "c"

    result = RAGResult(; question = "a", answer = "b", final_answer = "c",
        conversations = Dict(:answer => [PT.UserMessage("a")]))
    @test last_message(result) == PT.UserMessage("a")

    # serialization
    # We cannot recover all type information !!!
    result = RAGResult(; question = "a", answer = "b", final_answer = "c",
        conversations = Dict(:answer => [PT.UserMessage("a")]))
    tmp, _ = mktemp()
    JSON3.write(tmp, result)
    resultx = JSON3.read(tmp, RAGResult)
    @test resultx == result
end
