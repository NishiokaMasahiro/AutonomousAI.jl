"""
    LLM

Provider abstraction for the reasoning layer (spec sections 36, 37).

The LLM occupies exactly one position in this system: it proposes a *plan* over the
closed action space in `Schema`, and it interprets failures.  It never produces a
number that is used as a result, never emits shell text, and never sees a device.

`MockLLM` is the default so that the whole loop runs offline and deterministically.
It is a heuristic stand-in, and is labelled as such wherever it appears in reports:
its plans are hand-written rules, not model output, and must not be presented as
evidence about what a real model would plan.
"""
module LLM

using Downloads, Dates, Printf
using ..MiniJSON
using ..Schema

export LLMBackend, MockLLM, AnthropicLLM, OpenAICompatibleLLM, complete,
       propose_plan, SYSTEM_PROMPT, backend_name, LLMResult

abstract type LLMBackend end

struct LLMResult
    text::String
    ok::Bool
    error::String
    latency_s::Float64
    backend::String
end

backend_name(::LLMBackend) = "unknown"

const SYSTEM_PROMPT = """
You are the planning module of a Julia computational-intelligence runtime.
You do not compute results and you do not write code. You emit exactly one JSON
object and nothing else:

{"plan_id": "...", "rationale": "...", "steps": [ <action>, ... ]}

Allowed actions (any deviation is rejected by the schema validator):
  {"action":"inspect_hardware"}
  {"action":"propose_candidates","candidates":[{"algorithm":..,"backend":..,
      "precision":..,"transforms":[..],"params":{..}}],"rationale":".."}
  {"action":"generate_code","candidate":{..}}
  {"action":"benchmark_algorithm","candidate":{..},"input_size":[..],"samples":N}
  {"action":"tune_parameter","candidate":{..},"name":"..","grid":[..],"budget":N}
  {"action":"verify_result","candidate":{..},"reference_backend":"cpu_serial",
      "reference_precision":"Float64","rtol":1e-6,"atol":0.0}
  {"action":"execute_final","candidate":{..},"input_size":[..]}
  {"action":"abort","reason":".."}

Constraints you must respect:
  - Only algorithms, backends, precisions and transforms listed in the context are
    permitted. Inventing one aborts the plan.
  - Every candidate you propose for execution must first be benchmarked and verified.
  - Prefer the cheapest evidence: a benchmark on a small size before a large one.
  - If the context says a resource is unavailable, do not plan around it; drop it.
"""

# ------------------------------------------------------------------ mock ----

"""
Deterministic offline planner.  Encodes the heuristics of spec section 9 directly:
size and arithmetic intensity choose the backend, precision starts at the algorithm's
`min_safe_precision`, and every plan ends with verification before execution.
"""
struct MockLLM <: LLMBackend
    verbose::Bool
end
MockLLM(; verbose::Bool = false) = MockLLM(verbose)
backend_name(::MockLLM) = "mock-heuristic"

function complete(m::MockLLM, system::AbstractString, user::AbstractString)
    t0 = time()
    return LLMResult(mock_plan_json(user), true, "", time() - t0, backend_name(m))
end

"Extracts the context fields the heuristic needs; unknown fields are ignored."
function mock_plan_json(user::AbstractString)
    ctx = try
        parse_json(user)
    catch
        Dict{String,Any}()
    end
    alg = Symbol(get(ctx, "algorithm", "zscore_anomaly"))
    n = Int(get(ctx, "n_elements", 1_000_000))
    backends = Symbol[Symbol(b) for b in get(ctx, "available_backends", ["cpu_serial"])]
    gpu_friendly = Bool(get(ctx, "gpu_friendly", false))
    size = Int[Int(x) for x in get(ctx, "input_size", Any[n])]
    small = n < 100_000
    prefer = if small
        :cpu_serial
    elseif gpu_friendly && :cuda in backends
        :cuda
    elseif :cpu_threads in backends
        :cpu_threads
    else
        :cpu_simd
    end
    prefer in backends || (prefer = first(backends))
    base = Dict{String,Any}("algorithm" => String(alg), "backend" => "cpu_serial",
                            "precision" => "Float64", "transforms" => String[],
                            "params" => Dict{String,Int}())
    opt = Dict{String,Any}("algorithm" => String(alg), "backend" => String(prefer),
                           "precision" => small ? "Float64" : "Float32",
                           "transforms" => ["inbounds", "simd"],
                           "params" => Dict{String,Int}())
    steps = Any[
        Dict{String,Any}("action" => "inspect_hardware"),
        Dict{String,Any}("action" => "propose_candidates",
                         "candidates" => Any[base, opt],
                         "rationale" => "baseline for reference, plus the " *
                             "size/intensity-selected candidate"),
        Dict{String,Any}("action" => "benchmark_algorithm", "candidate" => base,
                         "input_size" => size, "samples" => 7),
        Dict{String,Any}("action" => "generate_code", "candidate" => opt),
        Dict{String,Any}("action" => "benchmark_algorithm", "candidate" => opt,
                         "input_size" => size, "samples" => 7),
        Dict{String,Any}("action" => "verify_result", "candidate" => opt,
                         "reference_backend" => "cpu_serial",
                         "reference_precision" => "Float64",
                         "rtol" => 1.0e-6, "atol" => 0.0),
        Dict{String,Any}("action" => "execute_final", "candidate" => opt,
                         "input_size" => size),
    ]
    return to_json(Dict{String,Any}("plan_id" => "mock-" * string(hash(user), base = 16)[1:8],
                                    "rationale" => "heuristic plan (offline stand-in)",
                                    "steps" => steps))
end

# ------------------------------------------------------------- HTTP backends --

"""
Anthropic Messages API.  Uses the `Downloads` stdlib rather than HTTP.jl so the
core package keeps zero external dependencies; the key is read from the environment
and never logged.
"""
struct AnthropicLLM <: LLMBackend
    model::String
    max_tokens::Int
    api_key_env::String
    url::String
    timeout_s::Float64
end
AnthropicLLM(; model = "claude-sonnet-4-6", max_tokens = 2048,
             api_key_env = "ANTHROPIC_API_KEY",
             url = "https://api.anthropic.com/v1/messages", timeout_s = 60.0) =
    AnthropicLLM(model, max_tokens, api_key_env, url, timeout_s)
backend_name(b::AnthropicLLM) = "anthropic:" * b.model

function complete(b::AnthropicLLM, system::AbstractString, user::AbstractString)
    t0 = time()
    key = get(ENV, b.api_key_env, "")
    isempty(key) && return LLMResult("", false, "missing $(b.api_key_env)", 0.0,
                                     backend_name(b))
    body = to_json(Dict{String,Any}("model" => b.model, "max_tokens" => b.max_tokens,
        "system" => String(system),
        "messages" => Any[Dict{String,Any}("role" => "user", "content" => String(user))]))
    out = IOBuffer()
    try
        Downloads.request(b.url; method = "POST", input = IOBuffer(body), output = out,
                          headers = ["content-type" => "application/json",
                                     "x-api-key" => key,
                                     "anthropic-version" => "2023-06-01"],
                          timeout = b.timeout_s)
    catch err
        return LLMResult("", false, "transport error: $(err)", time() - t0, backend_name(b))
    end
    txt = String(take!(out))
    parsed = try
        parse_json(txt)
    catch err
        return LLMResult(txt, false, "unparseable response: $(err)", time() - t0,
                         backend_name(b))
    end
    content = get(parsed, "content", nothing)
    if content isa AbstractVector
        buf = IOBuffer()
        for blk in content
            blk isa AbstractDict && get(blk, "type", "") == "text" &&
                print(buf, get(blk, "text", ""))
        end
        return LLMResult(String(take!(buf)), true, "", time() - t0, backend_name(b))
    end
    return LLMResult(txt, false, "unexpected response shape", time() - t0, backend_name(b))
end

"OpenAI-compatible chat endpoint; also covers local servers (llama.cpp, vLLM, Ollama)."
struct OpenAICompatibleLLM <: LLMBackend
    model::String
    url::String
    api_key_env::String
    max_tokens::Int
    timeout_s::Float64
end
OpenAICompatibleLLM(; model = "local-model", url = "http://127.0.0.1:8080/v1/chat/completions",
                    api_key_env = "OPENAI_API_KEY", max_tokens = 2048, timeout_s = 120.0) =
    OpenAICompatibleLLM(model, url, api_key_env, max_tokens, timeout_s)
backend_name(b::OpenAICompatibleLLM) = "openai-compatible:" * b.model

function complete(b::OpenAICompatibleLLM, system::AbstractString, user::AbstractString)
    t0 = time()
    body = to_json(Dict{String,Any}("model" => b.model, "max_tokens" => b.max_tokens,
        "messages" => Any[Dict{String,Any}("role" => "system", "content" => String(system)),
                          Dict{String,Any}("role" => "user", "content" => String(user))]))
    out = IOBuffer()
    headers = ["content-type" => "application/json"]
    key = get(ENV, b.api_key_env, "")
    isempty(key) || push!(headers, "authorization" => "Bearer " * key)
    try
        Downloads.request(b.url; method = "POST", input = IOBuffer(body), output = out,
                          headers = headers, timeout = b.timeout_s)
    catch err
        return LLMResult("", false, "transport error: $(err)", time() - t0, backend_name(b))
    end
    parsed = try
        parse_json(String(take!(out)))
    catch err
        return LLMResult("", false, "unparseable response: $(err)", time() - t0,
                         backend_name(b))
    end
    ch = get(parsed, "choices", nothing)
    if ch isa AbstractVector && !isempty(ch)
        msg = get(ch[1], "message", Dict{String,Any}())
        return LLMResult(String(get(msg, "content", "")), true, "", time() - t0,
                         backend_name(b))
    end
    return LLMResult("", false, "unexpected response shape", time() - t0, backend_name(b))
end

# ------------------------------------------------------------ plan proposal --

"Strips ```json fences that models add despite instructions."
function extract_json(text::AbstractString)
    s = strip(String(text))
    if startswith(s, "```")
        i = findfirst('\n', s)
        i === nothing || (s = s[(i+1):end])
        j = findlast("```", s)
        j === nothing || (s = s[1:(first(j)-1)])
    end
    a = findfirst('{', s)
    b = findlast('}', s)
    (a === nothing || b === nothing) && return String(s)
    return String(s[a:b])
end

"""
    propose_plan(backend, context, goal_id; retries=2) -> (Plan|nothing, log)

Schema-repair loop: a rejected plan is returned to the model together with the
validator's message.  After `retries` the caller must fall back to the runtime's own
default plan -- the system never degrades into executing something unvalidated.
"""
function propose_plan(b::LLMBackend, context::AbstractDict, goal_id::AbstractString;
                      retries::Int = 2)
    log = String[]
    user = to_json(context)
    for attempt in 0:retries
        r = complete(b, SYSTEM_PROMPT, user)
        if !r.ok
            push!(log, "attempt $(attempt): transport/backend failure: $(r.error)")
            continue
        end
        try
            plan = Schema.parse_plan(extract_json(r.text), goal_id)
            push!(log, @sprintf("attempt %d: accepted (%d steps, %.2fs, %s)", attempt,
                                length(plan.steps), r.latency_s, r.backend))
            return (plan, log)
        catch err
            msg = err isa Schema.SchemaError ? err.msg : string(err)
            push!(log, "attempt $(attempt): schema rejection: $(msg)")
            user = to_json(Dict{String,Any}("previous_attempt_rejected" => msg,
                                            "context" => context))
        end
    end
    return (nothing, log)
end

end # module
