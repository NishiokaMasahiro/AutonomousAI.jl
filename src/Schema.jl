"""
    Schema

Closed, typed action space for the LLM -> runtime boundary (spec sections 37, 38).

Design rule enforced here: **the LLM never emits executable text.**  It emits a JSON
object that must parse into one of a finite set of `Action` structs whose fields are
drawn from finite allowlists.  Anything else is a schema violation and is returned to
the planner as structured feedback, never executed.

This makes the trusted computing base of the "LLM decides" path equal to
`parse_action` plus the allowlists, instead of "the whole of Julia".
"""
module Schema

using ..MiniJSON

export Action, InspectHardware, ProposeCandidates, GenerateCode, BenchmarkAlgorithm,
       TuneParameter, VerifyResult, ExecuteFinal, RequestPermission, Abort,
       Candidate, Plan, SchemaError,
       parse_action, parse_plan, action_name, to_dict,
       BACKENDS, PRECISIONS, TRANSFORMS, ALGORITHMS, register_algorithm!,
       precision_type, precision_eps, is_gpu_backend

# ---------------------------------------------------------------- allowlists --

"Execution backends the planner may name."
const BACKENDS = Set{Symbol}([:cpu_serial, :cpu_simd, :cpu_threads, :cpu_blas,
                              :cuda, :distributed])

"Numeric formats the planner may name."
const PRECISIONS = Set{Symbol}([:Float64, :Float32, :Float16, :BFloat16])

"AST transformations the optimiser is allowed to apply (spec section 21)."
const TRANSFORMS = Set{Symbol}([:inbounds, :simd, :views, :inplace, :fma,
                                :fastmath, :tile, :hoist])

"""
Algorithm allowlist.  Populated by `Compute` at load time via `register_algorithm!`
so that the schema can never name an algorithm with no verified implementation.
"""
const ALGORITHMS = Set{Symbol}()

register_algorithm!(name::Symbol) = (push!(ALGORITHMS, name); name)

is_gpu_backend(b::Symbol) = b === :cuda

"""
    precision_type(::Symbol) -> Type

`:BFloat16` maps to `Float32` unless BFloat16s.jl is loaded; the caller must check
`hardware supports bf16` before selecting it.  We keep the symbol distinct from
`:Float32` so that memory/perf models are not silently wrong.
"""
function precision_type(p::Symbol)
    p === :Float64 && return Float64
    p === :Float32 && return Float32
    p === :Float16 && return Float16
    p === :BFloat16 && return Float32   # placeholder; see docs/05_limitations.md
    throw(SchemaError("unknown precision $(p)"))
end

precision_eps(p::Symbol) = p === :Float64 ? 2.22e-16 :
                           p === :Float32 ? 1.19e-7 :
                           p === :Float16 ? 9.77e-4 :
                           p === :BFloat16 ? 7.81e-3 :
                           throw(SchemaError("unknown precision $(p)"))

struct SchemaError <: Exception
    msg::String
end
Base.showerror(io::IO, e::SchemaError) = print(io, "SchemaError: ", e.msg)

# ------------------------------------------------------------------- actions --

abstract type Action end

"A concrete (algorithm, backend, precision, params) point in the search space."
struct Candidate
    algorithm::Symbol
    backend::Symbol
    precision::Symbol
    transforms::Vector{Symbol}
    params::Dict{String,Int}
end

Candidate(a::Symbol, b::Symbol, p::Symbol) = Candidate(a, b, p, Symbol[], Dict{String,Int}())

Base.:(==)(x::Candidate, y::Candidate) = x.algorithm == y.algorithm &&
    x.backend == y.backend && x.precision == y.precision &&
    sort(x.transforms) == sort(y.transforms) && x.params == y.params
Base.hash(c::Candidate, h::UInt) = hash((c.algorithm, c.backend, c.precision,
                                         sort(c.transforms), sort(collect(c.params))), h)

struct InspectHardware <: Action end

struct ProposeCandidates <: Action
    candidates::Vector{Candidate}
    rationale::String
end

struct GenerateCode <: Action
    candidate::Candidate
end

struct BenchmarkAlgorithm <: Action
    candidate::Candidate
    input_size::Vector{Int}
    samples::Int
end

struct TuneParameter <: Action
    candidate::Candidate
    name::String
    grid::Vector{Int}
    budget::Int
end

struct VerifyResult <: Action
    candidate::Candidate
    reference_backend::Symbol
    reference_precision::Symbol
    rtol::Float64
    atol::Float64
end

struct ExecuteFinal <: Action
    candidate::Candidate
    input_size::Vector{Int}
end

struct RequestPermission <: Action
    capability::Symbol
    reason::String
end

struct Abort <: Action
    reason::String
end

action_name(::InspectHardware)   = "inspect_hardware"
action_name(::ProposeCandidates) = "propose_candidates"
action_name(::GenerateCode)      = "generate_code"
action_name(::BenchmarkAlgorithm)= "benchmark_algorithm"
action_name(::TuneParameter)     = "tune_parameter"
action_name(::VerifyResult)      = "verify_result"
action_name(::ExecuteFinal)      = "execute_final"
action_name(::RequestPermission) = "request_permission"
action_name(::Abort)             = "abort"

"An ordered, validated action sequence produced by the reasoning layer."
struct Plan
    id::String
    goal_id::String
    steps::Vector{Action}
    rationale::String
end

# ------------------------------------------------------------- deserialising --

req(d::AbstractDict, k::String) = haskey(d, k) ? d[k] :
    throw(SchemaError("missing required field '$(k)'"))

function as_symbol(x, allowed::Set{Symbol}, field::String)
    x isa AbstractString || throw(SchemaError("field '$(field)' must be a string"))
    s = Symbol(x)
    s in allowed || throw(SchemaError("field '$(field)' value '$(x)' not in allowlist"))
    return s
end

function as_int_vector(x, field::String)
    x isa AbstractVector || throw(SchemaError("field '$(field)' must be an array"))
    out = Int[]
    for v in x
        v isa Integer || throw(SchemaError("field '$(field)' must contain integers"))
        v > 0 || throw(SchemaError("field '$(field)' must be positive"))
        push!(out, Int(v))
    end
    isempty(out) && throw(SchemaError("field '$(field)' must be non-empty"))
    return out
end

function as_string(x, field::String; maxlen::Int = 2000)
    x isa AbstractString || throw(SchemaError("field '$(field)' must be a string"))
    length(x) <= maxlen || throw(SchemaError("field '$(field)' too long"))
    return String(x)
end

function check_keys(d::AbstractDict, allowed::Vector{String})
    for k in keys(d)
        k in allowed || throw(SchemaError("unexpected field '$(k)'"))
    end
    return nothing
end

function parse_candidate(x)
    x isa AbstractDict || throw(SchemaError("candidate must be an object"))
    check_keys(x, ["algorithm", "backend", "precision", "transforms", "params"])
    alg = as_symbol(req(x, "algorithm"), ALGORITHMS, "algorithm")
    bk  = as_symbol(req(x, "backend"), BACKENDS, "backend")
    pr  = as_symbol(req(x, "precision"), PRECISIONS, "precision")
    tr = Symbol[]
    if haskey(x, "transforms")
        v = x["transforms"]
        v isa AbstractVector || throw(SchemaError("'transforms' must be an array"))
        length(v) <= 8 || throw(SchemaError("too many transforms"))
        for t in v
            push!(tr, as_symbol(t, TRANSFORMS, "transforms"))
        end
    end
    params = Dict{String,Int}()
    if haskey(x, "params")
        v = x["params"]
        v isa AbstractDict || throw(SchemaError("'params' must be an object"))
        length(v) <= 16 || throw(SchemaError("too many params"))
        for (k, pv) in v
            pv isa Integer || throw(SchemaError("param '$(k)' must be an integer"))
            params[String(k)] = Int(pv)
        end
    end
    return Candidate(alg, bk, pr, tr, params)
end

"""
    parse_action(obj) -> Action

`obj` is a `Dict{String,Any}` (already JSON-decoded) or a JSON string.  Throws
`SchemaError` on any deviation; callers must convert that into planner feedback
rather than into an execution attempt.
"""
parse_action(s::AbstractString) = parse_action(parse_json(s))

function parse_action(d::AbstractDict)
    a = req(d, "action")
    a isa AbstractString || throw(SchemaError("'action' must be a string"))
    if a == "inspect_hardware"
        check_keys(d, ["action"])
        return InspectHardware()
    elseif a == "propose_candidates"
        check_keys(d, ["action", "candidates", "rationale"])
        cs = req(d, "candidates")
        cs isa AbstractVector || throw(SchemaError("'candidates' must be an array"))
        (1 <= length(cs) <= 16) || throw(SchemaError("candidates must number 1..16"))
        return ProposeCandidates([parse_candidate(c) for c in cs],
                                 as_string(get(d, "rationale", ""), "rationale"))
    elseif a == "generate_code"
        check_keys(d, ["action", "candidate"])
        return GenerateCode(parse_candidate(req(d, "candidate")))
    elseif a == "benchmark_algorithm"
        check_keys(d, ["action", "candidate", "input_size", "samples"])
        n = get(d, "samples", 7)
        n isa Integer || throw(SchemaError("'samples' must be an integer"))
        (1 <= n <= 1000) || throw(SchemaError("'samples' out of range"))
        return BenchmarkAlgorithm(parse_candidate(req(d, "candidate")),
                                  as_int_vector(req(d, "input_size"), "input_size"),
                                  Int(n))
    elseif a == "tune_parameter"
        check_keys(d, ["action", "candidate", "name", "grid", "budget"])
        b = get(d, "budget", 8)
        b isa Integer || throw(SchemaError("'budget' must be an integer"))
        (1 <= b <= 64) || throw(SchemaError("'budget' out of range"))
        return TuneParameter(parse_candidate(req(d, "candidate")),
                             as_string(req(d, "name"), "name"; maxlen = 64),
                             as_int_vector(req(d, "grid"), "grid"), Int(b))
    elseif a == "verify_result"
        check_keys(d, ["action", "candidate", "reference_backend",
                       "reference_precision", "rtol", "atol"])
        rtol = Float64(get(d, "rtol", 1e-10))
        atol = Float64(get(d, "atol", 0.0))
        (0 < rtol < 1) || throw(SchemaError("'rtol' out of range"))
        (atol >= 0) || throw(SchemaError("'atol' must be >= 0"))
        return VerifyResult(parse_candidate(req(d, "candidate")),
                            as_symbol(get(d, "reference_backend", "cpu_serial"),
                                      BACKENDS, "reference_backend"),
                            as_symbol(get(d, "reference_precision", "Float64"),
                                      PRECISIONS, "reference_precision"), rtol, atol)
    elseif a == "execute_final"
        check_keys(d, ["action", "candidate", "input_size"])
        return ExecuteFinal(parse_candidate(req(d, "candidate")),
                            as_int_vector(req(d, "input_size"), "input_size"))
    elseif a == "request_permission"
        check_keys(d, ["action", "capability", "reason"])
        return RequestPermission(Symbol(as_string(req(d, "capability"), "capability";
                                                  maxlen = 64)),
                                 as_string(req(d, "reason"), "reason"))
    elseif a == "abort"
        check_keys(d, ["action", "reason"])
        return Abort(as_string(req(d, "reason"), "reason"))
    else
        throw(SchemaError("unknown action '$(a)'"))
    end
end

"""
    parse_plan(json, goal_id) -> Plan

Parses `{"plan_id":..., "rationale":..., "steps":[<action>...]}`.
"""
function parse_plan(s::AbstractString, goal_id::AbstractString)
    d = parse_json(s)
    d isa AbstractDict || throw(SchemaError("plan must be an object"))
    check_keys(d, ["plan_id", "rationale", "steps"])
    steps = req(d, "steps")
    steps isa AbstractVector || throw(SchemaError("'steps' must be an array"))
    (1 <= length(steps) <= 32) || throw(SchemaError("plan must have 1..32 steps"))
    return Plan(as_string(get(d, "plan_id", "plan"), "plan_id"; maxlen = 64),
                String(goal_id), Action[parse_action(s2) for s2 in steps],
                as_string(get(d, "rationale", ""), "rationale"))
end

# --------------------------------------------------------------- serialising --

to_dict(c::Candidate) = Dict{String,Any}(
    "algorithm" => String(c.algorithm), "backend" => String(c.backend),
    "precision" => String(c.precision),
    "transforms" => String[String(t) for t in c.transforms], "params" => c.params)

to_dict(a::InspectHardware) = Dict{String,Any}("action" => action_name(a))
to_dict(a::ProposeCandidates) = Dict{String,Any}("action" => action_name(a),
    "candidates" => [to_dict(c) for c in a.candidates], "rationale" => a.rationale)
to_dict(a::GenerateCode) = Dict{String,Any}("action" => action_name(a),
    "candidate" => to_dict(a.candidate))
to_dict(a::BenchmarkAlgorithm) = Dict{String,Any}("action" => action_name(a),
    "candidate" => to_dict(a.candidate), "input_size" => a.input_size,
    "samples" => a.samples)
to_dict(a::TuneParameter) = Dict{String,Any}("action" => action_name(a),
    "candidate" => to_dict(a.candidate), "name" => a.name, "grid" => a.grid,
    "budget" => a.budget)
to_dict(a::VerifyResult) = Dict{String,Any}("action" => action_name(a),
    "candidate" => to_dict(a.candidate),
    "reference_backend" => String(a.reference_backend),
    "reference_precision" => String(a.reference_precision),
    "rtol" => a.rtol, "atol" => a.atol)
to_dict(a::ExecuteFinal) = Dict{String,Any}("action" => action_name(a),
    "candidate" => to_dict(a.candidate), "input_size" => a.input_size)
to_dict(a::RequestPermission) = Dict{String,Any}("action" => action_name(a),
    "capability" => String(a.capability), "reason" => a.reason)
to_dict(a::Abort) = Dict{String,Any}("action" => action_name(a), "reason" => a.reason)

to_dict(p::Plan) = Dict{String,Any}("plan_id" => p.id, "rationale" => p.rationale,
    "steps" => [to_dict(s) for s in p.steps])

Base.show(io::IO, c::Candidate) = print(io, c.algorithm, "/", c.backend, "/",
    c.precision, isempty(c.transforms) ? "" : string(" +", join(c.transforms, "+")))

end # module
