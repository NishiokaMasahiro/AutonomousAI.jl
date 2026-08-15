"""
    Memory

Five separated memory systems (spec section 2.2).  The split is not cosmetic:
each has a different write policy, a different eviction rule and a different trust
level.

| store      | written by            | trusted?             | eviction        |
|------------|-----------------------|----------------------|-----------------|
| working    | current agent loop    | no (scratch)         | capacity ring   |
| episodic   | loop, append-only     | audit log            | never (archive) |
| semantic   | curated + measured    | yes (typed facts)    | manual          |
| procedural | validator + benchmark | only after passing   | by hash         |
| hardware   | benchmark harness     | yes, keyed by fp     | by fingerprint  |

The hardware store is keyed by `(algorithm, backend, precision, transforms,
hw_fingerprint, sw_fingerprint)`.  Dropping the software fingerprint would let a
timing recorded before a driver/Julia upgrade silently steer today's decisions.
"""
module Memory

using Dates, Serialization, Statistics, SHA
using ..MiniJSON
using ..Schema

export WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory,
       HardwareMemory, MemorySystem, BenchmarkRecord, Episode, CodeArtifact,
       AlgorithmFact, remember!, recall, forget!, record!, lookup, best_known,
       predict_runtime, save_memory, load_memory, register_fact!, fact,
       store_artifact!, artifact, n_records

# ------------------------------------------------------------ working memory --

"Bounded scratchpad for the active loop.  Bounded in *bytes*, not entries."
mutable struct WorkingMemory
    items::Vector{Pair{String,Any}}
    capacity_bytes::Int
    used_bytes::Int
end
WorkingMemory(; capacity_bytes::Int = 8 * 1024^2) =
    WorkingMemory(Pair{String,Any}[], capacity_bytes, 0)

approx_size(x) = try
    Base.summarysize(x)
catch
    64
end

function remember!(wm::WorkingMemory, key::AbstractString, value)
    forget!(wm, key)
    sz = approx_size(value)
    push!(wm.items, String(key) => value)
    wm.used_bytes += sz
    while wm.used_bytes > wm.capacity_bytes && length(wm.items) > 1
        old = popfirst!(wm.items)
        wm.used_bytes -= approx_size(old.second)
    end
    return value
end

function recall(wm::WorkingMemory, key::AbstractString, default = nothing)
    for p in Iterators.reverse(wm.items)
        p.first == key && return p.second
    end
    return default
end

function forget!(wm::WorkingMemory, key::AbstractString)
    idx = findfirst(p -> p.first == key, wm.items)
    idx === nothing && return false
    wm.used_bytes -= approx_size(wm.items[idx].second)
    deleteat!(wm.items, idx)
    return true
end

# ----------------------------------------------------------- episodic memory --

struct Episode
    id::String
    started::DateTime
    finished::DateTime
    goal::String
    plan_id::String
    n_iterations::Int
    success::Bool
    failure_reason::String
    final_candidate::String
    baseline_runtime_s::Float64
    final_runtime_s::Float64
    speedup::Float64
    accuracy_rel_error::Float64
    notes::String
end

mutable struct EpisodicMemory
    episodes::Vector{Episode}
end
EpisodicMemory() = EpisodicMemory(Episode[])
record!(em::EpisodicMemory, e::Episode) = (push!(em.episodes, e); e)

# ----------------------------------------------------------- semantic memory --

"""
Declarative facts about an algorithm.  `flop_coeff`/`flop_exponent` define
F(n) = flop_coeff * n^flop_exponent; `bytes_per_element` drives the roofline model.
"""
struct AlgorithmFact
    name::Symbol
    flop_coeff::Float64
    flop_exponent::Float64
    bytes_per_element::Float64
    arithmetic_intensity::Float64
    parallel::Bool
    gpu_friendly::Bool
    min_safe_precision::Symbol
    stability::Symbol            # :stable, :conditionally_stable, :unstable
    description::String
end

mutable struct SemanticMemory
    facts::Dict{Symbol,AlgorithmFact}
end
SemanticMemory() = SemanticMemory(Dict{Symbol,AlgorithmFact}())
register_fact!(sm::SemanticMemory, f::AlgorithmFact) = (sm.facts[f.name] = f)
fact(sm::SemanticMemory, name::Symbol) = get(sm.facts, name, nothing)

# --------------------------------------------------------- procedural memory --

"A generated implementation that has *passed* validation; keyed by content hash."
struct CodeArtifact
    hash::String
    candidate::Schema.Candidate
    source::String
    validated::Bool
    validation_notes::Vector{String}
    created::DateTime
    n_success::Int
    n_failure::Int
end

mutable struct ProceduralMemory
    artifacts::Dict{String,CodeArtifact}
end
ProceduralMemory() = ProceduralMemory(Dict{String,CodeArtifact}())

source_hash(src::AbstractString) = bytes2hex(sha256(String(src)))[1:24]

function store_artifact!(pm::ProceduralMemory, a::CodeArtifact)
    pm.artifacts[a.hash] = a
    return a
end
artifact(pm::ProceduralMemory, h::AbstractString) = get(pm.artifacts, String(h), nothing)

# ----------------------------------------------------------- hardware memory --

"One measured (candidate, size, machine) point.  This is the system's ground truth."
struct BenchmarkRecord
    algorithm::Symbol
    backend::Symbol
    precision::Symbol
    transforms::Vector{Symbol}
    n_elements::Int
    input_size::Vector{Int}
    compile_s::Float64
    runtime_min_s::Float64
    runtime_median_s::Float64
    runtime_mad_s::Float64
    samples::Int
    alloc_bytes::Int
    alloc_count::Int
    gflops::Float64
    bytes_moved::Int
    host_device_bytes::Int
    vram_peak_bytes::Int
    rel_error::Float64
    energy_j::Float64
    failed::Bool
    hw_fingerprint::String
    sw_fingerprint::String
    timestamp::DateTime
end

mutable struct HardwareMemory
    records::Vector{BenchmarkRecord}
    index::Dict{NTuple{6,Any},Vector{Int}}
end
HardwareMemory() = HardwareMemory(BenchmarkRecord[], Dict{NTuple{6,Any},Vector{Int}}())

key(r::BenchmarkRecord) = (r.algorithm, r.backend, r.precision,
                           Tuple(sort(r.transforms)), r.hw_fingerprint, r.sw_fingerprint)

function record!(hm::HardwareMemory, r::BenchmarkRecord)
    push!(hm.records, r)
    push!(get!(hm.index, key(r), Int[]), length(hm.records))
    return r
end

n_records(hm::HardwareMemory) = length(hm.records)

"""
    lookup(hm, candidate, n; hw, sw) -> Vector{BenchmarkRecord}

Exact-key retrieval.  Deliberately does *not* fall back to a different machine:
cross-machine transfer is handled explicitly by `predict_runtime`, which reports
that it is extrapolating.
"""
function lookup(hm::HardwareMemory, c::Schema.Candidate; hw::AbstractString,
                sw::AbstractString)
    k = (c.algorithm, c.backend, c.precision, Tuple(sort(c.transforms)),
         String(hw), String(sw))
    idxs = get(hm.index, k, Int[])
    return BenchmarkRecord[hm.records[i] for i in idxs if !hm.records[i].failed]
end

"""
    predict_runtime(hm, candidate, n; hw, sw) -> (t::Float64, conf::Float64)

Log-log least-squares fit of t = a * n^b over the recorded sizes for this exact
candidate.  Returns `(NaN, 0.0)` when there is no evidence -- the caller must then
fall back to the analytic roofline prior rather than pretend to know.

Confidence combines sample count and fit residual; it is fed to the acquisition
function in `Optimization`, so an over-confident extrapolation is penalised by the
next measurement rather than trusted forever.
"""
function predict_runtime(hm::HardwareMemory, c::Schema.Candidate, n::Integer;
                         hw::AbstractString, sw::AbstractString)
    rs = lookup(hm, c; hw = hw, sw = sw)
    isempty(rs) && return (NaN, 0.0)
    exact = filter(r -> r.n_elements == n, rs)
    if !isempty(exact)
        ts = [r.runtime_min_s for r in exact]
        conf = min(0.95, 0.5 + 0.1 * length(exact))
        return (median(ts), conf)
    end
    length(rs) < 2 && return (rs[1].runtime_min_s * n / max(rs[1].n_elements, 1), 0.25)
    x = [log(max(r.n_elements, 1)) for r in rs]
    y = [log(max(r.runtime_min_s, 1e-12)) for r in rs]
    xbar, ybar = mean(x), mean(y)
    sxx = sum((x .- xbar) .^ 2)
    sxx <= 0 && return (median(exp.(y)), 0.2)
    b = sum((x .- xbar) .* (y .- ybar)) / sxx
    a = ybar - b * xbar
    resid = y .- (a .+ b .* x)
    rms = sqrt(mean(resid .^ 2))
    lx = log(max(n, 1))
    extrapolating = lx < minimum(x) || lx > maximum(x)
    conf = clamp(exp(-rms) * (extrapolating ? 0.5 : 0.9) *
                 min(1.0, length(rs) / 5), 0.0, 0.9)
    return (exp(a + b * lx), conf)
end

"""
    best_known(hm, algorithm, n; hw, sw) -> Union{Nothing,BenchmarkRecord}
"""
function best_known(hm::HardwareMemory, algorithm::Symbol, n::Integer;
                    hw::AbstractString, sw::AbstractString)
    cands = [r for r in hm.records if r.algorithm == algorithm && !r.failed &&
             r.hw_fingerprint == hw && r.sw_fingerprint == sw && r.n_elements == n]
    isempty(cands) && return nothing
    return cands[argmin([r.runtime_min_s for r in cands])]
end

# ---------------------------------------------------------------- aggregate --

struct MemorySystem
    working::WorkingMemory
    episodic::EpisodicMemory
    semantic::SemanticMemory
    procedural::ProceduralMemory
    hardware::HardwareMemory
    dir::String
end

function MemorySystem(dir::AbstractString = joinpath(homedir(), ".autonomousai"))
    return MemorySystem(WorkingMemory(), EpisodicMemory(), SemanticMemory(),
                        ProceduralMemory(), HardwareMemory(), String(dir))
end

"""
    save_memory(ms)

Persists the durable stores.  Working memory is intentionally *not* persisted: it
is scratch state whose survival across runs would leak stale beliefs into planning.
"""
function save_memory(ms::MemorySystem)
    mkpath(ms.dir)
    serialize(joinpath(ms.dir, "episodic.jls"), ms.episodic)
    serialize(joinpath(ms.dir, "semantic.jls"), ms.semantic)
    serialize(joinpath(ms.dir, "procedural.jls"), ms.procedural)
    serialize(joinpath(ms.dir, "hardware.jls"), ms.hardware)
    open(joinpath(ms.dir, "hardware.json"), "w") do io
        write(io, to_json(Dict("records" => [record_dict(r) for r in ms.hardware.records]),
                          indent = 2))
    end
    return ms.dir
end

record_dict(r::BenchmarkRecord) = Dict{String,Any}(
    "algorithm" => String(r.algorithm), "backend" => String(r.backend),
    "precision" => String(r.precision),
    "transforms" => String[String(t) for t in r.transforms],
    "n_elements" => r.n_elements, "input_size" => r.input_size,
    "compile_s" => r.compile_s, "runtime_min_s" => r.runtime_min_s,
    "runtime_median_s" => r.runtime_median_s, "samples" => r.samples,
    "alloc_bytes" => r.alloc_bytes, "gflops" => r.gflops,
    "rel_error" => r.rel_error, "failed" => r.failed,
    "hw" => r.hw_fingerprint, "sw" => r.sw_fingerprint,
    "timestamp" => string(r.timestamp))

function load_memory(dir::AbstractString = joinpath(homedir(), ".autonomousai"))
    ms = MemorySystem(dir)
    load_into(p, default) = isfile(p) ? (try
        deserialize(p)
    catch err
        @warn "memory store unreadable, starting empty" path = p exception = err
        default
    end) : default
    ep = load_into(joinpath(dir, "episodic.jls"), EpisodicMemory())
    sm = load_into(joinpath(dir, "semantic.jls"), SemanticMemory())
    pm = load_into(joinpath(dir, "procedural.jls"), ProceduralMemory())
    hm = load_into(joinpath(dir, "hardware.jls"), HardwareMemory())
    return MemorySystem(WorkingMemory(), ep, sm, pm, hm, String(dir))
end

end # module
