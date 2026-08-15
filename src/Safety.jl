"""
    Safety

Policy engine, resource limits, capabilities and emergency stop (spec sections 26,
29).

The constraint hierarchy of section 26 is implemented as *lexicographic* order, not
as weights in the cost function:

    safety  >  permission  >  goal  >  optimisation objective

A weight, however large, can always be outvoted by a large enough gain elsewhere;
a feasibility filter cannot.  `authorize` therefore returns a hard verdict and the
cost model in `Optimization` only ever ranks candidates that already passed it.
"""
module Safety

using Dates, Printf
using ..Schema
using ..WorldModel

export ResourceLimits, Capabilities, Policy, PolicyEngine, AuthDecision,
       EmergencyStop, authorize, trip!, reset!, check_stop, is_tripped,
       within_limits, default_limits, default_capabilities, ResourceViolation,
       with_watchdog

struct ResourceViolation <: Exception
    what::String
    observed::Float64
    limit::Float64
end
Base.showerror(io::IO, e::ResourceViolation) = print(io,
    "ResourceViolation: ", e.what, " observed ", e.observed, " > limit ", e.limit)

"""
Hard numeric envelope.  Every field is a *ceiling*; the agent may never propose an
action whose estimate exceeds one, and the watchdog aborts execution that reaches one.
"""
struct ResourceLimits
    max_cpu_utilization::Float64
    max_gpu_utilization::Float64
    max_ram_bytes::Int
    max_vram_bytes::Int
    max_temperature_c::Float64
    max_power_w::Float64
    max_runtime_s::Float64
    max_compile_s::Float64
    max_alloc_bytes::Int
    max_processes::Int
end

function default_limits(; ram_total::Int = 8 * 1024^3, vram_total::Int = 0)
    return ResourceLimits(0.95, 0.98, max(1024^3, div(ram_total * 6, 10)),
                          vram_total == 0 ? 0 : div(vram_total * 8, 10),
                          88.0, 400.0, 300.0, 120.0, 4 * 1024^3, 8)
end

"""
Capability set.  Absent capability == denied; there is no wildcard.  Filesystem
access is expressed as explicit path prefixes so a generated program cannot be
granted "the filesystem".
"""
struct Capabilities
    fs_read::Vector{String}
    fs_write::Vector{String}
    network::Bool
    devices::Vector{Symbol}
    gpu::Bool
    threads::Bool
    distributed::Bool
    self_modify::Bool
    real_hardware::Bool
end

default_capabilities(; workdir::AbstractString = mktempdir()) =
    Capabilities([String(workdir)], [String(workdir)], false, Symbol[], true, true,
                 false, false, false)

struct Policy
    limits::ResourceLimits
    capabilities::Capabilities
    allow_fastmath::Bool
    allow_inbounds::Bool
    require_verification::Bool
    max_optimization_iterations::Int
    max_total_benchmark_s::Float64
    min_relative_improvement::Float64
end

function Policy(limits::ResourceLimits, caps::Capabilities; allow_fastmath::Bool = false,
                allow_inbounds::Bool = true, require_verification::Bool = true,
                max_optimization_iterations::Int = 8,
                max_total_benchmark_s::Float64 = 600.0,
                min_relative_improvement::Float64 = 0.02)
    return Policy(limits, caps, allow_fastmath, allow_inbounds, require_verification,
                  max_optimization_iterations, max_total_benchmark_s,
                  min_relative_improvement)
end

struct AuthDecision
    allowed::Bool
    reason::String
    downgrade::Union{Nothing,Schema.Action}
end
AuthDecision(ok::Bool, reason::AbstractString) = AuthDecision(ok, String(reason), nothing)

"Process-wide abort latch, checked by every long-running loop."
struct EmergencyStop
    tripped::Threads.Atomic{Bool}
    reason::Ref{String}
end
EmergencyStop() = EmergencyStop(Threads.Atomic{Bool}(false), Ref(""))

function trip!(es::EmergencyStop, reason::AbstractString)
    es.reason[] = String(reason)
    Threads.atomic_xchg!(es.tripped, true)
    @warn "EMERGENCY STOP" reason
    return es
end
reset!(es::EmergencyStop) = (Threads.atomic_xchg!(es.tripped, false); es.reason[] = ""; es)
is_tripped(es::EmergencyStop) = es.tripped[]
check_stop(es::EmergencyStop) = is_tripped(es) &&
    error("execution aborted by emergency stop: ", es.reason[])

mutable struct PolicyEngine
    policy::Policy
    stop::EmergencyStop
    denials::Vector{Tuple{DateTime,String}}
end
PolicyEngine(p::Policy) = PolicyEngine(p, EmergencyStop(), Tuple{DateTime,String}[])

deny(pe::PolicyEngine, msg::AbstractString) =
    (push!(pe.denials, (now(), String(msg))); AuthDecision(false, msg))

"""
    within_limits(state, limits) -> (ok, violations)

Telemetry that could not be measured (NaN) is *not* treated as satisfying the limit
silently; it is reported so the policy layer can decide whether to run at all.
"""
function within_limits(s::WorldModel.SystemState, l::ResourceLimits)
    v = String[]
    chk(name, x, lim) = begin
        if isnan(x)
            push!(v, "$(name): unmeasured")
        elseif x > lim
            push!(v, @sprintf("%s: %.2f > %.2f", name, x, lim))
        end
    end
    chk("cpu_utilization", s.cpu_utilization, l.max_cpu_utilization)
    chk("cpu_temperature_c", s.cpu_temperature_c, l.max_temperature_c)
    chk("gpu_temperature_c", s.gpu_temperature_c, l.max_temperature_c)
    chk("power_w", s.power_w, l.max_power_w)
    if s.ram_used_bytes > l.max_ram_bytes
        push!(v, @sprintf("ram: %.1f GiB > %.1f GiB", s.ram_used_bytes / 1024^3,
                          l.max_ram_bytes / 1024^3))
    end
    if l.max_vram_bytes > 0 && s.vram_used_bytes > l.max_vram_bytes
        push!(v, @sprintf("vram: %.1f GiB > %.1f GiB", s.vram_used_bytes / 1024^3,
                          l.max_vram_bytes / 1024^3))
    end
    hard = filter(x -> !endswith(x, "unmeasured"), v)
    return (isempty(hard), v)
end

"""
    authorize(engine, action, world) -> AuthDecision

Single choke point.  Every action reaching the executor passed through here.
"""
function authorize(pe::PolicyEngine, a::Schema.Action, w::WorldModel.WorldState)
    is_tripped(pe.stop) && return deny(pe, "emergency stop is latched")

    # For execution actions, fail first on static candidate/footprint checks so the
    # caller gets actionable planning feedback independent of ambient telemetry.
    if a isa Schema.BenchmarkAlgorithm || a isa Schema.ExecuteFinal
        d = authorize_action(pe, a, w)
        d.allowed || return d
        ok, viol = within_limits(w.hardware.system, pe.policy.limits)
        ok || return deny(pe, "resource envelope exceeded: " * join(viol, "; "))
        return d
    end

    ok, viol = within_limits(w.hardware.system, pe.policy.limits)
    ok || return deny(pe, "resource envelope exceeded: " * join(viol, "; "))
    return authorize_action(pe, a, w)
end

authorize_action(::PolicyEngine, ::Schema.InspectHardware, ::WorldModel.WorldState) =
    AuthDecision(true, "read-only introspection")
authorize_action(::PolicyEngine, ::Schema.ProposeCandidates, ::WorldModel.WorldState) =
    AuthDecision(true, "planning only")
authorize_action(::PolicyEngine, a::Schema.Abort, ::WorldModel.WorldState) =
    AuthDecision(true, "abort is always permitted")

function authorize_action(pe::PolicyEngine, a::Schema.RequestPermission,
                          ::WorldModel.WorldState)
    return deny(pe, "capability '$(a.capability)' requires out-of-band human grant")
end

function authorize_action(pe::PolicyEngine, a::Schema.GenerateCode,
                          ::WorldModel.WorldState)
    return authorize_candidate(pe, a.candidate)
end

function authorize_action(pe::PolicyEngine, a::Schema.BenchmarkAlgorithm,
                          w::WorldModel.WorldState)
    d = authorize_candidate(pe, a.candidate)
    d.allowed || return d
    return authorize_size(pe, a.candidate, a.input_size, w)
end

function authorize_action(pe::PolicyEngine, a::Schema.ExecuteFinal,
                          w::WorldModel.WorldState)
    d = authorize_candidate(pe, a.candidate)
    d.allowed || return d
    return authorize_size(pe, a.candidate, a.input_size, w)
end

authorize_action(pe::PolicyEngine, a::Schema.TuneParameter, ::WorldModel.WorldState) =
    authorize_candidate(pe, a.candidate)
authorize_action(pe::PolicyEngine, a::Schema.VerifyResult, ::WorldModel.WorldState) =
    authorize_candidate(pe, a.candidate)

function authorize_candidate(pe::PolicyEngine, c::Schema.Candidate)
    caps = pe.policy.capabilities
    if Schema.is_gpu_backend(c.backend) && !caps.gpu
        return deny(pe, "gpu capability not granted")
    end
    if c.backend === :cpu_threads && !caps.threads
        return deny(pe, "thread capability not granted")
    end
    if c.backend === :distributed && !caps.distributed
        return deny(pe, "distributed capability not granted")
    end
    if :fastmath in c.transforms && !pe.policy.allow_fastmath
        return AuthDecision(false, "fastmath denied by policy (precision-sensitive)",
                            Schema.GenerateCode(Schema.Candidate(c.algorithm, c.backend,
                                c.precision, filter(!=(:fastmath), c.transforms),
                                c.params)))
    end
    if :inbounds in c.transforms && !pe.policy.allow_inbounds
        return deny(pe, "inbounds denied by policy")
    end
    return AuthDecision(true, "candidate within capability set")
end

"Reject work whose *estimated* footprint already exceeds the envelope."
function authorize_size(pe::PolicyEngine, c::Schema.Candidate, size::Vector{Int},
                        w::WorldModel.WorldState)
    n = prod(size)
    bytes_per = c.precision === :Float64 ? 8 : c.precision === :Float32 ? 4 : 2
    working = 3 * n * bytes_per          # in + out + scratch, conservative
    l = pe.policy.limits
    if working > l.max_ram_bytes
        return deny(pe, @sprintf("estimated working set %.2f GiB exceeds RAM ceiling %.2f GiB",
                                 working / 1024^3, l.max_ram_bytes / 1024^3))
    end
    if Schema.is_gpu_backend(c.backend)
        vram_ceiling = l.max_vram_bytes
        if vram_ceiling == 0
            return deny(pe, "gpu requested but no VRAM ceiling is configured")
        end
        if working > vram_ceiling
            return deny(pe, @sprintf("estimated %.2f GiB exceeds VRAM ceiling %.2f GiB (chunking required)",
                                     working / 1024^3, vram_ceiling / 1024^3))
        end
    end
    return AuthDecision(true, "footprint within envelope")
end

"""
    with_watchdog(f, engine, world; period=0.25, timeout)

Runs `f()` while a sampling task polls telemetry.  On a limit breach the emergency
stop is latched; `f` is expected to poll `check_stop`.  We do *not* kill the calling
task asynchronously -- `Base.throwto` into arbitrary compute is unsafe.  Killing is
the sandbox's job (separate OS process), which is why untrusted work never runs here.
"""
function with_watchdog(f::Function, pe::PolicyEngine, w::WorldModel.WorldState;
                       period::Float64 = 0.25, timeout::Float64 = NaN)
    t0 = time()
    limit = isnan(timeout) ? pe.policy.limits.max_runtime_s : timeout
    done = Threads.Atomic{Bool}(false)
    monitor = Threads.@spawn begin
        while !done[]
            sleep(period)
            s = WorldModel.observe(w.hardware.profile)
            ok, viol = within_limits(s, pe.policy.limits)
            ok || trip!(pe.stop, join(viol, "; "))
            (time() - t0) > limit && trip!(pe.stop, "runtime limit $(limit)s exceeded")
        end
    end
    try
        return f()
    finally
        Threads.atomic_xchg!(done, true)
        try
            wait(monitor)
        catch
        end
    end
end

end # module
