"""
    WorldModel

Observable state of the machine plus the state of the task (spec section 22).

Convention: `NaN` means *not measured on this platform*, and is distinct from `0.0`.
We use `Float64` + NaN rather than `Union{Float64,Missing}` deliberately -- the state
struct is read inside hot planning loops and `Union` fields would introduce dynamic
dispatch, which is exactly what section 17 asks the system to avoid in its own code.
"""
module WorldModel

using Dates, Printf
using ..HAL

export SystemState, HardwareState, EnvironmentState, WorldState,
       observe, observe!, cpu_utilization, is_measured, describe

"Instantaneous, sampled system telemetry."
struct SystemState
    timestamp::DateTime
    cpu_utilization::Float64     # [0,1], NaN if unavailable
    gpu_utilization::Float64
    ram_used_bytes::Int
    ram_total_bytes::Int
    vram_used_bytes::Int
    vram_total_bytes::Int
    cpu_temperature_c::Float64
    gpu_temperature_c::Float64
    power_w::Float64
    gc_live_bytes::Int
    gc_total_time_s::Float64
end

"Static-per-boot hardware description plus a rolling telemetry sample."
struct HardwareState
    profile::HAL.HardwareProfile
    system::SystemState
    fingerprint::String
end

"Everything outside the process that the planner is allowed to know about."
struct EnvironmentState
    working_dir::String
    dataset_paths::Vector{String}
    dataset_bytes::Int
    free_storage_bytes::Int
    network_permitted::Bool
end

"Task-level belief state; `confidence` is the planner's own calibration estimate."
mutable struct WorldState
    hardware::HardwareState
    environment::EnvironmentState
    task_progress::Float64
    estimated_remaining_s::Float64
    confidence::Float64
    iteration::Int
end

is_measured(x::Float64) = !isnan(x)

# ------------------------------------------------------------ /proc sampling --

const _PREV_CPU = Ref{Tuple{Float64,Float64}}((0.0, 0.0))

"""
    cpu_utilization() -> Float64

Delta of /proc/stat between calls.  The first call has no baseline and returns NaN
rather than a fabricated 0.0.
"""
function cpu_utilization()
    isfile("/proc/stat") || return NaN
    line = ""
    for l in eachline("/proc/stat")
        if startswith(l, "cpu ")
            line = l
            break
        end
    end
    isempty(line) && return NaN
    f = split(line)
    vals = Float64[something(tryparse(Float64, String(x)), 0.0) for x in f[2:end]]
    length(vals) >= 4 || return NaN
    idle = vals[4] + (length(vals) >= 5 ? vals[5] : 0.0)
    total = sum(vals)
    prev_total, prev_idle = _PREV_CPU[]
    _PREV_CPU[] = (total, idle)
    (prev_total == 0.0 || total <= prev_total) && return NaN
    return clamp(1 - (idle - prev_idle) / (total - prev_total), 0.0, 1.0)
end

function cpu_temperature()
    best = NaN
    for zone in 0:15
        p = "/sys/class/thermal/thermal_zone$(zone)/temp"
        isfile(p) || continue
        v = tryparse(Float64, strip(read(p, String)))
        v === nothing && continue
        t = v > 1000 ? v / 1000 : v
        (isnan(best) || t > best) && (best = t)
    end
    return best
end

"GPU telemetry via the read-only nvidia-smi fallback; CUDA.jl overrides through HAL."
function gpu_telemetry()
    Sys.which("nvidia-smi") === nothing && return (NaN, 0, 0, NaN, NaN)
    q = "utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    out = try
        read(`nvidia-smi --query-gpu=$(q) --format=csv,noheader,nounits`, String)
    catch
        return (NaN, 0, 0, NaN, NaN)
    end
    line = first(split(strip(out), '\n'))
    f = strip.(split(line, ','))
    length(f) >= 5 || return (NaN, 0, 0, NaN, NaN)
    num(i) = something(tryparse(Float64, String(f[i])), NaN)
    return (num(1) / 100, round(Int, num(2) * 1024^2), round(Int, num(3) * 1024^2),
            num(4), num(5))
end

function observe(profile::HAL.HardwareProfile)
    mem = HAL.probe_memory()
    gu, vused, vtotal, gtemp, power = gpu_telemetry()
    if vtotal == 0 && !isempty(profile.gpus)
        vtotal = profile.gpus[1].vram_total_bytes
    end
    return SystemState(now(), cpu_utilization(), gu,
                       mem.total_bytes - mem.available_bytes, mem.total_bytes,
                       vused, vtotal, cpu_temperature(), gtemp, power,
                       Int(Base.gc_live_bytes()), Base.gc_time_ns() / 1e9)
end

function observe!(w::WorldState)
    w.hardware = HardwareState(w.hardware.profile, observe(w.hardware.profile),
                               w.hardware.fingerprint)
    w.iteration += 1
    return w
end

function WorldState(; storage_path::AbstractString = pwd(),
                    dataset_paths::Vector{String} = String[],
                    network_permitted::Bool = false)
    profile = HAL.probe_hardware(storage_path = storage_path)
    hw = HardwareState(profile, observe(profile), HAL.fingerprint(profile))
    bytes = 0
    for p in dataset_paths
        bytes += isfile(p) ? filesize(p) : 0
    end
    env = EnvironmentState(String(storage_path), dataset_paths, bytes,
                           profile.storage.available_bytes, network_permitted)
    return WorldState(hw, env, 0.0, NaN, 0.5, 0)
end

function describe(w::WorldState)
    s = w.hardware.system
    io = IOBuffer()
    println(io, HAL.summarize(w.hardware.profile))
    fmt(x) = isnan(x) ? "n/a" : @sprintf("%.1f", x)
    println(io, @sprintf("telemetry : cpu %s%%  gpu %s%%  ram %.1f/%.1f GiB  vram %.1f/%.1f GiB",
                         fmt(100 * s.cpu_utilization), fmt(100 * s.gpu_utilization),
                         s.ram_used_bytes / 1024^3, s.ram_total_bytes / 1024^3,
                         s.vram_used_bytes / 1024^3, s.vram_total_bytes / 1024^3))
    print(io, @sprintf("            temp cpu %s C  gpu %s C  power %s W  gc-live %.2f GiB",
                       fmt(s.cpu_temperature_c), fmt(s.gpu_temperature_c),
                       fmt(s.power_w), s.gc_live_bytes / 1024^3))
    return String(take!(io))
end

end # module
