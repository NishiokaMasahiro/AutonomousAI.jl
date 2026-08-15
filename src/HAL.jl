"""
    HAL

Hardware Abstraction Layer (spec sections 6, 7, 27, 28).

Two rules are structural, not stylistic:

1. Nothing above this layer touches a device, a file in `/sys`, or a driver.  The
   planner sees only immutable descriptor structs.
2. Hardware capability is **discovered**, never assumed.  The spec names an
   RTX 5070 / Core Ultra 7 as the minimum machine; treating that as a hard
   precondition would make the agent crash on the machine it is supposed to be
   reasoning about.  Instead `check_requirements` returns a *report* and the
   planner degrades the candidate set (see docs/06_design_review.md, issue D5).

GPU discovery is pluggable: `CUDA.jl` registers a probe through the package
extension in `ext/`, and `nvidia-smi` is used as a read-only fallback.
"""
module HAL

using SHA, Printf

export CPUInfo, GPUInfo, MemoryInfo, StorageInfo, HardwareProfile,
       probe_cpu, probe_memory, probe_storage, probe_gpus, probe_hardware,
       register_gpu_probe!, fingerprint, has_gpu, machine_balance,
       peak_flops, check_requirements, summarize

# ------------------------------------------------------------------ structs --

struct CPUInfo
    model::String
    arch::String
    physical_cores::Int
    logical_threads::Int
    julia_threads::Int
    base_clock_ghz::Float64      # NaN == unknown
    l1d_bytes::Int
    l2_bytes::Int
    l3_bytes::Int
    simd::Vector{Symbol}         # e.g. [:avx2, :avx512f, :amx_bf16]
    simd_width_bits::Int
end

struct GPUInfo
    index::Int
    name::String
    vendor::Symbol               # :nvidia, :amd, :intel, :none
    compute_capability::VersionNumber
    vram_total_bytes::Int
    vram_free_bytes::Int
    sm_count::Int
    memory_bandwidth_gbs::Float64
    cuda_runtime::VersionNumber
    driver::VersionNumber
    tensor_cores::Bool
    fp16::Bool
    bf16::Bool
    fp64_ratio::Float64          # FP64 : FP32 throughput ratio (1/64 on GeForce)
end

struct MemoryInfo
    total_bytes::Int
    available_bytes::Int
    page_size::Int
end

struct StorageInfo
    path::String
    total_bytes::Int
    available_bytes::Int
end

struct HardwareProfile
    cpu::CPUInfo
    gpus::Vector{GPUInfo}
    memory::MemoryInfo
    storage::StorageInfo
    hostname::String
    julia_version::VersionNumber
    os::String
end

has_gpu(p::HardwareProfile) = !isempty(p.gpus)

# ------------------------------------------------------------------- probes --

read_first(path::String) = try
    strip(read(path, String))
catch
    ""
end

function probe_cpu()
    model, flags = "unknown", String[]
    physical, logical = 0, Sys.CPU_THREADS
    clock = NaN
    coreids = Set{Tuple{String,String}}()
    if isfile("/proc/cpuinfo")
        pid, cid = "", ""
        for line in eachline("/proc/cpuinfo")
            parts = split(line, ':'; limit = 2)
            length(parts) == 2 || continue
            k, v = strip(parts[1]), strip(parts[2])
            if k == "model name"
                model = String(v)
            elseif k == "flags"
                isempty(flags) && (flags = String.(split(v)))
            elseif k == "physical id"
                pid = String(v)
            elseif k == "core id"
                cid = String(v)
                push!(coreids, (pid, cid))
            elseif k == "cpu MHz" && isnan(clock)
                p = tryparse(Float64, v)
                p === nothing || (clock = p / 1000)
            end
        end
        physical = length(coreids)
    end
    if isempty(model) || model == "unknown"
        model = string(Sys.CPU_NAME)
    end
    physical == 0 && (physical = max(1, div(logical, 2)))
    simd = Symbol[]
    for f in ("sse2", "avx", "avx2", "avx512f", "avx512bw", "avx512vnni",
              "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16", "amx_int8", "fma")
        f in flags && push!(simd, Symbol(f))
    end
    width = :avx512f in simd ? 512 : :avx2 in simd ? 256 : :sse2 in simd ? 128 : 64
    return CPUInfo(model, String(Sys.ARCH), physical, logical, Threads.nthreads(),
                   clock, cache_bytes("index0"), cache_bytes("index2"),
                   cache_bytes("index3"), simd, width)
end

function cache_bytes(index::String)
    p = "/sys/devices/system/cpu/cpu0/cache/$(index)/size"
    s = read_first(p)
    isempty(s) && return 0
    m = match(r"^(\d+)([KMG]?)", s)
    m === nothing && return 0
    n = parse(Int, m.captures[1])
    unit = m.captures[2]
    return unit == "K" ? n * 1024 : unit == "M" ? n * 1024^2 : unit == "G" ? n * 1024^3 : n
end

function probe_memory()
    total = Int(Sys.total_memory())
    avail = Int(Sys.free_memory())
    if isfile("/proc/meminfo")
        for line in eachline("/proc/meminfo")
            if startswith(line, "MemAvailable:")
                v = tryparse(Int, strip(replace(split(line, ':')[2], "kB" => "")))
                v === nothing || (avail = v * 1024)
                break
            end
        end
    end
    return MemoryInfo(total, avail, 4096)
end

function probe_storage(path::AbstractString = homedir())
    try
        st = Base.diskstat(String(path))
        return StorageInfo(String(path), Int(st.total), Int(st.available))
    catch
        return StorageInfo(String(path), 0, 0)
    end
end

"""
GPU probe registry.  A probe returns `Vector{GPUInfo}` or an empty vector.
`CUDA.jl` installs the authoritative probe via the package extension; the
`nvidia-smi` probe is the read-only fallback used when CUDA.jl is absent.
"""
const GPU_PROBES = Function[]
register_gpu_probe!(f::Function) = (pushfirst!(GPU_PROBES, f); nothing)

function nvidia_smi_probe()
    Sys.which("nvidia-smi") === nothing && return GPUInfo[]
    q = "index,name,memory.total,memory.free,compute_cap,driver_version"
    out = try
        read(`nvidia-smi --query-gpu=$(q) --format=csv,noheader,nounits`, String)
    catch
        return GPUInfo[]
    end
    gpus = GPUInfo[]
    for line in split(strip(out), '\n')
        isempty(strip(line)) && continue
        f = strip.(split(line, ','))
        length(f) >= 6 || continue
        cc = something(tryparse(VersionNumber, String(f[5])), v"0.0")
        total = Int(something(tryparse(Float64, String(f[3])), 0.0) * 1024^2)
        free = Int(something(tryparse(Float64, String(f[4])), 0.0) * 1024^2)
        push!(gpus, GPUInfo(something(tryparse(Int, String(f[1])), 0), String(f[2]),
                            :nvidia, cc, total, free, 0, NaN, v"0.0",
                            something(tryparse(VersionNumber, String(f[6])), v"0.0"),
                            cc >= v"7.0", cc >= v"5.3", cc >= v"8.0",
                            geforce_fp64_ratio(String(f[2]))))
    end
    return gpus
end

"""
GeForce parts run FP64 at 1/64 of FP32.  This single number is why "use the GPU for
the FP64 reference implementation" is a mistake on the target machine: the CPU is
usually the better FP64 oracle.  See docs/06_design_review.md issue D7.
"""
geforce_fp64_ratio(name::AbstractString) =
    occursin(r"GeForce|RTX \d0[5-9]0|GTX"i, name) ? 1 / 64 :
    occursin(r"A100|H100|H200|GH200|V100"i, name) ? 1 / 2 : 1 / 32

function probe_gpus()
    for p in GPU_PROBES
        gpus = try
            p()
        catch err
            @debug "GPU probe failed" probe = p exception = err
            GPUInfo[]
        end
        isempty(gpus) || return gpus
    end
    return nvidia_smi_probe()
end

function probe_hardware(; storage_path::AbstractString = homedir())
    return HardwareProfile(probe_cpu(), probe_gpus(), probe_memory(),
                           probe_storage(storage_path), gethostname(), VERSION,
                           string(Sys.KERNEL, " ", Sys.MACHINE))
end

# ------------------------------------------------------------- derived model --

"""
    peak_flops(cpu, precision) -> Float64

Rough vendor-independent peak: cores * clock * 2 (FMA) * lanes.  Used only as a
*prior* for the roofline estimate; every number that matters is replaced by a
measurement from `HardwareMemory`.
"""
function peak_flops(cpu::CPUInfo, precision::Symbol = :Float64)
    clock = isnan(cpu.base_clock_ghz) ? 2.5 : cpu.base_clock_ghz
    bits = precision === :Float64 ? 64 : precision === :Float32 ? 32 : 16
    lanes = cpu.simd_width_bits / bits
    return cpu.physical_cores * clock * 1e9 * 2 * lanes
end

function peak_flops(g::GPUInfo, precision::Symbol = :Float32)
    base = g.sm_count > 0 ? g.sm_count * 128 * 2 * 2.0e9 : 3.0e13
    precision === :Float64 && return base * g.fp64_ratio
    precision === :Float32 && return base
    return g.tensor_cores ? base * 8 : base * 2
end

"Bytes per FLOP the machine can sustain; the roofline ridge point."
machine_balance(cpu::CPUInfo, bandwidth_gbs::Float64) =
    peak_flops(cpu) / (bandwidth_gbs * 1e9)

"""
    fingerprint(profile) -> String

Benchmarks are only transferable across runs if the machine *and* the software
stack are the same.  Reusing a timing after a driver or Julia upgrade is a silent
correctness bug in the optimiser, so the key includes both.
"""
function fingerprint(p::HardwareProfile)
    io = IOBuffer()
    print(io, p.cpu.model, "|", p.cpu.physical_cores, "|", p.cpu.logical_threads, "|",
          join(sort(String.(p.cpu.simd)), ","), "|", p.memory.total_bytes, "|")
    for g in p.gpus
        print(io, g.name, ":", g.compute_capability, ":", g.vram_total_bytes, ":",
              g.driver, ";")
    end
    return bytes2hex(sha256(take!(io)))[1:16]
end

function software_fingerprint()
    io = IOBuffer()
    print(io, VERSION, "|", Sys.MACHINE, "|", Threads.nthreads())
    return bytes2hex(sha256(take!(io)))[1:16]
end

# --------------------------------------------------------------- capability --

"""
    check_requirements(profile; kwargs...) -> (ok::Bool, report::Vector{String})

Returns a report instead of throwing: on a machine below spec the agent must still
run, with the GPU/threaded candidates pruned from the search space.
"""
function check_requirements(p::HardwareProfile; min_ram_bytes::Int = 32 * 1024^3,
                            min_cc::VersionNumber = v"8.9",
                            min_vram_bytes::Int = 10 * 1024^3,
                            min_cores::Int = 8)
    report = String[]
    ok = true
    if p.memory.total_bytes < min_ram_bytes
        ok = false
        push!(report, @sprintf("RAM %.1f GiB < required %.1f GiB -> out-of-core paths only",
                               p.memory.total_bytes / 1024^3, min_ram_bytes / 1024^3))
    end
    if p.cpu.physical_cores < min_cores
        ok = false
        push!(report, "physical cores $(p.cpu.physical_cores) < $(min_cores) -> thread-scaling candidates pruned")
    end
    if isempty(p.gpus)
        ok = false
        push!(report, "no GPU visible -> :cuda backend removed from the candidate set")
    else
        g = p.gpus[1]
        g.compute_capability < min_cc && (ok = false;
            push!(report, "compute capability $(g.compute_capability) < $(min_cc)"))
        g.vram_total_bytes < min_vram_bytes && (ok = false;
            push!(report, @sprintf("VRAM %.1f GiB < %.1f GiB -> chunked execution mandatory",
                                   g.vram_total_bytes / 1024^3, min_vram_bytes / 1024^3)))
    end
    isempty(report) && push!(report, "all baseline requirements satisfied")
    return ok, report
end

function summarize(p::HardwareProfile)
    io = IOBuffer()
    println(io, "host      : ", p.hostname, "  (", p.os, ", julia ", p.julia_version, ")")
    println(io, "cpu       : ", p.cpu.model)
    println(io, @sprintf("            %d physical / %d logical cores, %d julia threads, SIMD %d-bit",
                         p.cpu.physical_cores, p.cpu.logical_threads,
                         p.cpu.julia_threads, p.cpu.simd_width_bits))
    println(io, "            isa: ", isempty(p.cpu.simd) ? "-" : join(p.cpu.simd, " "))
    println(io, @sprintf("memory    : %.1f GiB total, %.1f GiB available",
                         p.memory.total_bytes / 1024^3, p.memory.available_bytes / 1024^3))
    println(io, @sprintf("storage   : %s %.1f GiB available", p.storage.path,
                         p.storage.available_bytes / 1024^3))
    if isempty(p.gpus)
        println(io, "gpu       : none detected")
    else
        for g in p.gpus
            println(io, @sprintf("gpu[%d]    : %s  cc %s  %.1f GiB VRAM  fp16=%s bf16=%s tc=%s fp64=1/%d",
                                 g.index, g.name, string(g.compute_capability),
                                 g.vram_total_bytes / 1024^3, g.fp16, g.bf16,
                                 g.tensor_cores, round(Int, 1 / g.fp64_ratio)))
        end
    end
    print(io, "fingerprint: ", fingerprint(p), " / sw ", software_fingerprint())
    return String(take!(io))
end

end # module
