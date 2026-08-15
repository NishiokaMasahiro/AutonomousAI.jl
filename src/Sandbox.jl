"""
    Sandbox

Out-of-process execution of generated code (spec section 30).

**The load-bearing claim of this module is deliberately narrow.**  A Julia process
cannot sandbox itself: `eval`, `ccall`, `@generated`, macro expansion and
`Base.invokelatest` all defeat any in-process restriction, and the static validator
in `CodeGeneration` is a *first* filter, not a boundary.  Therefore:

* generated code is never `eval`ed in the agent process -- not once, not for
  "just a syntax check";
* it runs in a fresh OS process with `ulimit` caps, a scrubbed environment, a
  private working directory and a hard wall-clock kill;
* `isolation = :container` additionally wraps that process in a container with no
  network and a read-only root, which is the configuration intended for anything
  that came from a model rather than from `TEMPLATES`.

`:process` isolation bounds *resources*, not *authority*: a program running as the
same uid can still read that uid's files.  This is stated here rather than buried,
because treating ulimit as a security boundary is the most common way this design
fails in practice (docs/06_design_review.md, issue D2).
"""
module Sandbox

using Dates, Printf
using ..MiniJSON
using ..Safety

export SandboxResult, SandboxSpec, run_sandboxed, sandbox_available

const RESULT_MARKER = "@@AUTONOMOUSAI_RESULT@@"

struct SandboxResult
    ok::Bool
    payload::Dict{String,Any}
    stdout::String
    stderr::String
    wall_s::Float64
    exit_code::Int
    killed::Bool
    error::String
end

struct SandboxSpec
    isolation::Symbol            # :process | :container
    image::String
    julia::String
    timeout_s::Float64
    address_space_kb::Int
    cpu_time_s::Int
    file_size_kb::Int
    threads::Int
    gpu::Bool
end

function SandboxSpec(limits::Safety.ResourceLimits; isolation::Symbol = :process,
                     image::String = "julia:1.11", threads::Int = 1, gpu::Bool = false)
    return SandboxSpec(isolation, image, joinpath(Sys.BINDIR, "julia"),
                       limits.max_runtime_s,
                       max(1024^2, div(limits.max_ram_bytes, 1024)),
                       max(1, ceil(Int, limits.max_runtime_s * 2)),
                       4 * 1024^2, threads, gpu)
end

sandbox_available() = isfile(joinpath(Sys.BINDIR, "julia"))

"""
    run_sandboxed(unit_source, driver_source, spec; payload=Dict()) -> SandboxResult

`unit_source` is the validated generated code.  `driver_source` is *trusted* text
composed by the benchmark harness -- never by the model -- that builds inputs, calls
the entry point and prints one JSON line prefixed by `RESULT_MARKER`.
"""
function run_sandboxed(unit_source::AbstractString, driver_source::AbstractString,
                       spec::SandboxSpec; payload::AbstractDict = Dict{String,Any}())
    dir = mktempdir(; prefix = "aai_sbx_")
    try
        write(joinpath(dir, "unit.jl"), String(unit_source))
        write(joinpath(dir, "payload.json"), to_json(payload))
        write(joinpath(dir, "main.jl"), main_script(driver_source))
        cmd = build_command(dir, spec)
        return execute(cmd, spec, dir)
    finally
        try
            rm(dir; recursive = true, force = true)
        catch
        end
    end
end

function main_script(driver::AbstractString)
    return string("""
    # trusted harness driver -- composed by the benchmark layer, not by the model
    const PAYLOAD_PATH = joinpath(@__DIR__, "payload.json")
    include(joinpath(@__DIR__, "unit.jl"))
    const RESULT_MARKER = "$(RESULT_MARKER)"
    """, "\n", String(driver), "\n")
end

function build_command(dir::String, spec::SandboxSpec)
    jl = spec.julia
    depot = get(ENV, "JULIA_DEPOT_PATH", "")
    if isempty(depot)
        depot = joinpath(dir, "depot")
        mkpath(depot)
    end

    if Sys.iswindows() && spec.isolation === :process
        env = Dict(
            "PATH" => get(ENV, "PATH", ""),
            "HOME" => dir,
            "TMP" => dir,
            "TEMP" => dir,
            "JULIA_NUM_THREADS" => string(spec.threads),
            "JULIA_DEPOT_PATH" => depot,
        )
        cmd = Cmd(String[jl, "--startup-file=no", "--history-file=no",
                         "--color=no", "--threads=$(spec.threads)",
                         joinpath(dir, "main.jl")])
        return setenv(cmd, env; dir = dir)
    end

    inner = string("ulimit -v ", spec.address_space_kb, "; ",
                   "ulimit -t ", spec.cpu_time_s, "; ",
                   "ulimit -f ", spec.file_size_kb, "; ",
                   "ulimit -c 0; ",
                   "exec ", jl, " --startup-file=no --history-file=no ",
                   "--color=no --threads=", spec.threads, " ",
                   joinpath(dir, "main.jl"))
    env = Dict("PATH" => "/usr/bin:/bin", "HOME" => dir, "TMPDIR" => dir,
               "JULIA_NUM_THREADS" => string(spec.threads),
               "JULIA_DEPOT_PATH" => depot,
               "LANG" => "C")
    if spec.isolation === :container
        args = String["docker", "run", "--rm", "--network=none", "--read-only",
                      "--cap-drop=ALL", "--security-opt=no-new-privileges",
                      "--pids-limit=64",
                      "--memory=$(div(spec.address_space_kb, 1024))m",
                      "--cpus=1", "-v", "$(dir):/work:ro",
                      "--tmpfs", "/tmp:rw,size=64m", "-w", "/work"]
        spec.gpu && append!(args, ["--gpus", "all"])
        append!(args, [spec.image, "julia", "--startup-file=no",
                       "--threads=$(spec.threads)", "/work/main.jl"])
        return setenv(Cmd(args), env)
    end
    return setenv(Cmd(String["/bin/sh", "-c", inner]), env; dir = dir)
end

function execute(cmd::Cmd, spec::SandboxSpec, dir::String)
    out, errio = IOBuffer(), IOBuffer()
    t0 = time()
    proc = try
        run(pipeline(ignorestatus(cmd), stdout = out, stderr = errio); wait = false)
    catch err
        return SandboxResult(false, Dict{String,Any}(), "", "", 0.0, -1, false,
                             "failed to spawn sandbox: $(err)")
    end
    killed = Threads.Atomic{Bool}(false)
    watchdog = Threads.@spawn begin
        deadline = t0 + spec.timeout_s
        while process_running(proc) && time() < deadline
            sleep(0.05)
        end
        if process_running(proc)
            Threads.atomic_xchg!(killed, true)
            try
                kill(proc, Base.SIGKILL)
            catch
            end
        end
    end
    wait(proc)
    try
        wait(watchdog)
    catch
    end
    wall = time() - t0
    so, se = String(take!(out)), String(take!(errio))
    code = proc.exitcode
    payload, perr = extract_payload(so)
    was_killed = killed[]
    ok = (code == 0) && !was_killed && isempty(perr)
    return SandboxResult(ok, payload, so, se, wall, code, was_killed,
                         was_killed ? "killed after $(spec.timeout_s)s" : perr)
end

function extract_payload(stdout_text::AbstractString)
    for line in Iterators.reverse(split(stdout_text, '\n'))
        startswith(line, RESULT_MARKER) || continue
        json = String(line[(length(RESULT_MARKER)+1):end])
        try
            v = parse_json(json)
            v isa AbstractDict && return (Dict{String,Any}(v), "")
            return (Dict{String,Any}("value" => v), "")
        catch err
            return (Dict{String,Any}(), "unparseable result payload: $(err)")
        end
    end
    return (Dict{String,Any}(), "no result marker in sandbox output")
end

end # module
