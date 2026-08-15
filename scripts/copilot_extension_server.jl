#!/usr/bin/env julia

module CopilotExtensionServer

using Sockets
using Dates
using AutonomousAI

const MJ = AutonomousAI.MiniJSON
const SC = AutonomousAI.Schema
const AC = AutonomousAI.AgentCore
const LLM = AutonomousAI.LLM

struct ServerConfig
    host::String
    port::Int
    chunk_ms::Int
    backend::String
end

function parse_args(args::Vector{String})
    host = get(ENV, "AAI_SERVER_HOST", "127.0.0.1")
    port = parse(Int, get(ENV, "AAI_SERVER_PORT", "8081"))
    chunk_ms = parse(Int, get(ENV, "AAI_SSE_CHUNK_MS", "35"))
    backend = get(ENV, "AAI_LLM_BACKEND", "mock")

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--host" && i < length(args)
            host = args[i + 1]
            i += 2
        elseif a == "--port" && i < length(args)
            port = parse(Int, args[i + 1])
            i += 2
        elseif a == "--chunk-ms" && i < length(args)
            chunk_ms = parse(Int, args[i + 1])
            i += 2
        elseif a == "--backend" && i < length(args)
            backend = args[i + 1]
            i += 2
        else
            i += 1
        end
    end
    return ServerConfig(host, port, chunk_ms, backend)
end

function build_backend(name::String)
    n = lowercase(strip(name))
    if n == "mock"
        return LLM.MockLLM()
    elseif n == "openai"
        model = get(ENV, "AAI_OPENAI_MODEL", "gpt-4o-mini")
        url = get(ENV, "AAI_OPENAI_URL", "http://127.0.0.1:8080/v1/chat/completions")
        return LLM.OpenAICompatibleLLM(model = model, url = url)
    elseif n == "anthropic"
        model = get(ENV, "AAI_ANTHROPIC_MODEL", "claude-sonnet-4-6")
        return LLM.AnthropicLLM(model = model)
    end
    error("unknown backend '$(name)'; use mock|openai|anthropic")
end

function http_date_now()
    return Dates.format(now(Dates.UTC), Dates.RFC1123Format)
end

function read_request(sock::TCPSocket)
    line = try
        readline(sock)
    catch
        return nothing
    end
    isempty(line) && return nothing

    parts = split(line)
    length(parts) == 3 || error("malformed request line")
    method, target, _http = parts

    headers = Dict{String,String}()
    while true
        h = readline(sock)
        h == "" && break
        k, v = split(h, ":"; limit = 2)
        headers[lowercase(strip(k))] = strip(v)
    end

    n = try
        parse(Int, get(headers, "content-length", "0"))
    catch
        0
    end
    body = n > 0 ? String(read(sock, n)) : ""
    return (method = method, target = target, headers = headers, body = body)
end

function write_response(sock::TCPSocket, code::Int, body::String;
                        content_type::String = "application/json; charset=utf-8")
    reason = code == 200 ? "OK" : code == 404 ? "Not Found" :
             code == 405 ? "Method Not Allowed" :
             code == 400 ? "Bad Request" : "Internal Server Error"
    bytes = codeunits(body)
    print(sock, "HTTP/1.1 $(code) $(reason)\r\n")
    print(sock, "Date: $(http_date_now())\r\n")
    print(sock, "Server: AutonomousAI-CopilotBridge/0.1\r\n")
    print(sock, "Content-Type: $(content_type)\r\n")
    print(sock, "Access-Control-Allow-Origin: *\r\n")
    print(sock, "Access-Control-Allow-Headers: content-type, authorization\r\n")
    print(sock, "Access-Control-Allow-Methods: POST, GET, OPTIONS\r\n")
    print(sock, "Content-Length: $(length(bytes))\r\n")
    print(sock, "Connection: close\r\n\r\n")
    write(sock, bytes)
    flush(sock)
end

function sse_start(sock::TCPSocket)
    print(sock, "HTTP/1.1 200 OK\r\n")
    print(sock, "Date: $(http_date_now())\r\n")
    print(sock, "Server: AutonomousAI-CopilotBridge/0.1\r\n")
    print(sock, "Content-Type: text/event-stream; charset=utf-8\r\n")
    print(sock, "Cache-Control: no-cache\r\n")
    print(sock, "Connection: close\r\n")
    print(sock, "Access-Control-Allow-Origin: *\r\n")
    print(sock, "X-Accel-Buffering: no\r\n\r\n")
    flush(sock)
end

function sse_event(sock::TCPSocket, event::String, payload::AbstractDict)
    print(sock, "event: ", event, "\n")
    print(sock, "data: ", MJ.to_json(payload), "\n\n")
    flush(sock)
end

function bad_request(msg::String)
    return MJ.to_json(Dict("ok" => false, "error" => msg, "ts" => string(now())))
end

function infer_payload(req::AbstractDict)
    prompt = String(get(req, "prompt", ""))
    context = get(req, "context", Dict{String,Any}())
    code_context = String(get(req, "code_context", ""))
    goal_id = String(get(req, "goal_id", "copilot-goal"))

    isempty(prompt) && error("'prompt' is required")

    algorithm = AC.infer_algorithm(prompt)
    input_size = Int[10_000]
    if context isa AbstractDict && haskey(context, "input_size") && context["input_size"] isa AbstractVector
        raw = context["input_size"]
        input_size = [Int(x) for x in raw]
    end

    goal = AC.Goal(prompt; algorithm = algorithm, input_size = input_size, n_calls = 20)
    agent = AC.Agent(verbose = false)
    ctx = AC.build_context(agent, goal)
    ctx["goal_id"] = goal_id
    ctx["copilot_context"] = context
    isempty(code_context) || (ctx["code_context"] = code_context)

    return (goal_id = goal_id, goal = goal, plan_context = ctx)
end

function synthesize_response(backend::LLM.LLMBackend, payload)
    plan, logs = LLM.propose_plan(backend, payload.plan_context, payload.goal_id)
    if plan === nothing
        plan = AC.default_plan(AC.Agent(verbose = false), payload.goal)
        push!(logs, "fallback: default_plan used")
    end

    plan_json = MJ.to_json(SC.to_dict(plan); indent = 0)
    return Dict{String,Any}(
        "ok" => true,
        "goal_id" => payload.goal_id,
        "algorithm" => String(payload.goal.algorithm),
        "plan" => SC.to_dict(plan),
        "plan_json" => plan_json,
        "planner_log" => logs,
        "backend" => LLM.backend_name(backend),
        "ts" => string(now()),
    )
end

function chunk_text(s::String, n::Int)
    out = String[]
    i = firstindex(s)
    while i <= lastindex(s)
        j = i
        k = 0
        while j <= lastindex(s) && k < n
            j = nextind(s, j)
            k += 1
        end
        push!(out, s[i:prevind(s, j)])
        i = j
    end
    return out
end

function handle_infer(sock::TCPSocket, body::String, backend::LLM.LLMBackend)
    req = MJ.parse_json(body)
    req isa AbstractDict || error("request body must be a JSON object")
    payload = infer_payload(req)
    result = synthesize_response(backend, payload)
    write_response(sock, 200, MJ.to_json(result; indent = 0))
end

function handle_infer_stream(sock::TCPSocket, body::String, backend::LLM.LLMBackend,
                             chunk_ms::Int)
    req = MJ.parse_json(body)
    req isa AbstractDict || error("request body must be a JSON object")
    payload = infer_payload(req)
    result = synthesize_response(backend, payload)

    sse_start(sock)
    sse_event(sock, "meta", Dict("goal_id" => result["goal_id"],
                                  "algorithm" => result["algorithm"],
                                  "backend" => result["backend"]))

    text = String(result["plan_json"])
    for (idx, piece) in enumerate(chunk_text(text, 48))
        sse_event(sock, "delta", Dict("index" => idx, "text" => piece))
        sleep(max(chunk_ms, 0) / 1000)
    end

    sse_event(sock, "done", Dict("ok" => true,
                                  "planner_log" => result["planner_log"],
                                  "ts" => result["ts"]))
end

function handle_client(sock::TCPSocket, backend::LLM.LLMBackend, cfg::ServerConfig)
    try
        req = read_request(sock)
        req === nothing && return

        if req.method == "OPTIONS"
            write_response(sock, 200, "")
            return
        end

        if req.method == "GET" && req.target == "/health"
            body = MJ.to_json(Dict("ok" => true, "service" => "AutonomousAI Copilot Bridge",
                                   "backend" => LLM.backend_name(backend),
                                   "ts" => string(now())))
            write_response(sock, 200, body)
            return
        end

        if req.method == "POST" && req.target == "/v1/infer"
            handle_infer(sock, req.body, backend)
            return
        end

        if req.method == "POST" && req.target == "/v1/infer/stream"
            handle_infer_stream(sock, req.body, backend, cfg.chunk_ms)
            return
        end

        write_response(sock, 404, bad_request("route not found"))
    catch err
        write_response(sock, 400, bad_request(string(err)))
    finally
        try
            close(sock)
        catch
        end
    end
end

function serve(cfg::ServerConfig)
    backend = build_backend(cfg.backend)
    server = listen(parse(IPAddr, cfg.host), cfg.port)
    println("[copilot-bridge] listening on http://", cfg.host, ":", cfg.port,
            " backend=", LLM.backend_name(backend))
    println("[copilot-bridge] routes: GET /health, POST /v1/infer, POST /v1/infer/stream")

    while true
        sock = accept(server)
        @async handle_client(sock, backend, cfg)
    end
end

function print_usage()
    println("Usage:")
    println("  julia --project=. scripts/copilot_extension_server.jl [--host 127.0.0.1] [--port 8081] [--backend mock|openai|anthropic]")
    println("")
    println("Environment:")
    println("  AAI_SERVER_HOST, AAI_SERVER_PORT, AAI_SSE_CHUNK_MS")
    println("  AAI_LLM_BACKEND=mock|openai|anthropic")
    println("  AAI_OPENAI_URL, AAI_OPENAI_MODEL, OPENAI_API_KEY")
    println("  AAI_ANTHROPIC_MODEL, ANTHROPIC_API_KEY")
end

function main(args::Vector{String} = ARGS)
    if any(a -> a in ("-h", "--help"), args)
        print_usage()
        return nothing
    end
    cfg = parse_args(args)
    serve(cfg)
    return nothing
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    CopilotExtensionServer.main()
end
