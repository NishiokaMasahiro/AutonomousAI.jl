"""
    MiniJSON

Dependency-free JSON encoder/decoder.

Rationale: the LLM <-> Julia runtime boundary is defined by a typed schema serialised
as JSON (spec section 37/38).  The core package deliberately has **zero** non-stdlib
dependencies so that it is installable and auditable in an air-gapped environment.
If `JSON3.jl` is available in the host project it should be preferred for
performance; this module exists so the agent never fails to start.
"""
module MiniJSON

export parse_json, to_json, JSONError

struct JSONError <: Exception
    msg::String
    pos::Int
end
Base.showerror(io::IO, e::JSONError) = print(io, "JSONError at byte ", e.pos, ": ", e.msg)

mutable struct P
    b::Vector{UInt8}
    i::Int
end

@inline peekb(p::P) = p.i <= length(p.b) ? p.b[p.i] : 0x00

@inline function skipws!(p::P)
    while p.i <= length(p.b)
        c = p.b[p.i]
        if c == 0x20 || c == 0x09 || c == 0x0a || c == 0x0d
            p.i += 1
        else
            break
        end
    end
    return nothing
end

function expectb!(p::P, c::UInt8)
    skipws!(p)
    if peekb(p) != c
        throw(JSONError(string("expected '", Char(c), "'"), p.i))
    end
    p.i += 1
    return nothing
end

"""
    parse_json(s) -> Any

Objects become `Dict{String,Any}`, arrays `Vector{Any}`, `null` becomes `nothing`.
"""
function parse_json(s::AbstractString)
    p = P(Vector{UInt8}(codeunits(String(s))), 1)
    v = parse_value!(p)
    skipws!(p)
    p.i <= length(p.b) && throw(JSONError("trailing content", p.i))
    return v
end

function parse_value!(p::P)
    skipws!(p)
    c = peekb(p)
    if c == UInt8('{')
        return parse_object!(p)
    elseif c == UInt8('[')
        return parse_array!(p)
    elseif c == UInt8('"')
        return parse_string!(p)
    elseif c == UInt8('t')
        return parse_lit!(p, "true", true)
    elseif c == UInt8('f')
        return parse_lit!(p, "false", false)
    elseif c == UInt8('n')
        return parse_lit!(p, "null", nothing)
    elseif c == 0x00
        throw(JSONError("unexpected end of input", p.i))
    else
        return parse_number!(p)
    end
end

function parse_lit!(p::P, lit::String, val)
    n = ncodeunits(lit)
    if p.i + n - 1 > length(p.b) || String(p.b[p.i:p.i+n-1]) != lit
        throw(JSONError(string("invalid literal, expected ", lit), p.i))
    end
    p.i += n
    return val
end

function parse_object!(p::P)
    expectb!(p, UInt8('{'))
    d = Dict{String,Any}()
    skipws!(p)
    if peekb(p) == UInt8('}')
        p.i += 1
        return d
    end
    while true
        skipws!(p)
        k = parse_string!(p)
        expectb!(p, UInt8(':'))
        d[k] = parse_value!(p)
        skipws!(p)
        c = peekb(p)
        if c == UInt8(',')
            p.i += 1
        elseif c == UInt8('}')
            p.i += 1
            return d
        else
            throw(JSONError("expected ',' or '}'", p.i))
        end
    end
end

function parse_array!(p::P)
    expectb!(p, UInt8('['))
    a = Any[]
    skipws!(p)
    if peekb(p) == UInt8(']')
        p.i += 1
        return a
    end
    while true
        push!(a, parse_value!(p))
        skipws!(p)
        c = peekb(p)
        if c == UInt8(',')
            p.i += 1
        elseif c == UInt8(']')
            p.i += 1
            return a
        else
            throw(JSONError("expected ',' or ']'", p.i))
        end
    end
end

function parse_string!(p::P)
    expectb!(p, UInt8('"'))
    io = IOBuffer()
    while true
        p.i > length(p.b) && throw(JSONError("unterminated string", p.i))
        c = p.b[p.i]
        p.i += 1
        if c == UInt8('"')
            return String(take!(io))
        elseif c == UInt8('\\')
            p.i > length(p.b) && throw(JSONError("unterminated escape", p.i))
            e = p.b[p.i]
            p.i += 1
            if e == UInt8('"')
                write(io, UInt8('"'))
            elseif e == UInt8('\\')
                write(io, UInt8('\\'))
            elseif e == UInt8('/')
                write(io, UInt8('/'))
            elseif e == UInt8('b')
                write(io, UInt8(0x08))
            elseif e == UInt8('f')
                write(io, UInt8(0x0c))
            elseif e == UInt8('n')
                write(io, UInt8('\n'))
            elseif e == UInt8('r')
                write(io, UInt8('\r'))
            elseif e == UInt8('t')
                write(io, UInt8('\t'))
            elseif e == UInt8('u')
                cp = read_hex4!(p)
                if 0xD800 <= cp <= 0xDBFF
                    if p.i + 1 <= length(p.b) && p.b[p.i] == UInt8('\\') && p.b[p.i+1] == UInt8('u')
                        p.i += 2
                        lo = read_hex4!(p)
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00)
                    end
                end
                write(io, Char(cp))
            else
                throw(JSONError("invalid escape", p.i))
            end
        else
            write(io, c)
        end
    end
end

function read_hex4!(p::P)
    p.i + 3 > length(p.b) && throw(JSONError("truncated \\u escape", p.i))
    v = UInt32(0)
    for _ in 1:4
        c = p.b[p.i]
        p.i += 1
        d = if UInt8('0') <= c <= UInt8('9')
            c - UInt8('0')
        elseif UInt8('a') <= c <= UInt8('f')
            c - UInt8('a') + 0x0a
        elseif UInt8('A') <= c <= UInt8('F')
            c - UInt8('A') + 0x0a
        else
            throw(JSONError("invalid hex digit", p.i))
        end
        v = v * UInt32(16) + UInt32(d)
    end
    return v
end

@inline isnumbyte(c::UInt8) = (UInt8('0') <= c <= UInt8('9')) || c == UInt8('-') ||
    c == UInt8('+') || c == UInt8('.') || c == UInt8('e') || c == UInt8('E')

function parse_number!(p::P)
    start = p.i
    isfloat = false
    while p.i <= length(p.b) && isnumbyte(p.b[p.i])
        c = p.b[p.i]
        if c == UInt8('.') || c == UInt8('e') || c == UInt8('E')
            isfloat = true
        end
        p.i += 1
    end
    p.i == start && throw(JSONError("invalid number", p.i))
    s = String(p.b[start:p.i-1])
    if !isfloat
        v = tryparse(Int64, s)
        v === nothing || return v
    end
    v = tryparse(Float64, s)
    v === nothing && throw(JSONError(string("invalid number '", s, "'"), start))
    return v
end

# ---------------------------------------------------------------- encoding ---

"""
    to_json(x; indent=0) -> String
"""
function to_json(x; indent::Int = 0)
    io = IOBuffer()
    write_json(io, x, indent, 0)
    return String(take!(io))
end

function write_json(io::IO, x::AbstractDict, indent::Int, depth::Int)
    isempty(x) && (print(io, "{}"); return nothing)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        newline_indent(io, indent, depth + 1)
        write_json_string(io, string(k))
        print(io, indent > 0 ? ": " : ":")
        write_json(io, v, indent, depth + 1)
    end
    newline_indent(io, indent, depth)
    print(io, "}")
    return nothing
end

function write_json(io::IO, x::Union{AbstractVector,Tuple}, indent::Int, depth::Int)
    isempty(x) && (print(io, "[]"); return nothing)
    print(io, "[")
    first = true
    for v in x
        first || print(io, ",")
        first = false
        newline_indent(io, indent, depth + 1)
        write_json(io, v, indent, depth + 1)
    end
    newline_indent(io, indent, depth)
    print(io, "]")
    return nothing
end

write_json(io::IO, x::Nothing, ::Int, ::Int) = print(io, "null")
write_json(io::IO, x::Missing, ::Int, ::Int) = print(io, "null")
write_json(io::IO, x::Bool, ::Int, ::Int) = print(io, x ? "true" : "false")
write_json(io::IO, x::Integer, ::Int, ::Int) = print(io, string(x))
write_json(io::IO, x::AbstractString, ::Int, ::Int) = write_json_string(io, String(x))
write_json(io::IO, x::Symbol, ::Int, ::Int) = write_json_string(io, String(x))
write_json(io::IO, x::Enum, ::Int, ::Int) = write_json_string(io, string(x))

function write_json(io::IO, x::AbstractFloat, ::Int, ::Int)
    if isnan(x) || isinf(x)
        # JSON has no NaN/Inf.  We encode as null and rely on the schema layer to
        # distinguish "unknown" from "zero" -- see WorldModel: NaN == unmeasured.
        print(io, "null")
    else
        print(io, string(Float64(x)))
    end
    return nothing
end

# Fallback: structs are serialised field-wise.  Keeps the LLM boundary total.
function write_json(io::IO, x::T, indent::Int, depth::Int) where {T}
    fns = fieldnames(T)
    if isempty(fns)
        write_json_string(io, string(x))
        return nothing
    end
    d = Dict{String,Any}(String(f) => getfield(x, f) for f in fns)
    write_json(io, d, indent, depth)
    return nothing
end

function newline_indent(io::IO, indent::Int, depth::Int)
    indent > 0 || return nothing
    print(io, "\n")
    for _ in 1:(indent*depth)
        print(io, " ")
    end
    return nothing
end

function write_json_string(io::IO, s::String)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' '
            print(io, "\\u", lpad(string(UInt32(c), base = 16), 4, '0'))
        else
            print(io, c)
        end
    end
    print(io, '"')
    return nothing
end

end # module
