"""
    CodeGeneration

Template-driven Julia synthesis, allowlisted AST rewriting, and static validation
(spec sections 12, 14, 17, 21, 30).

Three properties are enforced structurally.

1. **The LLM does not write text that gets compiled.**  It selects a template plus a
   subset of registered `TRANSFORMS`.  A free-form path exists (`validate_source`)
   but is off by default, because a validator over free-form Julia is a blocklist in
   disguise and blocklists over a language with `eval`, `ccall`, `@generated` and
   macro expansion do not hold.
2. **Validation is an allowlist over the AST**, evaluated *before* macro expansion,
   and rejects any head, macro, or callee not explicitly permitted.
3. **`@inbounds` requires a discharged proof obligation.**  It is only accepted when
   the enclosing function carries a generated axes guard, because `@inbounds` on
   unverified index arithmetic converts a bug into arbitrary memory access.  This is
   the one transform that can turn a wrong answer into a compromised host.
"""
module CodeGeneration

using Printf
using ..Schema

export ValidationReport, validate, validate_source, generate, generate_source,
       apply_transforms, applicable_transforms, ALLOWED_CALLS, ALLOWED_MACROS,
       ALLOWED_HEADS, TEMPLATES, GPU_TEMPLATES, prelude, entrypoint_name

# ------------------------------------------------------------- allowlisting --

const ALLOWED_HEADS = Set{Symbol}([
    :function, :(=), :call, :block, :if, :elseif, :return, :for, :while, :ref,
    :tuple, :vect, :(->), :comparison, :(&&), :(||), :(::), :parameters, :kw,
    :macrocall, :let, :local, :generator, :curly, :where, :(...), :(+=), :(-=),
    :(*=), :(/=), :do, :break, :continue, :quote, :string, :typed_vcat, :hcat,
    :vcat, :(:), :(.), :toplevel, :const, :meta, :inert,
])

const ALLOWED_MACROS = Set{Symbol}([
    Symbol("@inbounds"), Symbol("@simd"), Symbol("@views"), Symbol("@view"),
    Symbol("@fastmath"), Symbol("@inline"), Symbol("@noinline"), Symbol("@assert"),
    Symbol("@boundscheck"), Symbol("@kernel"), Symbol("@index"), Symbol("@uniform"),
    Symbol("@cuda"), Symbol("@sync"), Symbol("@threads"), Symbol("@spawn"),
])

"""
Callee allowlist.  Anything not here, and not defined inside the generated unit
itself, is a validation error.  Notably absent: `eval`, `include`, `open`, `run`,
`ccall`, `pointer`, `unsafe_*`, `Base.invokelatest`, `download`, `read`, `write`.
"""
const ALLOWED_CALLS = Set{Symbol}([
    # arithmetic / math
    :+, :-, :*, :/, :^, :div, :rem, :mod, :fld, :cld, :muladd, :fma, :abs, :abs2,
    :sqrt, :cbrt, :exp, :log, :log2, :log10, :sin, :cos, :tan, :atan, :tanh,
    :max, :min, :clamp, :sign, :floor, :ceil, :round, :inv, :hypot, :zero, :one,
    :isnan, :isinf, :isfinite, :iszero, :signbit, :nextpow, :prevpow, :ldexp,
    # comparison / logic
    :(==), :(!=), :(<), :(<=), :(>), :(>=), :!, :xor, :ifelse, :isequal, :isless,
    # containers / iteration
    :length, :size, :axes, :eachindex, :firstindex, :lastindex, :ndims, :eltype,
    :similar, :zeros, :ones, :fill, :fill!, :copy, :copyto!, :collect, :reshape,
    :view, :reverse, :sort, :sort!, :partialsort!, :searchsortedfirst, :getindex,
    :setindex!, :push!, :append!, :empty!, :resize!, :first, :last, :vec,
    :enumerate, :zip, :range, :LinRange, :stride, :strides, :parent, :count,
    :isempty, :checkbounds, :convert, :promote_type, :float, :typemax, :typemin,
    :eps, :real, :imag, :Tuple, :Vector, :Matrix, :Array, :BitVector, :Int, :Bool,
    :Float64, :Float32, :Float16, :UInt8, :Int32, :Int64, :AbstractFloat, :Base,
    Symbol(":"),
    # reductions / stats
    :sum, :prod, :maximum, :minimum, :extrema, :cumsum, :any, :all, :findall,
    :findfirst, :mean, :median, :median!, :std, :var, :quantile, :norm, :dot,
    :mul!, :transpose, :adjoint, :diagm, :I,
    # threading / gpu helpers that the templates may emit
    :nthreads, :threadid, :get_backend, :allocate, :synchronize, :KernelAbstractions,
    :CuArray, :Array, :cu, :adapt,
    # internal generated helpers (defined in `prelude`)
    :_axes_guard, :_dims_guard, :_chunk_ranges,
    # controlled failure
    :throw, :ArgumentError, :DimensionMismatch, :BoundsError,
])

"Qualified names the generated code may reference, as `Module.name` pairs."
const ALLOWED_QUALIFIED = Set{Tuple{Symbol,Symbol}}([
    (:Base, :length), (:Base, :size), (:Base, :axes), (:Base, :eachindex),
    (:LinearAlgebra, :mul!), (:LinearAlgebra, :norm), (:LinearAlgebra, :dot),
    (:Statistics, :mean), (:Statistics, :median), (:Statistics, :std),
    (:Threads, :nthreads), (:Threads, :threadid), (:Threads, Symbol("@threads")),
    (:KernelAbstractions, :get_backend), (:KernelAbstractions, :allocate),
    (:CUDA, :synchronize), (:CUDA, :CuArray), (:CUDA, Symbol("@sync")),
])

struct ValidationReport
    ok::Bool
    errors::Vector{String}
    warnings::Vector{String}
    defined::Vector{Symbol}
    n_nodes::Int
end

Base.show(io::IO, r::ValidationReport) = print(io,
    "ValidationReport(", r.ok ? "ok" : "REJECTED", ", ", r.n_nodes, " nodes, ",
    length(r.errors), " errors, ", length(r.warnings), " warnings)")

mutable struct VCtx
    errors::Vector{String}
    warnings::Vector{String}
    defined::Set{Symbol}
    locals::Set{Symbol}
    nodes::Int
    guarded::Bool
    allow_fastmath::Bool
    allow_inbounds::Bool
end

"""
    validate(expr; allow_fastmath=false, allow_inbounds=true) -> ValidationReport

Static allowlist check over the *unexpanded* AST.
"""
function validate(ex; allow_fastmath::Bool = false, allow_inbounds::Bool = true)
    ctx = VCtx(String[], String[], Set{Symbol}(), Set{Symbol}(), 0, false,
               allow_fastmath, allow_inbounds)
    collect_definitions!(ctx, ex)
    walk!(ctx, ex)
    return ValidationReport(isempty(ctx.errors), ctx.errors, ctx.warnings,
                            sort(collect(ctx.defined)), ctx.nodes)
end

"""
    validate_source(src) -> ValidationReport

Parses text and validates it.  Used only for the (disabled-by-default) free-form
path; parse failure is itself a rejection.
"""
function validate_source(src::AbstractString; kwargs...)
    ex = try
        Meta.parseall(String(src))
    catch err
        return ValidationReport(false, ["parse error: $(err)"], String[], Symbol[], 0)
    end
    return validate(ex; kwargs...)
end

function collect_definitions!(ctx::VCtx, ex)
    ex isa Expr || return nothing
    if ex.head === :function || (ex.head === :(=) && ex.args[1] isa Expr &&
                                 ex.args[1].head === :call)
        sig = ex.args[1]
        while sig isa Expr && sig.head === :where
            sig = sig.args[1]
        end
        if sig isa Expr && sig.head === :call
            name = sig.args[1]
            name isa Symbol && push!(ctx.defined, name)
        end
    end
    for a in ex.args
        collect_definitions!(ctx, a)
    end
    return nothing
end

err!(ctx::VCtx, msg::AbstractString) = push!(ctx.errors, String(msg))
warn!(ctx::VCtx, msg::AbstractString) = push!(ctx.warnings, String(msg))

const FORBIDDEN_NAMES = Set{Symbol}([
    :eval, :include, :ccall, :cglobal, :pointer, :unsafe_load, :unsafe_store!,
    :unsafe_wrap, :unsafe_convert, :unsafe_pointer_to_objref, :run, :open, :read,
    :write, :download, :readdir, :rm, :mv, :cp, :touch, :mkpath, :chmod, :setenv,
    :Core, :Main, :Libc, :Libdl, :invokelatest, :parse, :Meta, :getfield,
    :setfield!, :getproperty, :setproperty!, :task_local_storage, :atexit,
    :systemerror, :exit, :ENV, :Base64, :Sockets, :Distributed, :addprocs,
])

function walk!(ctx::VCtx, ex)
    ctx.nodes += 1
    if ex isa Symbol
        ex in FORBIDDEN_NAMES && err!(ctx, "forbidden identifier '$(ex)'")
        return nothing
    end
    (ex isa Expr) || return nothing   # literals, LineNumberNode, QuoteNode payloads
    if !(ex.head in ALLOWED_HEADS)
        err!(ctx, "forbidden expression head ':$(ex.head)'")
        return nothing
    end
    if ex.head === :call
        f = ex.args[1]
        check_callee!(ctx, f)
        for a in ex.args[2:end]
            walk!(ctx, a)
        end
        return nothing
    elseif ex.head === :macrocall
        m = ex.args[1]
        mm = m isa Expr && m.head === :. ? m.args[end] : m
        mname = mm isa QuoteNode ? mm.value : mm
        if !(mname isa Symbol) || !(mname in ALLOWED_MACROS)
            err!(ctx, "forbidden macro '$(mname)'")
            return nothing
        end
        if mname === Symbol("@fastmath") && !ctx.allow_fastmath
            err!(ctx, "@fastmath is not permitted by the active policy")
        end
        if mname === Symbol("@inbounds")
            if !ctx.allow_inbounds
                err!(ctx, "@inbounds is not permitted by the active policy")
            elseif !ctx.guarded
                err!(ctx, "@inbounds without a discharged axes guard (_axes_guard) " *
                          "in the enclosing unit")
            end
        end
        for a in ex.args[2:end]
            walk!(ctx, a)
        end
        return nothing
    elseif ex.head === :.
        base = ex.args[1]
        prop = ex.args[2]
        pname = prop isa QuoteNode ? prop.value : prop
        if base isa Symbol && pname isa Symbol
            if !((base, pname) in ALLOWED_QUALIFIED)
                err!(ctx, "forbidden qualified reference '$(base).$(pname)'")
            end
            return nothing
        end
        # broadcast fusion `f.(x)` also lands here in some forms; walk children
        for a in ex.args
            walk!(ctx, a)
        end
        return nothing
    elseif ex.head === :quote || ex.head === :inert
        err!(ctx, "quoting/interpolation is not permitted in generated code")
        return nothing
    end
    for a in ex.args
        walk!(ctx, a)
    end
    return nothing
end

function check_callee!(ctx::VCtx, f)
    if f isa Symbol
        if f in FORBIDDEN_NAMES
            err!(ctx, "forbidden call to '$(f)'")
        elseif !(f in ALLOWED_CALLS) && !(f in ctx.defined)
            err!(ctx, "call to non-allowlisted function '$(f)'")
        end
        (f === :_axes_guard || f === :_dims_guard) && (ctx.guarded = true)
    elseif f isa Expr && f.head === :.
        base, prop = f.args[1], f.args[2]
        pname = prop isa QuoteNode ? prop.value : prop
        if !(base isa Symbol && pname isa Symbol && (base, pname) in ALLOWED_QUALIFIED)
            err!(ctx, "forbidden qualified call '$(f)'")
        end
    elseif f isa Expr && f.head === :curly
        check_callee!(ctx, f.args[1])
    else
        err!(ctx, "non-symbolic callee '$(f)' rejected")
    end
    return nothing
end

# ------------------------------------------------------------ AST transforms --

"""
Each transform is `(guard, rewrite)`.  `guard(alg)` says whether the transform is
meaningful for that template; applying a transform outside its guard is a
generation error, not a silent no-op, so the planner gets honest feedback.
"""
const TRANSFORM_APPLICABILITY = Dict{Symbol,Set{Symbol}}(
    :zscore_anomaly => Set([:inbounds, :simd, :views, :fma, :fastmath, :inplace, :hoist]),
    :mad_anomaly    => Set([:inbounds, :views, :inplace, :hoist]),
    :matmul         => Set([:inbounds, :simd, :fma, :fastmath, :tile, :hoist]),
    :sum_reduction  => Set([:inbounds, :simd, :fastmath, :hoist]),
    :stencil3       => Set([:inbounds, :simd, :views, :fma, :fastmath, :inplace]),
)

applicable_transforms(alg::Symbol) = get(TRANSFORM_APPLICABILITY, alg, Set{Symbol}())

"Wrap every `for` body in `@inbounds` (only emitted when a guard is present)."
function xf_inbounds(ex)
    return maprewrite(ex) do e
        (e isa Expr && e.head === :for) || return e
        body = e.args[2]
        return Expr(:for, e.args[1], Expr(:block, Expr(:macrocall,
                    Symbol("@inbounds"), LineNumberNode(0), body)))
    end
end

"Attach `@simd` to innermost `for` loops (no nested `for` in the body)."
function xf_simd(ex)
    return maprewrite(ex) do e
        (e isa Expr && e.head === :for) || return e
        has_inner_for(e.args[2]) && return e
        return Expr(:macrocall, Symbol("@simd"), LineNumberNode(0), e)
    end
end

xf_views(ex) = Expr(:macrocall, Symbol("@views"), LineNumberNode(0), ex)
xf_fastmath(ex) = Expr(:macrocall, Symbol("@fastmath"), LineNumberNode(0), ex)

"Contract `a*b + c` into `muladd(a, b, c)`; changes rounding, so it is reported."
function xf_fma(ex)
    return maprewrite(ex) do e
        (e isa Expr && e.head === :call && length(e.args) == 3 && e.args[1] === :+) ||
            return e
        l, r = e.args[2], e.args[3]
        if l isa Expr && l.head === :call && l.args[1] === :* && length(l.args) == 3
            return Expr(:call, :muladd, l.args[2], l.args[3], r)
        elseif r isa Expr && r.head === :call && r.args[1] === :* && length(r.args) == 3
            return Expr(:call, :muladd, r.args[2], r.args[3], l)
        end
        return e
    end
end

"Hoist `length(x)`/`size(x,d)` out of loop bodies into a `let`-bound local."
function xf_hoist(ex)
    return maprewrite(ex) do e
        (e isa Expr && e.head === :for) || return e
        hoisted = Expr[]
        seen = Dict{String,Symbol}()
        body = maprewrite(e.args[2]) do inner
            (inner isa Expr && inner.head === :call &&
             (inner.args[1] === :length || inner.args[1] === :size)) || return inner
            k = string(inner)
            if !haskey(seen, k)
                s = Symbol("_hoist_", length(seen) + 1)
                seen[k] = s
                push!(hoisted, Expr(:(=), s, inner))
            end
            return seen[k]
        end
        isempty(hoisted) && return e
        return Expr(:block, hoisted..., Expr(:for, e.args[1], body))
    end
end

has_inner_for(ex) = ex isa Expr &&
    (ex.head === :for || any(has_inner_for, ex.args))

"Bottom-up rewrite: children first, then the node itself."
function maprewrite(f, ex)
    if ex isa Expr
        args = Any[maprewrite(f, a) for a in ex.args]
        return f(Expr(ex.head, args...))
    end
    return f(ex)
end

const TRANSFORM_FUNCS = Dict{Symbol,Function}(
    :inbounds => xf_inbounds, :simd => xf_simd, :views => xf_views,
    :fastmath => xf_fastmath, :fma => xf_fma, :hoist => xf_hoist,
)

"""
    apply_transforms(ex, alg, transforms) -> (Expr, notes)

Order matters: hoist and fma rewrite the loop body, simd wraps the loop, inbounds
wraps the body, and `@views`/`@fastmath` wrap the whole unit last.
"""
function apply_transforms(ex, alg::Symbol, transforms::Vector{Symbol})
    notes = String[]
    ok = applicable_transforms(alg)
    order = [:hoist, :fma, :inbounds, :simd, :views, :fastmath]
    out = ex
    for t in order
        t in transforms || continue
        if !(t in ok)
            push!(notes, "transform :$(t) is not applicable to :$(alg); refused")
            continue
        end
        f = get(TRANSFORM_FUNCS, t, nothing)
        if f === nothing
            push!(notes, "transform :$(t) has no rewrite rule (template-level only)")
            continue
        end
        out = f(out)
        push!(notes, "applied :$(t)")
        t === :fma && push!(notes, "NOTE: :fma changes rounding; verification tolerance widened")
        t === :fastmath && push!(notes, "NOTE: :fastmath relaxes IEEE semantics; requires policy grant")
    end
    for t in transforms
        (t in (:inplace, :tile)) && push!(notes, "transform :$(t) handled at template level")
    end
    return out, notes
end

# ---------------------------------------------------------------- templates --

"""
Helpers injected ahead of every generated unit.  `_axes_guard` is the proof
obligation that `@inbounds` depends on: it *runtime-checks* the shape relationship
that the elided bounds checks would have caught.
"""
function prelude()
    return """
    # --- generated prelude (trusted, fixed text) ---
    # Discharges the proof obligation that `@inbounds` relies on:
    #   (a) every array has identical axes, and
    #   (b) indexing is 1-based, so literal ranges like `2:(n-1)` are in-bounds.
    # Cost is O(#arrays); it runs once per call, never inside a loop.
    @inline function _axes_guard(xs...)
        a = axes(xs[1])
        first(a[1]) == 1 || throw(DimensionMismatch("axes guard: not 1-based"))
        for x in xs
            axes(x) == a || throw(DimensionMismatch("axes guard: shape mismatch"))
        end
        return a
    end

    # Matrix form: C[i,j] += A[i,p]*B[p,j] is in-bounds for the literal loop nests
    # emitted by the matmul template exactly when these relations hold.
    @inline function _dims_guard(A, B, C)
        (first(axes(A, 1)) == 1 && first(axes(A, 2)) == 1 &&
         first(axes(B, 1)) == 1 && first(axes(B, 2)) == 1 &&
         first(axes(C, 1)) == 1 && first(axes(C, 2)) == 1) ||
            throw(DimensionMismatch("dims guard: not 1-based"))
        size(A, 2) == size(B, 1) || throw(DimensionMismatch("inner dimension"))
        size(C, 1) == size(A, 1) || throw(DimensionMismatch("C rows"))
        size(C, 2) == size(B, 2) || throw(DimensionMismatch("C cols"))
        return (size(A, 1), size(A, 2), size(B, 2))
    end

    @inline function _chunk_ranges(n::Int, chunk::Int)
        chunk > 0 || throw(ArgumentError("chunk must be positive"))
        return [i:min(i + chunk - 1, n) for i in 1:chunk:n]
    end
    """
end

entrypoint_name(alg::Symbol) = string("generated_", alg)

"""
Template registry.  `TEMPLATES[alg](T, backend, params) -> Expr`.

Every template is generic over `T<:AbstractFloat` and over the array type, so that
Multiple Dispatch -- not a code branch -- selects CPU vs GPU (spec section 13).
"""
const TEMPLATES = Dict{Symbol,Function}()

"""
GPU templates, registered by the KernelAbstractions extension.  Kept separate from
`TEMPLATES` because a scalar-indexed CPU loop is not merely slow on a `CuArray`, it
is rejected (scalar indexing is disallowed): silently reusing the CPU template for
`:cuda` would produce code that cannot run.  An empty registry therefore means
"GPU code generation is unavailable", which the planner can act on.
"""
const GPU_TEMPLATES = Dict{Symbol,Function}()

function tmpl_zscore(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(entrypoint_name(:zscore_anomaly))
    threshold = get(params, "threshold_milli", 3000) / 1000
    body = quote
        function $(fname)(x::AbstractVector{$T}, out::AbstractVector{Bool})
            _axes_guard(x, out)
            n = length(x)
            n > 1 || throw(ArgumentError("need at least two samples"))
            s = zero($T)
            for i in eachindex(x)
                s = s + x[i]
            end
            mu = s / n
            acc = zero($T)
            for i in eachindex(x)
                d = x[i] - mu
                acc = acc + d * d
            end
            sigma = sqrt(acc / (n - 1))
            thr = $(T)($threshold) * sigma
            c = 0
            for i in eachindex(x)
                flag = abs(x[i] - mu) > thr
                out[i] = flag
                c = c + ifelse(flag, 1, 0)
            end
            return c
        end
    end
    return Base.remove_linenums!(body)
end

function tmpl_mad(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(entrypoint_name(:mad_anomaly))
    threshold = get(params, "threshold_milli", 3000) / 1000
    body = quote
        function $(fname)(x::AbstractVector{$T}, out::AbstractVector{Bool},
                          scratch::AbstractVector{$T})
            _axes_guard(x, out, scratch)
            n = length(x)
            n > 1 || throw(ArgumentError("need at least two samples"))
            copyto!(scratch, x)
            med = median!(scratch)
            for i in eachindex(x)
                scratch[i] = abs(x[i] - med)
            end
            mad = median!(scratch)
            scale = $(T)(1.4826) * mad
            thr = $(T)($threshold) * scale
            c = 0
            for i in eachindex(x)
                flag = abs(x[i] - med) > thr
                out[i] = flag
                c = c + ifelse(flag, 1, 0)
            end
            return c
        end
    end
    return Base.remove_linenums!(body)
end

function tmpl_matmul(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(entrypoint_name(:matmul))
    tile = get(params, "tile", 0)
    if tile > 0
        body = quote
            function $(fname)(A::AbstractMatrix{$T}, B::AbstractMatrix{$T},
                              C::AbstractMatrix{$T})
                m = size(A, 1)
                k = size(A, 2)
                n = size(B, 2)
                _dims_guard(A, B, C)
                fill!(C, zero($T))
                ts = $tile
                for jj in 1:ts:n
                    for kk in 1:ts:k
                        for j in jj:min(jj + ts - 1, n)
                            for p in kk:min(kk + ts - 1, k)
                                b = B[p, j]
                                for i in 1:m
                                    C[i, j] = C[i, j] + A[i, p] * b
                                end
                            end
                        end
                    end
                end
                return C
            end
        end
    else
        body = quote
            function $(fname)(A::AbstractMatrix{$T}, B::AbstractMatrix{$T},
                              C::AbstractMatrix{$T})
                m = size(A, 1)
                k = size(A, 2)
                n = size(B, 2)
                _dims_guard(A, B, C)
                fill!(C, zero($T))
                for j in 1:n
                    for p in 1:k
                        b = B[p, j]
                        for i in 1:m
                            C[i, j] = C[i, j] + A[i, p] * b
                        end
                    end
                end
                return C
            end
        end
    end
    return Base.remove_linenums!(body)
end

function tmpl_sum(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(entrypoint_name(:sum_reduction))
    body = quote
        function $(fname)(x::AbstractVector{$T})
            _axes_guard(x)
            n = length(x)
            s = zero($T)
            c = zero($T)
            for i in eachindex(x)
                y = x[i] - c
                t = s + y
                c = (t - s) - y
                s = t
            end
            return s
        end
    end
    return Base.remove_linenums!(body)
end

function tmpl_stencil(T::Symbol, backend::Symbol, params::Dict{String,Int})
    fname = Symbol(entrypoint_name(:stencil3))
    body = quote
        function $(fname)(x::AbstractVector{$T}, out::AbstractVector{$T})
            _axes_guard(x, out)
            n = length(x)
            n >= 3 || throw(ArgumentError("stencil needs n >= 3"))
            third = $(T)(1) / $(T)(3)
            out[1] = x[1]
            out[n] = x[n]
            for i in 2:(n - 1)
                out[i] = (x[i - 1] + x[i] + x[i + 1]) * third
            end
            return out
        end
    end
    return Base.remove_linenums!(body)
end

TEMPLATES[:zscore_anomaly] = tmpl_zscore
TEMPLATES[:mad_anomaly]    = tmpl_mad
TEMPLATES[:matmul]         = tmpl_matmul
TEMPLATES[:sum_reduction]  = tmpl_sum
TEMPLATES[:stencil3]       = tmpl_stencil

struct GeneratedUnit
    candidate::Schema.Candidate
    expr::Expr
    source::String
    notes::Vector{String}
    report::ValidationReport
end

"""
    generate(candidate; allow_fastmath, allow_inbounds) -> GeneratedUnit

Template -> transforms -> validation.  The returned unit is *never* `eval`ed here;
`Sandbox` runs it in a separate OS process.
"""
function generate(c::Schema.Candidate; allow_fastmath::Bool = false,
                  allow_inbounds::Bool = true)
    gpu = Schema.is_gpu_backend(c.backend)
    registry = gpu ? GPU_TEMPLATES : TEMPLATES
    if !haskey(registry, c.algorithm)
        throw(Schema.SchemaError(gpu ?
            "no GPU template for '$(c.algorithm)'; load KernelAbstractions.jl (and " *
            "CUDA.jl) to enable GPU code generation" :
            "no template for algorithm '$(c.algorithm)'"))
    end
    T = c.precision === :BFloat16 ? :Float32 : c.precision
    ex = registry[c.algorithm](T, c.backend, c.params)
    ex2, notes = apply_transforms(ex, c.algorithm, c.transforms)
    ex2 isa Expr || (ex2 = Expr(:block, ex2))
    report = validate(ex2; allow_fastmath = allow_fastmath,
                      allow_inbounds = allow_inbounds)
    src = string(prelude(), "\n", string(ex2))
    return GeneratedUnit(c, ex2, src, notes, report)
end

generate_source(c::Schema.Candidate; kwargs...) = generate(c; kwargs...).source

export GeneratedUnit

end # module
