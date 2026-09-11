"""
Benchmark jroots on the sparse and dense polynomial test suites.

Set the fields in `Config` at the bottom of this file, then run
``julia run_jroots_combined.jl``. One script covers the three variants that used
to live in separate files:

    suite="sparse", backend="multipower"   sparse terms expanded to a dense
                                           coefficient tensor, solved as MultiPower
    suite="sparse", backend="lambda"       sparse terms evaluated by a closure
    suite="dense",  backend="multipower"   dense coefficient tensors (Mathematica
                                           JSON or .npy), solved as MultiPower

For every degree in [mindeg, maxdeg] it solves num_tests systems of dim
polynomials on [-1, 1]^dim and writes, under the run's base directory
(``<result_dir>/dim{dim}/nonzero{nonzero}`` for sparse,
``<result_dir>/dim{dim}`` for dense):

    avg_times.txt              average solve time, one value per degree
    max_resids.txt             max |f_i(root)|, one value per degree
    avg_resids.txt             mean |f_i(root)|, one value per degree
    sum_resids.txt             sum of |f_i(root)|, one value per degree
    roots/roots_deg{deg}.json  {test number: roots} for that degree

The .txt files hold one plain value per line in degree order; the JSON files
match the layout the Python benchmark writes, so the two can be diffed directly.

Output names can be overridden three ways, from least to most specific:

    result_dir      moves the whole run somewhere else
    file_prefix     prepended to every output filename, e.g. "dim5_avg_times.txt"
    summary_files   exact path per summary, e.g. Dict("avg_times" => "times.txt")
    roots_file      exact path for the roots JSON; {deg} and {dim} are substituted

Overridden paths are used verbatim — result_dir, file_prefix and the dim/nonzero
folders are not applied to them. Missing parent folders are created either way.
"""

# Path to the solver source. Adjust to wherever your checkout lives.
const SOLVER_PATH = "../../hyun/Julia-Rootfinding/src/CombinedSolver.jl"
include(SOLVER_PATH)

using NPZ
using JSON3
using Statistics
using Printf

const SUMMARIES = ("avg_times", "max_resids", "avg_resids", "sum_resids")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

Base.@kwdef mutable struct Config
    suite::String = "dense"            # "sparse" | "dense"
    backend::String = "multipower"     # "multipower" | "lambda"  (lambda: sparse only)
    dim::Int = 5
    mindeg::Int = 2
    maxdeg::Int = 15
    nonzero::Int = 3                   # sparse only: terms per polynomial
    num_tests::Int = 100
    dense_format::String = "npy"       # dense only: "json" (Mathematica .txt) | "npy"
    coeff_dir::Union{String,Nothing} = nothing   # defaults per suite
    result_dir::Union{String,Nothing} = "../$(suite)/results/jroots_results2"
    append::Bool = false               # true to add degrees to an existing run
    warmup::Int = 10                   # untimed solves at mindeg to absorb JIT cost
    zero_constant_term::Bool = false   # dense only: force the constant coefficient to 0
    file_prefix::String = ""           # prepended to every output filename
    # Exact output paths, bypassing base_dir entirely. Keys of summary_files must be
    # among SUMMARIES; roots_file may contain {deg} and {dim} placeholders.
    summary_files::Union{Nothing,Dict{String,String}} = nothing
    roots_file::Union{Nothing,String} = nothing
end

"""Fill in the per-suite defaults and validate. Mirrors the Python __post_init__."""
function resolve!(cfg::Config)
    cfg.suite in ("sparse", "dense") || error("unknown suite: $(cfg.suite)")
    cfg.backend in ("multipower", "lambda") || error("unknown backend: $(cfg.backend)")
    if cfg.suite == "dense" && cfg.backend == "lambda"
        error("the lambda backend expects sparse term lists")
    end
    if cfg.coeff_dir === nothing
        if cfg.suite == "sparse"
            cfg.coeff_dir = "../sparse/coeffs"
        elseif cfg.dense_format == "json"
            cfg.coeff_dir = "../dense/coeffs_mathematica_json"
        else
            cfg.coeff_dir = "../dense/coeffs"
        end
    end
    if cfg.result_dir === nothing
        cfg.result_dir = "../$(cfg.suite)/results/jroots_results"
    end
    if cfg.summary_files !== nothing
        for name in keys(cfg.summary_files)
            name in SUMMARIES || error("unknown summary file: $(name)")
        end
    end
    return cfg
end

"""Where this run's summaries and roots/ folder live."""
function base_dir(cfg::Config)
    d = joinpath(cfg.result_dir, "dim$(cfg.dim)")
    return cfg.suite == "sparse" ? joinpath(d, "nonzero$(cfg.nonzero)") : d
end

"""Where one summary goes: the explicit override if given, else
<base_dir>/<file_prefix><name>.txt."""
function summary_path(cfg::Config, name::AbstractString)
    if cfg.summary_files !== nothing && haskey(cfg.summary_files, name)
        return cfg.summary_files[name]
    end
    return joinpath(base_dir(cfg), "$(cfg.file_prefix)$(name).txt")
end

"""Where one degree's roots JSON goes: the explicit override if given (with {deg} and
{dim} filled in), else <base_dir>/roots/<file_prefix>roots_deg{deg}.json."""
function roots_path(cfg::Config, deg::Int)
    if cfg.roots_file !== nothing
        return replace(cfg.roots_file, "{deg}" => string(deg), "{dim}" => string(cfg.dim))
    end
    return joinpath(base_dir(cfg), "roots", "$(cfg.file_prefix)roots_deg$(deg).json")
end

"""Create the containing folder for a path, if it has one."""
function ensure_dir(path::AbstractString)
    d = dirname(path)
    isempty(d) || mkpath(d)
    return path
end


# ----------------------------------------------------------------------------
# Loading coefficients
# ----------------------------------------------------------------------------

"""Turn JSON3's nested arrays into a plain Array, preserving the logical shape."""
function flatten_nested!(out::Vector{Float64}, x)
    if x isa AbstractVector
        for e in x
            flatten_nested!(out, e)
        end
    else
        push!(out, Float64(x))
    end
    return out
end

function json_to_array(data)
    dims = Int[]
    node = data
    while node isa AbstractVector
        push!(dims, length(node))
        node = node[1]
    end
    flat = flatten_nested!(Float64[], data)
    # JSON nests row-major; reshape column-major with reversed dims, then transpose.
    return permutedims(reshape(flat, reverse(dims)...), length(dims):-1:1)
end

"""
Load every test for one degree.

sparse -> (n_tests, dim, n_terms, dim+1); the last axis holds an exponent
per dimension followed by the coefficient.
dense  -> (n_tests, dim, deg+1, ..., deg+1) coefficient tensors.
"""
function load_degree(cfg::Config, deg::Int)
    if cfg.suite == "sparse"
        return npzread(joinpath(cfg.coeff_dir, "dim$(cfg.dim)", "deg$(deg)",
                                "num$(cfg.nonzero).npy"))
    end
    if cfg.dense_format == "json"
        return json_to_array(JSON3.read(read(
            joinpath(cfg.coeff_dir, "dim$(cfg.dim)_deg$(deg)_randn.txt"), String)))
    end
    return npzread(joinpath(cfg.coeff_dir, "dim$(cfg.dim)_deg$(deg)_randn.npy"))
end


# ----------------------------------------------------------------------------
# Building the system for one test
# ----------------------------------------------------------------------------

"""Expand one test's sparse terms into a dense coefficient tensor for one function."""
function coeff_matrix(coeff_data, dim::Int, deg::Int, test_num::Int, func_num::Int)
    coeffs = zeros(ntuple(_ -> deg + 1, dim))
    coordinates = coeff_data[test_num, func_num, :, 1:dim]
    values = coeff_data[test_num, func_num, :, dim + 1]
    for i in axes(coordinates, 1)
        idx = Tuple(Int.(coordinates[i, :]) .+ 1)   # numpy is 0-based, Julia is 1-based
        coeffs[idx...] = values[i]
    end
    return coeffs
end

"""Pull one dense coefficient tensor out of the (n_tests, dim, deg+1, ...) block."""
function dense_tensor(cfg::Config, coeff_data, test_num::Int, func_num::Int)
    c = coeff_data[test_num, func_num, ntuple(_ -> Colon(), cfg.dim)...]
    if cfg.zero_constant_term
        c[ntuple(_ -> 1, cfg.dim)...] = 0
    end
    return c
end

"""Build a closure evaluating one sparse polynomial, called as poly(x...)."""
function poly_from_terms(coeff_data, dim::Int, test_num::Int, func_num::Int)
    exps = Int.(coeff_data[test_num, func_num, :, 1:dim])
    coeffs = Float64.(coeff_data[test_num, func_num, :, dim + 1])

    return function (x...)
        @assert length(x) == dim
        s = 0.0
        @inbounds for k in axes(exps, 1)
            m = 1.0
            for j in 1:dim
                ej = exps[k, j]
                if ej != 0
                    m *= x[j]^ej
                end
            end
            s += coeffs[k] * m
        end
        return s
    end
end

"""Turn one test's coefficient block into a list of dim callables."""
function build_funcs(cfg::Config, coeff_data, test_num::Int, deg::Int)
    if cfg.backend == "lambda"
        return [poly_from_terms(coeff_data, cfg.dim, test_num, j) for j in 1:cfg.dim]
    end
    if cfg.suite == "sparse"
        return [MultiPower(to_python(coeff_matrix(coeff_data, cfg.dim, deg, test_num, j)))
                for j in 1:cfg.dim]
    end
    return [MultiPower(to_python(dense_tensor(cfg, coeff_data, test_num, j)))
            for j in 1:cfg.dim]
end


# ----------------------------------------------------------------------------
# Solving and scoring
# ----------------------------------------------------------------------------

"""Push |value| onto all_res, whether the evaluation returned a scalar or an array."""
function push_abs!(all_res, value)
    if value isa AbstractArray
        append!(all_res, abs.(value))
    else
        push!(all_res, abs(value))
    end
    return all_res
end

"""|f_i(root_j)| for every function i and root j."""
function residuals!(all_res, cfg::Config, funcs, roots)
    for root in roots
        for idx in 1:cfg.dim
            if cfg.backend == "lambda"
                push_abs!(all_res, funcs[idx](root...))
            else
                push_abs!(all_res, eval_MultiPower(funcs[idx], root))
            end
        end
    end
    return all_res
end

"""Solve one system on [-1, 1]^dim, returning the roots and the elapsed seconds."""
function solve_test(funcs, dim::Int)
    ans = @timed solve(funcs, -ones(dim), ones(dim))
    return ans.value, ans.time
end


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

"""Write {test number: list of roots}, one root per line, sorted by test number."""
function write_roots_json(path, roots_by_test)
    open(path, "w") do f
        write(f, "{\n")
        items = sort(collect(roots_by_test), by = x -> x[1])
        for (i, (test_num, roots)) in enumerate(items)
            write(f, "  \"$(test_num)\": ")
            if isempty(roots)
                write(f, "[]")
            else
                write(f, "[\n")
                for (j, root) in enumerate(roots)
                    write(f, "    " * JSON3.write(root))
                    write(f, j < length(roots) ? ",\n" : "\n")
                end
                write(f, "  ]")
            end
            write(f, i < length(items) ? ",\n" : "\n")
        end
        write(f, "}\n")
    end
end

function append_value(path, value)
    open(path, "a") do f
        write(f, @sprintf("%.17g\n", value))
    end
end


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

function run_benchmark(cfg::Config)
    resolve!(cfg)

    summary_paths = Dict(name => ensure_dir(summary_path(cfg, name)) for name in SUMMARIES)
    if !cfg.append
        for path in values(summary_paths)
            write(path, "")
        end
    end

    for deg in cfg.mindeg:cfg.maxdeg
        println("--- $(cfg.suite) dim $(cfg.dim) degree $(deg)/$(cfg.maxdeg) ---")
        flush(stdout)
        tests = load_degree(cfg, deg)
        reps = min(cfg.num_tests, size(tests, 1))

        if deg == cfg.mindeg
            # Warm up Julia's JIT so it isn't charged to the first timed tests.
            println("  warming up")
            flush(stdout)
            for i in 1:min(cfg.warmup, reps)
                solve_test(build_funcs(cfg, tests, i, deg), cfg.dim)
            end
        end

        times = Float64[]
        all_res = Float64[]
        roots_by_test = Dict{Int,Any}()

        for i in 1:reps
            if i % 10 == 0
                println("  test $(i)/$(reps)")
                flush(stdout)
            end

            funcs = build_funcs(cfg, tests, i, deg)
            roots, elapsed = solve_test(funcs, cfg.dim)

            push!(times, elapsed)
            residuals!(all_res, cfg, funcs, roots)
            roots_by_test[i] = roots
        end

        write_roots_json(ensure_dir(roots_path(cfg, deg)), roots_by_test)

        append_value(summary_paths["avg_times"], isempty(times) ? NaN : mean(times))
        append_value(summary_paths["max_resids"], isempty(all_res) ? NaN : maximum(all_res))
        append_value(summary_paths["avg_resids"], isempty(all_res) ? NaN : mean(all_res))
        append_value(summary_paths["sum_resids"], isempty(all_res) ? NaN : sum(all_res))
    end

    println("Program Finished")
end


run_benchmark(Config())
