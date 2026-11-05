"""
Benchmark Telen's EigenvalueSolver (MTV) on the sparse and dense polynomial test suites.

Set the fields in `Config` at the bottom of this file, then run
``julia run_telen_combined.jl``. This replaces run_telen_sparse.jl and
run_telen_dense.jl:

    suite="sparse"   solved with EigenvalueSolver.solve_CI_mixed
    suite="dense"    solved with EigenvalueSolver.solve_CI_dense

EigenvalueSolver is not in the General registry; install it once with

    julia -e 'using Pkg; Pkg.add(url="https://github.com/simontelen/JuliaEigenvalueSolver.git");
              Pkg.add(["DynamicPolynomials", "NPZ", "JSON3"])'

The solver returns every root in C^dim, so after each solve the roots are filtered
down to those that are real (max |imag| <= imag_tol) and lie in the closed box
[-1, 1]^dim. For every degree in [mindeg, maxdeg] it solves num_tests systems and
writes

    avg_times, max_resids, avg_resids, sum_resids   one value per degree
    roots/roots_deg{deg}.json                       {test number: roots}
    failed_tests.txt                                tests where the solver threw

Sparse runs write these under ``<result_dir>/dim{dim}/nonzero{nonzero}/``. Dense
runs write the summaries as ``<result_dir>/dim{dim}_{name}.txt``, where
dense/graph_all.py reads them, and the roots under ``<result_dir>/dim{dim}/roots/``.
"""

using EigenvalueSolver
using DynamicPolynomials
using NPZ
using JSON3
using Statistics
using Printf
using LinearAlgebra: BLAS

const SUMMARIES = ("avg_times", "max_resids", "avg_resids", "sum_resids")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

Base.@kwdef mutable struct Config
    suite::String = "sparse"           # "sparse" | "dense"
    dim::Int = 5
    mindeg::Int = 2
    maxdeg::Int = 10
    nonzero::Int = 3                   # sparse only: terms per polynomial
    num_tests::Int = 100
    coeff_dir::Union{String,Nothing} = nothing   # defaults per suite
    result_dir::Union{String,Nothing} = nothing  # defaults to ../<suite>/results/telen_results
    append::Bool = false               # true to add degrees to an existing run
    warmup::Int = 10                   # untimed solves at mindeg to absorb JIT cost
    imag_tol::Union{Float64,Nothing} = nothing   # defaults: 1e-13 sparse, 1e-14 dense
    blas_threads::Int = 40             # the solver's linear algebra runs on BLAS threads
    zero_constant_term::Bool = true    # dense only: force the constant coefficient to 0,
                                       # as the MTV/Bertini/Chebfun/NSolve dense runs do
end

"""Fill in the per-suite defaults and validate."""
function resolve!(cfg::Config)
    cfg.suite in ("sparse", "dense") || error("unknown suite: $(cfg.suite)")
    if cfg.coeff_dir === nothing
        cfg.coeff_dir = cfg.suite == "sparse" ? "../sparse/coeffs" : "../dense/coeffs"
    end
    if cfg.result_dir === nothing
        cfg.result_dir = "../$(cfg.suite)/results/telen_results"
    end
    if cfg.imag_tol === nothing
        cfg.imag_tol = cfg.suite == "sparse" ? 1e-13 : 1e-14
    end
    return cfg
end

"""Where this run's roots/ folder and failure log live."""
function base_dir(cfg::Config)
    d = joinpath(cfg.result_dir, "dim$(cfg.dim)")
    return cfg.suite == "sparse" ? joinpath(d, "nonzero$(cfg.nonzero)") : d
end

function summary_path(cfg::Config, name)
    if cfg.suite == "sparse"
        return joinpath(base_dir(cfg), "$(name).txt")
    end
    return joinpath(cfg.result_dir, "dim$(cfg.dim)_$(name).txt")
end

roots_path(cfg::Config, deg) = joinpath(base_dir(cfg), "roots", "roots_deg$(deg).json")


# ----------------------------------------------------------------------------
# Loading coefficients
# ----------------------------------------------------------------------------

"""
Load every test for one degree.

sparse -> (n_tests, dim, n_terms, dim+1); the last axis holds an exponent
per dimension followed by the coefficient.
dense  -> (n_tests, dim, deg+1, ..., deg+1) coefficient tensors, where entry
[i_1, ..., i_dim] is the coefficient of x_1^(i_1-1) * ... * x_dim^(i_dim-1).
"""
function load_degree(cfg::Config, deg::Int)
    if cfg.suite == "sparse"
        return npzread(joinpath(cfg.coeff_dir, "dim$(cfg.dim)", "deg$(deg)",
                                "num$(cfg.nonzero).npy"))
    end
    return npzread(joinpath(cfg.coeff_dir, "dim$(cfg.dim)_deg$(deg)_randn.npy"))
end


# ----------------------------------------------------------------------------
# Building the system for one test
# ----------------------------------------------------------------------------

"""
One test's polynomials as term lists: for each function, a matrix of exponents
(one row per term) and a vector of coefficients, zero coefficients dropped. Both
suites reduce to this; it builds the solver's polynomials and evaluates residuals.
"""
function term_lists(cfg::Config, coeff_data, test_num::Int)
    dim = cfg.dim
    terms = Tuple{Matrix{Int},Vector{Float64}}[]
    for j in 1:dim
        if cfg.suite == "sparse"
            exps = Int.(coeff_data[test_num, j, :, 1:dim])
            coeffs = Float64.(coeff_data[test_num, j, :, dim + 1])
        else
            tensor = coeff_data[test_num, j, ntuple(_ -> Colon(), dim)...]
            cfg.zero_constant_term && (tensor[ntuple(_ -> 1, dim)...] = 0)
            idxs = [I for I in CartesianIndices(tensor) if tensor[I] != 0]
            exps = isempty(idxs) ? zeros(Int, 0, dim) :
                   reduce(vcat, [collect(Tuple(I))' .- 1 for I in idxs])
            coeffs = Float64[tensor[I] for I in idxs]
        end
        keep = coeffs .!= 0
        push!(terms, (exps[keep, :], coeffs[keep]))
    end
    return terms
end

"""Build DynamicPolynomials polynomials from term lists."""
function build_polys(terms, x)
    dim = length(x)
    return map(terms) do (exps, coeffs)
        isempty(coeffs) && return zero(x[1]) + 0.0
        sum(coeffs[k] * prod(x[l]^exps[k, l] for l in 1:dim) for k in eachindex(coeffs))
    end
end

"""Evaluate one polynomial, given as a term list, at a real point."""
function eval_terms(exps, coeffs, root)
    s = 0.0
    @inbounds for k in eachindex(coeffs)
        m = 1.0
        for l in eachindex(root)
            exps[k, l] != 0 && (m *= root[l]^exps[k, l])
        end
        s += coeffs[k] * m
    end
    return s
end


# ----------------------------------------------------------------------------
# Solving and scoring
# ----------------------------------------------------------------------------

"""Solve one system, returning every root the solver found and the elapsed seconds."""
function solve_test(cfg::Config, f, x)
    if cfg.suite == "sparse"
        ans = @timed EigenvalueSolver.solve_CI_mixed(f, x)
        return ans.value[1], ans.time
    end
    ans = @timed EigenvalueSolver.solve_CI_dense(f, x)
    return ans.value, ans.time
end

"""The roots that are real (max |imag| <= imag_tol, no NaN) and lie in [-1, 1]^dim."""
function real_roots_in_box(cfg::Config, roots)
    kept = Vector{Float64}[]
    for r in roots
        any(isnan, r) && continue
        maximum(abs ∘ imag, r) <= cfg.imag_tol || continue
        rr = Float64.(real.(collect(r)))
        all(v -> abs(v) <= 1, rr) || continue
        push!(kept, rr)
    end
    return kept
end

"""|f_i(root_j)| for every function i and root j."""
function residuals!(all_res, terms, roots)
    for root in roots, (exps, coeffs) in terms
        push!(all_res, abs(eval_terms(exps, coeffs, root)))
    end
    return all_res
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

safe(stat, v) = isempty(v) ? NaN : stat(v)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

function run_benchmark(cfg::Config)
    resolve!(cfg)
    BLAS.set_num_threads(cfg.blas_threads)
    @polyvar x[1:cfg.dim]

    mkpath(joinpath(base_dir(cfg), "roots"))
    summary_paths = Dict(name => summary_path(cfg, name) for name in SUMMARIES)
    if !cfg.append
        for path in values(summary_paths)
            write(path, "")
        end
    end
    failures_path = joinpath(base_dir(cfg), "failed_tests.txt")
    cfg.append || write(failures_path, "")

    for deg in cfg.mindeg:cfg.maxdeg
        println("--- MTV $(cfg.suite) dim $(cfg.dim) degree $(deg)/$(cfg.maxdeg) ---")
        flush(stdout)
        tests = load_degree(cfg, deg)
        reps = min(cfg.num_tests, size(tests, 1))

        if deg == cfg.mindeg
            # Warm up Julia's JIT so it isn't charged to the first timed tests.
            println("  warming up")
            flush(stdout)
            for i in 1:min(cfg.warmup, reps)
                try
                    solve_test(cfg, build_polys(term_lists(cfg, tests, i), x), x)
                catch
                end
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

            terms = term_lists(cfg, tests, i)
            f = build_polys(terms, x)
            try
                roots, elapsed = solve_test(cfg, f, x)
                kept = real_roots_in_box(cfg, roots)

                push!(times, elapsed)
                residuals!(all_res, terms, kept)
                roots_by_test[i] = kept
            catch e
                e isa InterruptException && rethrow()
                msg = sprint(showerror, e)
                println("  [warn] test $(i) failed: $(msg)")
                open(failures_path, "a") do io
                    println(io, "deg=$(deg) test=$(i): $(msg)")
                end
            end
        end

        write_roots_json(roots_path(cfg, deg), roots_by_test)

        append_value(summary_paths["avg_times"], safe(mean, times))
        append_value(summary_paths["max_resids"], safe(maximum, all_res))
        append_value(summary_paths["avg_resids"], safe(mean, all_res))
        append_value(summary_paths["sum_resids"], safe(sum, all_res))
    end

    println("Program Finished")
end


run_benchmark(Config())
