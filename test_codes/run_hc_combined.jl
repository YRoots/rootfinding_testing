"""
Benchmark HomotopyContinuation.jl on the sparse and dense polynomial test suites.

Set the fields in `Config` at the bottom of this file, then run
``julia run_hc_combined.jl``. Install the dependencies once with

    julia -e 'using Pkg; Pkg.add(["HomotopyContinuation", "NPZ", "JSON3"])'

HomotopyContinuation finds every isolated root in C^dim, so after each solve the
roots are filtered down to those that are real (max |imag| <= imag_tol) and lie in
[-1, 1]^dim, which makes the output comparable with the other solvers.

For every degree in [mindeg, maxdeg] it solves num_tests systems of dim
polynomials and writes, under the run's base directory
(``<result_dir>/dim{dim}/nonzero{nonzero}`` for sparse,
``<result_dir>/dim{dim}`` for dense):

    avg_times.txt              average solve time, one value per degree
    max_resids.txt             max |f_i(root)|, one value per degree
    avg_resids.txt             mean |f_i(root)|, one value per degree
    sum_resids.txt             sum of |f_i(root)|, one value per degree
    avg_paths.txt              average number of paths tracked, one value per degree
    avg_failed_paths.txt       average number of paths that failed, one value per degree
    roots/roots_deg{deg}.json  {test number: roots} for that degree

The .txt files hold one plain value per line in degree order; the JSON files
match the layout of run_jroots_combined.jl and run_yroots_combined.py, so the
runs can be diffed directly with compare_roots.py.
"""

# Threads HomotopyContinuation uses when Config.threading is true. Julia fixes its
# thread count at startup, so if it was started with a different number the script
# relaunches itself with this many. `nothing` keeps whatever Julia was started with
# (`julia -t N`).
const NUM_THREADS = nothing

if NUM_THREADS !== nothing && Threads.nthreads() != NUM_THREADS &&
   !isempty(PROGRAM_FILE) && !haskey(ENV, "HC_BENCH_RELAUNCHED")
    cmd = `$(Base.julia_cmd()) --threads=$(NUM_THREADS) $(abspath(PROGRAM_FILE)) $(ARGS)`
    exit(run(ignorestatus(addenv(cmd, "HC_BENCH_RELAUNCHED" => "1"))).exitcode)
end

using HomotopyContinuation
using NPZ
using JSON3
using Statistics
using Printf
using Random

const SUMMARIES = ("avg_times", "max_resids", "avg_resids", "sum_resids",
                   "avg_paths", "avg_failed_paths")


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

Base.@kwdef mutable struct Config
    suite::String = "sparse"           # "sparse" | "dense"
    dim::Int = 2
    mindeg::Int = 2
    maxdeg::Int = 10
    nonzero::Int = 3                   # sparse only: terms per polynomial
    num_tests::Int = 100
    coeff_dir::Union{String,Nothing} = nothing   # defaults per suite
    result_dir::Union{String,Nothing} = nothing  # defaults to ../<suite>/results/hc_results
    append::Bool = false               # true to add degrees to an existing run
    warmup::Int = 10                   # untimed solves at mindeg to absorb JIT cost
    imag_tol::Float64 = 1e-10          # a root counts as real if max |imag| <= imag_tol
    box_tol::Float64 = 0.0             # keep roots with |x_k| <= 1 + box_tol
    zero_constant_term::Bool = true    # dense only: force the constant coefficient to 0,
                                       # as the MTV/Bertini/Chebfun/NSolve dense runs do
    # HomotopyContinuation.solve options
    start_system::Symbol = :polyhedral # :polyhedral | :total_degree
    compile::Bool = false              # compiling costs seconds per new system support
    threading::Bool = false            # true uses NUM_THREADS threads (top of file)
    seed::UInt32 = 0x12345678          # fixes the random start system for reproducibility
end

"""Fill in the per-suite defaults and validate."""
function resolve!(cfg::Config)
    cfg.suite in ("sparse", "dense") || error("unknown suite: $(cfg.suite)")
    cfg.start_system in (:polyhedral, :total_degree) ||
        error("unknown start_system: $(cfg.start_system)")
    if cfg.coeff_dir === nothing
        cfg.coeff_dir = cfg.suite == "sparse" ? "../sparse/coeffs" : "../dense/coeffs"
    end
    if cfg.result_dir === nothing
        cfg.result_dir = "../$(cfg.suite)/results/hc_results"
    end
    return cfg
end

"""Where this run's summaries and roots/ folder live."""
function base_dir(cfg::Config)
    d = joinpath(cfg.result_dir, "dim$(cfg.dim)")
    return cfg.suite == "sparse" ? joinpath(d, "nonzero$(cfg.nonzero)") : d
end

summary_path(cfg::Config, name) = joinpath(base_dir(cfg), "$(name).txt")
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
(one row per term) and a vector of coefficients. Both suites reduce to this, and
it is used both to build the HC system and to evaluate residuals independently of HC.
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

"""Build a HomotopyContinuation System from term lists."""
function build_system(terms, x)
    dim = length(x)
    polys = map(terms) do (exps, coeffs)
        isempty(coeffs) && return Expression(0)
        sum(coeffs[k] * prod(x[l]^exps[k, l] for l in 1:dim) for k in eachindex(coeffs))
    end
    return System(polys; variables = x)
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

"""Solve one system, returning the HC result and the elapsed seconds."""
function solve_test(cfg::Config, F)
    ans = @timed solve(F;
                       start_system = cfg.start_system,
                       compile = cfg.compile,
                       threading = cfg.threading,
                       seed = cfg.seed,
                       show_progress = false)
    return ans.value, ans.time
end

"""The finite roots of an HC result that are real and lie in [-1, 1]^dim."""
function real_roots_in_box(cfg::Config, result)
    roots = Vector{Float64}[]
    for s in solutions(result; only_nonsingular = false, only_finite = true)
        maximum(abs ∘ imag, s) <= cfg.imag_tol || continue
        r = real.(s)
        all(v -> abs(v) <= 1 + cfg.box_tol, r) || continue
        push!(roots, r)
    end
    return roots
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
    @var x[1:cfg.dim]
    println("threading = $(cfg.threading), Julia threads = $(Threads.nthreads())")

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
        println("--- HC $(cfg.suite) dim $(cfg.dim) degree $(deg)/$(cfg.maxdeg) ---")
        flush(stdout)
        tests = load_degree(cfg, deg)
        reps = min(cfg.num_tests, size(tests, 1))

        if deg == cfg.mindeg
            # Warm up Julia's JIT so it isn't charged to the first timed tests.
            println("  warming up")
            flush(stdout)
            for i in 1:min(cfg.warmup, reps)
                solve_test(cfg, build_system(term_lists(cfg, tests, i), x))
            end
        end

        times = Float64[]
        all_res = Float64[]
        path_counts = Int[]
        failed_counts = Int[]
        roots_by_test = Dict{Int,Any}()

        for i in 1:reps
            if i % 10 == 0
                println("  test $(i)/$(reps)")
                flush(stdout)
            end

            terms = term_lists(cfg, tests, i)
            F = build_system(terms, x)
            try
                result, elapsed = solve_test(cfg, F)
                roots = real_roots_in_box(cfg, result)

                push!(times, elapsed)
                push!(path_counts, length(path_results(result)))
                push!(failed_counts, nfailed(result))  # paths that neither converged nor diverged
                residuals!(all_res, terms, roots)
                roots_by_test[i] = roots
            catch e
                e isa InterruptException && rethrow()
                msg = sprint(showerror, e)
                println("  [warn] test $(i) failed: $(msg)")
                open(failures_path, "a") do f
                    println(f, "deg=$(deg) test=$(i): $(msg)")
                end
            end
        end

        write_roots_json(roots_path(cfg, deg), roots_by_test)

        append_value(summary_paths["avg_times"], safe(mean, times))
        append_value(summary_paths["max_resids"], safe(maximum, all_res))
        append_value(summary_paths["avg_resids"], safe(mean, all_res))
        append_value(summary_paths["sum_resids"], safe(sum, all_res))
        append_value(summary_paths["avg_paths"], safe(mean, path_counts))
        append_value(summary_paths["avg_failed_paths"], safe(mean, failed_counts))
    end

    println("Program Finished")
end


run_benchmark(Config())
