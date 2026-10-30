using Pkg
using DynamicPolynomials
using NPZ
using DelimitedFiles
using Statistics
using EigenvalueSolver
using Printf
using Combinatorics  # for product()
using LinearAlgebra: BLAS


# ------------------ user-config ------------------
dim      = 7
degs     = 2:4          # keep your editable degrees list here
nonzero  = 3            # matches num{nonzero}.npy
numTests = 100
tol      = 1.0e-13      # imag-part tolerance
max_cpus = 40
@polyvar x[1:dim]

# Controls number of cpus used
BLAS.set_num_threads(max_cpus)
# ------------------ output layout (telen_results) ------------------
base_dir = "../sparse/results/telen_results/dim$(dim)/nonzero$(nonzero)"
isdir(base_dir) || mkpath(base_dir)

avg_time_path  = joinpath(base_dir, "avg_times.txt")
max_resid_path = joinpath(base_dir, "max_resids.txt")
avg_resid_path = joinpath(base_dir, "avg_resids.txt")
sum_resid_path = joinpath(base_dir, "sum_resids.txt")

# truncate summary files once
open(avg_time_path,  "w") do _ end
open(max_resid_path, "w") do _ end
open(avg_resid_path, "w") do _ end
open(sum_resid_path, "w") do _ end

# append one numeric value per line (no degree labels)
function append_value(path::AbstractString, v::Real)
    open(path, "a") do io
        @printf(io, "%.17g\n", float(v))
    end
end

# Change efficient coeffs into coeff matrix
function create_coeff_matrix(coeff_data, dim, deg, test_num, func_num)
    # shape = [dim] + [(deg+1)]*dim
    shape = ntuple(_ -> deg+1, dim)

    coeff_matrix = zeros(shape)
    coordinates = coeff_data[test_num, func_num, :, 1:dim]
    values = coeff_data[test_num, func_num, :, dim+1]

    N = size(coordinates, 1)

    for i in 1:N
        idx = Tuple(Int.(coordinates[i, :]) .+ 1)   # convert to Int, shift if zero-based
        coeff_matrix[idx...] = values[i]
    end

    return coeff_matrix
end

println("Program Started!")

for deg in degs
    bad_tests = []
    println("--- Degree $(deg) ---")
    flush(stdout)

    # per-degree roots folder
    out_dir = joinpath(base_dir, "deg$(deg)")
    isdir(out_dir) || mkpath(out_dir)
    roots_test_file = open("$out_dir/roots_test.txt","w")

    # Load coefficients with the same function you were using
    # Expected shape: (numTests, dim, deg+1, deg+1)
    coeff_path = "../sparse/coeffs/dim$(dim)/deg$(deg)/num$(nonzero).npy"
    coeffs = npzread(coeff_path)   # keep npzread per your request

    # CHECK THIS
    basis = [prod(x[k]^a[k] for k in 1:dim) for a in Iterators.product(fill(0:deg, dim)...)]
    reps  = min(size(coeffs, 1), numTests)

    # --- warm-up (same structure, no coeff modifications) ---
    for i in 1:min(10, reps)
        # CHECK THIS
        f = [vec(create_coeff_matrix(coeffs, dim, deg, i, j))' * vec(basis) for j in 1:dim]
        EigenvalueSolver.solve_CI_mixed(f, x)
    end

    timer_sum = 0.0
    all_res   = Float64[]

    for i in 1:reps
        if (i % 10) == 0
            println("  test $(i)/$(reps)")
        end
        flush(stdout)

        # Build polynomials exactly as before
        # CHECK THIS
        f = [vec(create_coeff_matrix(coeffs, dim, deg, i, j))' * vec(basis) for j in 1:dim]

        try
            ans = @timed EigenvalueSolver.solve_CI_mixed(f, x)
            A   = ans.value[1]
            timer_sum += ans.time

            # (a) nearly real (same strict tolerance check)
            mask = trues(length(A))
            for j in 1:length(A)
                # CHECK This
                # Check for and drop and NaN
                if any(isnan, A[j])
                    mask[j] = false
                    continue
                end


                for k in 1:dim
                    if abs(imag(A[j][k])) > tol
                        mask[j] = false
                        break
                    end
                end
            end
            A = A[mask]
            A = real.(A)

            # (b) inside the closed box [-1,1]^dim, the same rule as the other solvers
            mask = trues(length(A))
            for j in 1:length(A)
                inside = true
                for k in 1:dim
                    v = A[j][k]
                    if v > 1 || v < -1
                        inside = false
                        break
                    end
                end
                mask[j] = inside
            end
            Q = A[mask]

            # residuals: |f_k(root)|
            for j in 1:length(Q)
                vals = Tuple(Q[j])
                for k in 1:dim
                    # CHECK THIS
                    subs_dict = Dict(x[l] => vals[l] for l in 1:dim)
                    r = abs(real(f[k](subs_dict...)))
                    # r = abs(real(f[k](x[1] => vals[1], x[2] => vals[2])))      # Change with dim
                    push!(all_res, r)
                end
            end

            # write roots per test with writedlm (text), as before
            if length(Q) == 0
                roots_mat = Array{Float64}(undef, 0, dim)
            else
                roots_mat = Array{Float64}(undef, length(Q), dim)
                for (row, tup) in enumerate(Q)
                    for k in 1:dim
                        roots_mat[row, k] = real(tup[k])
                    end
                end
            end

            write(roots_test_file, "Test $(i)\n")
            writedlm(roots_test_file, roots_mat)
            write(roots_test_file, "\n")

            #writedlm(joinpath(out_dir, "roots_test$(i).txt"), roots_mat)
        catch e
            write(roots_test_file, "Test $(i)\n")
            write(roots_test_file, "Test $i failed: $(e)")
            write(roots_test_file, "\n")
            push!(bad_tests, i)
            continue
        end
    end

    # --- summaries computed exactly like your original ---
    # (no special-casing / no NaN/zero fallbacks)
    avg_time  = timer_sum / (reps - length(bad_tests))

    if isempty(all_res)
        avg_res = NaN
        max_res = NaN
        sum_res = NaN
    else
        avg_res = mean(all_res)
        max_res = maximum(all_res)
        sum_res = sum(all_res)
    end

    append_value(avg_time_path,  avg_time)
    append_value(max_resid_path, max_res)
    append_value(avg_resid_path, avg_res)
    append_value(sum_resid_path, sum_res)

    println("Bad tests for deg $(deg):", bad_tests)
end

println("Program Finished!")
