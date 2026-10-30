using Pkg
using DynamicPolynomials
using NPZ
using DelimitedFiles
using Statistics
using EigenvalueSolver
using Printf

# ------------------ user-config ------------------
dim      = 2
degs     = 2:10          # keep your editable degrees list here
nonzero  = 3            # matches num{nonzero}.npy
numTests = 300
tol      = 1.0e-14      # imag-part tolerance

@polyvar x[1:dim]

# ------------------ output layout (telen_results) ------------------
base_dir = "telen_results/dim$(dim)/nonzero$(nonzero)"
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

for deg in degs
    println("--- Degree $(deg) ---")

    # per-degree roots folder
    out_dir = joinpath(base_dir, "deg$(deg)")
    isdir(out_dir) || mkpath(out_dir)

    # Load coefficients with the same function you were using
    # Expected shape: (numTests, dim, deg+1, deg+1)
    coeff_path = "coeffs/dim$(dim)/deg$(deg)/num$(nonzero).npy"
    coeffs = npzread(coeff_path)   # keep npzread per your request

    basis = [x[1]^i * x[2]^j for i in 0:deg for j in 0:deg]
    reps  = min(size(coeffs, 1), numTests)

    # --- warm-up (same structure, no coeff modifications) ---
    for i in 1:min(10, reps)
        f = [vec(coeffs[i, j, :, :])' * basis for j in 1:dim]
        EigenvalueSolver.solve_CI_dense(f, x)
    end

    timer_sum = 0.0
    all_res   = Float64[]

    for i in 1:reps
        if (i % 10) == 0
            println("  test $(i)/$(reps)")
        end

        # Build polynomials exactly as before
        f = [vec(coeffs[i, j, :, :])' * basis for j in 1:dim]

        ans = @timed EigenvalueSolver.solve_CI_dense(f, x)
        A   = ans.value
        timer_sum += ans.time

        # (a) nearly real (same strict tolerance check)
        mask = trues(length(A))
        for j in 1:length(A)
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
                r = abs(real(f[k](x[1] => vals[1], x[2] => vals[2])))
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
        writedlm(joinpath(out_dir, "roots_test$(i).txt"), roots_mat)
    end

    # --- summaries computed exactly like your original ---
    # (no special-casing / no NaN/zero fallbacks)
    avg_time  = timer_sum / reps
    avg_res   = mean(all_res)
    max_res   = maximum(all_res)
    sum_res   = sum(all_res)

    append_value(avg_time_path,  avg_time)
    append_value(max_resid_path, max_res)
    append_value(avg_resid_path, avg_res)
    append_value(sum_resid_path, sum_res)
end