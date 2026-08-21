include("../../hyun/Julia-Rootfinding/src/CombinedSolver.jl")
using NPZ
using DelimitedFiles
using Statistics
using JSON3

dim = 2
degs = 2:30
nonzero = 3
numtests = 100

# Change efficient coeffs into coeff matrix
function create_coeff_matrix(coeff_data, dim, deg, test_num, func_num)
    # shape = [dim] + [(deg+1)]*dim
    shape = ntuple(_ -> deg+1, dim)

    # Initialize coefficient matrix with zeros
    coeff_matrix = zeros(shape)

    # coordinates: test_num[:, :, 1:dim]
    coordinates = coeff_data[test_num, func_num, :, 1:dim]

    # values: test_num[:, :, dim+1]
    values = coeff_data[test_num, func_num, :, dim+1]

    N = size(coordinates, 1)

    for i in 1:N
        idx = Tuple(Int.(coordinates[i, :]) .+ 1)   # convert to Int, shift if zero-based
        coeff_matrix[idx...] = values[i]
    end

    return coeff_matrix
end

function multipower_from_terms(terms::Matrix, dim::Int)::MultiPower
    max_exp = Int.(maximum(terms[:, 1:dim], dims=1)) .+ 1
    coeff = zeros(Tuple(max_exp)...)
    for row in eachrow(terms)
        idx = CartesianIndex(Tuple(Int.(row[1:dim]) .+ 1))
        coeff[idx] = Float64(row[dim+1])
    end
    return MultiPower(coeff)
end


# Writes roots to a JSON file
function write_to_json(output_file, data)
    open(output_file, "w") do f
        write(f, "{\n")

        items = sort(collect(data), by = x -> parse(Int, x[1]))
        for (i, (test_num, roots)) in enumerate(items)
            write(f, "  \"$test_num\": ")

            if isempty(roots)
                write(f, "[]")
            else
                write(f, "[\n")
                for (j, root) in enumerate(roots)
                    write(f, "    " * JSON3.write(root))
                    if j < length(roots)
                        write(f, ",")
                    end
                    write(f, "\n")
                end
                write(f, "  ]")
            end

            if i < length(items)
                write(f, ",")
            end
            write(f, "\n")
        end

        write(f, "}\n")
    end
end

dir = "../sparse/results/jroots_results3/dim$(dim)/nonzero$(nonzero)"
mkpath(dir)  # creates nested directories if missing

files = ["avg_times.txt", "avg_resids.txt", "max_resids.txt", "sum_resids.txt"]

for file in files
    filepath = joinpath(dir, file)
    open(filepath, "w") # Change to "a" when testing additional degrees
end

for deg in degs
    times_file = open("$dir/avg_times.txt","a")
    avg_resids_file = open("$dir/avg_resids.txt","a")
    max_resids_file = open("$dir/max_resids.txt","a")
    sum_resids_file = open("$dir/sum_resids.txt","a")

    new_dir = joinpath(dir, "roots")
    mkpath(new_dir)
    roots_test_file = "$new_dir/roots_deg$(deg).json"
    
    all_res = []
    timer = 0;
    coeffs = npzread("../sparse/coeffs/dim$(dim)/deg$(deg)/num$(nonzero).npy")
    reps = min(size(coeffs)[1], numtests); # should be 100
    A = []
    roots = Dict()
    
    if deg == 2
        # Minimize compilation time error by running a few tests without times
        println("Starting warmup test")
        flush(stdout)
        for i in 1:10
            f = [multipower_from_terms(coeffs[i, j, :, :], dim) for j in 1:dim]
            solve(f,-ones(dim),ones(dim))
        end
    end

    println("Starting tests")
    flush(stdout)

    for i in 1:reps
        if i%10 == 0
            println("Running dim $(dim) deg $(deg) test $(i)")
            flush(stdout)
        end

        f = [MultiPower(to_python(create_coeff_matrix(coeffs, dim, deg, i, j))) for j in 1:dim];
        ans = @timed solve(f,-ones(dim),ones(dim))
        A = ans.value;
        timer += ans.time;

        for j=1:length(A)
            for idx in 1:dim
                append!(all_res,abs(eval_MultiPower(f[idx], A[j])))
            end
        end
        
        roots[string(i)] = A

    end

    write_to_json(roots_test_file, roots)
    
    timings = timer/reps

    if isempty(all_res)
        avg_res = NaN
        max_res = NaN
        sum_res = NaN
    else
        avg_res = mean(all_res)
        max_res = maximum(all_res)
        sum_res = sum(all_res)
    end

    write(times_file,string(timings) * "\n")
    write(avg_resids_file,string(avg_res) * "\n")
    write(max_resids_file,string(max_res) * "\n")
    write(sum_resids_file,string(sum_res) * "\n")

    close(times_file)
    close(avg_resids_file)
    close(max_resids_file)
    close(sum_resids_file)
end

print("Program finished!")