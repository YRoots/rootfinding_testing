% Benchmark Chebfun2 on the dense or sparse test suite (dim 2 only).
%
% Set the parameters below, then run from test_codes/:
%   matlab -batch "run_chebfun_combined"
%
% Writes, under ../<suite>/results/chebfun_results/dim2[/nonzero{nonZero}]:
%   avg_times.txt, max_resids.txt, avg_resids.txt, sum_resids.txt
%                              one value per line, one line per degree
%   roots/roots_deg{deg}.json  {test number: [roots]}, same layout as yroots/jroots
%
% Coordinate order of the stored roots:
%   dense   M(i,j) multiplies y^(i-1) * x^(j-1), i.e. the tensor's first axis
%           is the *second* chebfun variable, so roots are stored as (x2, x1)
%           (graph_all.py marks this "roots_reversed"), matching the
%           Mathematica runs. Like every dense run, the constant term is 0.
%   sparse  each term row is [x_exp, y_exp, coeff], so roots are stored in
%           the natural (x1, x2) order, like yroots.
%
% Time limit: after finishing a degree whose average solve time is above
% maxAvgTime seconds, the remaining (higher) degrees are skipped.

addpath('../chebfun');
cheb.xy;

% ---------------- Parameters ----------------
suite      = "dense";   % "dense" or "sparse"
dim        = 2;         % Chebfun2 only solves dim 2
nonZero    = 3;         % sparse only: terms per polynomial
numTests   = 100;
minDeg     = 2;
maxDeg     = 30;
maxAvgTime = 1000;       % seconds; Inf to run every degree

nzText = "";
if suite == "sparse", nzText = sprintf(", nonzero=%d", nonZero); end
fprintf("Parameters: suite=%s, dim=%d, degrees=%d to %d, tests=%d%s\n", ...
        suite, dim, minDeg, maxDeg, numTests, nzText);

% ---------------- Output ----------------
if suite == "dense"
    baseDir = sprintf("../dense/results/chebfun_results_new/dim%d", dim);
else
    baseDir = sprintf("../sparse/results/chebfun_results_new/dim%d/nonzero%d", dim, nonZero);
end
rootsDir = fullfile(baseDir, "roots");
if ~exist(rootsDir, 'dir'), mkdir(rootsDir); end

% Summary files: overwritten each run, one line appended per degree
time_fid      = fopen(fullfile(baseDir, "avg_times.txt"),  'w');
avg_res_fid   = fopen(fullfile(baseDir, "avg_resids.txt"), 'w');
max_res_fid   = fopen(fullfile(baseDir, "max_resids.txt"), 'w');
sum_res_fid   = fopen(fullfile(baseDir, "sum_resids.txt"), 'w');

% ---------------- Main loop ----------------
for deg = minDeg:maxDeg
    % Same Mathematica-JSON coefficients the other solvers read
    if suite == "dense"
        coeffFile = sprintf("../dense/coeffs_mathematica_json/dim%d_deg%d_randn.txt", dim, deg);
    else
        coeffFile = sprintf("../sparse/coeffs_mathematica_json/dim%d/deg%d/num%d.txt", dim, deg, nonZero);
    end
    fprintf("Reading %s\n", coeffFile);
    % dense:  numTests x dim x (deg+1) x (deg+1)
    % sparse: numTests x dim x (nonZero+1) x (dim+1)
    A = jsondecode(fileread(coeffFile));
    reps = min(numTests, size(A,1));

    all_times     = nan(1, reps);
    all_resids    = [];
    roots_by_test = cell(1, reps);

    for test = 1:reps
        P = reshape(A(test, 1, :, :), size(A,3), size(A,4));
        Q = reshape(A(test, 2, :, :), size(A,3), size(A,4));
        if suite == "dense"
            P(1,1) = 0;   % zero constant term
            Q(1,1) = 0;
            F = @(x,y) evalDense2D(P, x, y);
            G = @(x,y) evalDense2D(Q, x, y);
        else
            F = @(x,y) evalSparse2D(P, x, y);
            G = @(x,y) evalSparse2D(Q, x, y);
        end

        try
            tStart = tic;
            roots_ = roots([chebfun2(F); chebfun2(G)], 'resultant');
            all_times(test) = toc(tStart);
        catch ME
            warning("Skipping deg %d test %d: %s", deg, test, ME.message);
            roots_ = zeros(0,2); % no roots recorded
        end

        for k = 1:size(roots_,1)
            all_resids(end+1) = abs(F(roots_(k,1), roots_(k,2))); %#ok<AGROW>
            all_resids(end+1) = abs(G(roots_(k,1), roots_(k,2))); %#ok<AGROW>
        end
        roots_by_test{test} = roots_;

        if mod(test,10) == 0
            fprintf("deg %d: %d/%d tests completed\n", deg, test, reps);
        end
    end

    writeRootsJSON(fullfile(rootsDir, sprintf("roots_deg%d.json", deg)), roots_by_test);

    % ---- Per-degree summaries ----
    avgTime = mean(all_times(~isnan(all_times)));
    fprintf(time_fid,    '%.17g\n', avgTime);
    fprintf(avg_res_fid, '%.17g\n', mean(all_resids));
    fprintf(max_res_fid, '%.17g\n', max([all_resids, NaN]));
    fprintf(sum_res_fid, '%.17g\n', sum(all_resids));
    fprintf("deg %d done: average solve time %.6g s\n", deg, avgTime);

    if avgTime > maxAvgTime && deg < maxDeg
        fprintf("Average solve time above %g s; skipping degrees %d to %d\n", ...
                maxAvgTime, deg + 1, maxDeg);
        break;
    end
end

fclose(time_fid);
fclose(avg_res_fid);
fclose(max_res_fid);
fclose(sum_res_fid);

exit;

% ---- helper functions ----
function val = evalDense2D(M, x, y)
    % M(i,j) multiplies y^(i-1) * x^(j-1); elementwise in x and y
    val = zeros(size(x));
    [I, J, c] = find(M);
    for k = 1:numel(c)
        val = val + c(k) .* (y.^(I(k)-1)) .* (x.^(J(k)-1));
    end
end

function val = evalSparse2D(T, x, y)
    % T: nTerms x 3, columns are [x_exp, y_exp, coeff]; elementwise in x and y
    val = zeros(size(x));
    for k = 1:size(T,1)
        val = val + T(k,3) .* (x.^T(k,1)) .* (y.^T(k,2));
    end
end

function writeRootsJSON(path, roots_by_test)
    % {test number: [roots]}, one root per line, as run_yroots_combined.py writes
    fid = fopen(path, 'w');
    fprintf(fid, "{\n");
    n = numel(roots_by_test);
    for t = 1:n
        R = roots_by_test{t};
        if isempty(R)
            fprintf(fid, '  "%d": []', t);
        else
            rows = strings(size(R,1), 1);
            for k = 1:size(R,1)
                rows(k) = "    [" + strjoin(compose("%.17g", R(k,:)), ",") + "]";
            end
            fprintf(fid, '  "%d": [\n%s\n  ]', t, strjoin(rows, sprintf(",\n")));
        end
        if t < n, fprintf(fid, ",\n"); else, fprintf(fid, "\n"); end
    end
    fprintf(fid, "}\n");
    fclose(fid);
end
