% Benchmark Chebfun2 on the sparse test suite (dim 2 only).
%
% Run from test_codes/. Writes, under
% ../sparse/results/chebfun_results/dim2/nonzero{nonZero}:
%   avg_times.txt, max_resids.txt, avg_resids.txt, sum_resids.txt
%                              one value per line, one line per degree
%   roots/roots_deg{deg}.json  {test number: [roots]}, same layout as yroots/jroots
%
% Each sparse term row is [x_exp, y_exp, coeff], so roots are stored in the
% natural (x1, x2) order, like yroots (not "roots_reversed" in graph_all.py).

addpath('../chebfun');
cheb.xy;

% User-configurable parameters
dim       = 2;
nonZero   = 3;
numTests  = 100;
minDeg    = 2;
maxDeg    = 30;

baseDir  = sprintf("../sparse/results/chebfun_results/dim%d/nonzero%d", dim, nonZero);
rootsDir = fullfile(baseDir, "roots");
if ~exist(rootsDir, 'dir'), mkdir(rootsDir); end

% Summary files: overwritten each run, one line appended per degree
time_fid      = fopen(fullfile(baseDir, "avg_times.txt"),  'w');
avg_res_fid   = fopen(fullfile(baseDir, "avg_resids.txt"), 'w');
max_res_fid   = fopen(fullfile(baseDir, "max_resids.txt"), 'w');
sum_res_fid   = fopen(fullfile(baseDir, "sum_resids.txt"), 'w');

for deg = minDeg:maxDeg
    coeffFile = sprintf("../sparse/coeffs_mathematica_json/dim%d/deg%d/num%d.txt", dim, deg, nonZero);
    fprintf("Reading %s\n", coeffFile);
    A = jsondecode(fileread(coeffFile));   % numTests x dim x (nonZero+1) x (dim+1)
    reps = min(numTests, size(A,1));

    all_times     = nan(1, reps);
    all_resids    = [];
    roots_by_test = cell(1, reps);

    for test = 1:reps
        F_terms = reshape(A(test, 1, :, :), size(A,3), size(A,4));
        G_terms = reshape(A(test, 2, :, :), size(A,3), size(A,4));

        F = @(x,y) evalSparse2D(F_terms, x, y);
        G = @(x,y) evalSparse2D(G_terms, x, y);

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
    fprintf(time_fid,    '%.17g\n', mean(all_times(~isnan(all_times))));
    fprintf(avg_res_fid, '%.17g\n', mean(all_resids));
    fprintf(max_res_fid, '%.17g\n', max([all_resids, NaN]));
    fprintf(sum_res_fid, '%.17g\n', sum(all_resids));
end

fclose(time_fid);
fclose(avg_res_fid);
fclose(max_res_fid);
fclose(sum_res_fid);

exit;

% ---- helper functions ----
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
