Off[Reduce::ratnz];
Off[Set::shape];

(* Benchmark Reduce on the dense or sparse test suite.

   Run from test_codes/:
     wolframscript -file run_reduce_combined.wls suite dim minDeg maxDeg numTests [nonzero]
       suite     dense | sparse
       nonzero   sparse only: terms per polynomial (default 3)

   For every degree in [minDeg, maxDeg] it solves numTests systems of dim
   polynomials on [-1, 1]^dim and writes, under
   ../<suite>/results/reduce_results/dim{dim}[/nonzero{nonzero}]:
     avg_times.txt, max_resids.txt, avg_resids.txt, sum_resids.txt
                                one value per line, one line per degree
     roots/roots_deg{deg}.json  {test number: [roots]}, same layout as yroots/jroots
   Roots are stored as (xd, ..., x1): useTensor maps the coefficient tensor's
   first axis to the last variable (graph_all.py marks this "roots_reversed").

   Time limit (setting below): after finishing a degree whose average solve
   time is above maxAvgTime seconds, the remaining (higher) degrees are skipped. *)

(* ---------------- Settings ---------------- *)
maxAvgTime       = 300;       (* seconds; Infinity to always run every degree *)
wp               = 15;        (* precision for numerically evaluating Reduce's output *)
zeroConstantTerm = True;      (* dense only: set every constant coefficient to 0,
                                 as every other dense run does *)

(* ---------------- Command line ---------------- *)
args = Rest[$ScriptCommandLine];
If[Length[args] < 5 || !MemberQ[{"dense", "sparse"}, args[[1]]],
  Print["usage: wolframscript -file run_reduce_combined.wls dense|sparse dim minDeg maxDeg numTests [nonzero]"];
  Exit[1]
];
suite    = args[[1]];
dim      = ToExpression[args[[2]]];
minDeg   = ToExpression[args[[3]]];
maxDeg   = ToExpression[args[[4]]];
numTests = ToExpression[args[[5]]];
nonzero  = If[Length[args] >= 6, ToExpression[args[[6]]], 3];

Print["Parameters: suite=", suite, ", dim=", dim, ", degrees=", minDeg, " to ", maxDeg,
      If[suite == "sparse", ", nonzero=" <> ToString[nonzero], ""],
      ", tests=", numTests];

(* ---------------- Polynomials ---------------- *)
vars    = Table[Symbol["x" <> ToString[i]], {i, 1, dim}];
myVars  = Table[ToExpression["x" <> ToString[i] <> "_"], {i, 1, dim}];
myFuncs = Table[Symbol["F" <> ToString[i]], {i, 1, dim}];
Clear @@ myFuncs;
(* Build polynomial from a flattened coefficient tensor via Catch/Throw *)
useTensor = (
  Module[{myFun, myCoord, i, cVal},
    myFun = 1;
    myCoord = Append[#2, 0];
    If[#1 == 0,
      cVal = 1;
      For[i = 1, i <= dim, i++,
        myFun *= vars[[i]]^(#2[[i]]);
        cVal += ((deg + 1)^(i - 1))*(#2[[i]]);
      ];
      myFun *= coeffs[[cVal]];
      Throw[myFun];
    ];
    myFun = 0;
    For[i = 0, i <= deg, i++,
      myCoord = ReplacePart[myCoord, Length[myCoord] -> i];
      myFun += Catch[useTensor[#1 - 1, myCoord]];
    ];
    Throw[myFun];
  ]
) &;

(* Sparse term list {{e1,..,ed,c},..} -> full (deg+1)^dim coefficient tensor *)
toCoeffMatrix[terms_List, dim_Integer, deg_Integer] := Module[{tensor},
  tensor = ConstantArray[0, ConstantArray[deg + 1, dim]];
  Do[
    (* exponents arrive as JSON floats (e.g. 1.0); Part needs integers *)
    With[{exps = Round[term[[;; dim]]], c = term[[dim + 1]]},
      tensor[[Sequence @@ (exps + 1)]] = c
    ],
    {term, terms}
  ];
  tensor
];

(* Flattened coefficient tensor of polynomial i of one test *)
testCoeffs[test_, i_] := Module[{c},
  If[suite == "sparse",
    Flatten[toCoeffMatrix[data[[test, i]], dim, deg]],
    c = Flatten[data[[test, i]]];
    If[zeroConstantTerm, c[[1]] = 0];
    c
  ]
];

coeffFile[deg_] := If[suite == "sparse",
  FileNameJoin[{"..", "sparse", "coeffs_mathematica_json", "dim" <> ToString[dim],
                "deg" <> ToString[deg], "num" <> ToString[nonzero] <> ".txt"}],
  FileNameJoin[{"..", "dense", "coeffs_mathematica_json",
                "dim" <> ToString[dim] <> "_deg" <> ToString[deg] <> "_randn.txt"}]
];

(* ---------------- Solver ---------------- *)
(* Returns the list of real roots in the box; this whole call is timed.
   Reduce gives an exact description of the solution set, which N evaluates
   and NSolveValues turns into explicit points (a quick last step). *)
solveSystem[eqs_] := NSolveValues[
  N[Reduce[And @@ eqs && And @@ Thread[-1 <= vars <= 1], vars, Reals,
           Backsubstitution -> True], wp],
  vars
];

(* numeric residual collector (uses per-degree global allResids) *)
appendResiduals[roots_List] := Module[{val},
  Do[
    With[{root = roots[[k]]},
      Do[
        val = N[Abs[ myFuncs[[j]][root] ]];
        allResids = Append[allResids, val],
        {j, 1, dim}
      ]
    ],
    {k, 1, Length[roots]}
  ];
];

(* ---------------- Output ---------------- *)
(* ToString[NumberForm[...]] alone writes small numbers in the two-line
   "mantissa x 10 / exponent" display form, so force one-line e-notation. *)
numFmt[x_ /; !NumericQ[x]] := "nan";  (* e.g. Indeterminate for a degree with no roots *)
numFmt[x_] := Module[{m, e, mm},
  {m, e} = MantissaExponent[N[x, 50]];
  mm = ToString[NumberForm[m, {Infinity, 17}]];
  mm <> "e" <> ToString[e]
];

(* JSON number: like numFmt, but 0 must be written as a valid literal *)
jsonNum[x_] := Module[{v = N[x], m, e},
  If[v == 0, Return["0.0"]];
  {m, e} = MantissaExponent[v];
  ToString[NumberForm[m, {Infinity, 17}]] <> "e" <> ToString[e]
];

(* <|test number -> roots|> as {test number: [roots]}, one root per line,
   as run_yroots_combined.py writes *)
writeRootsJSON[path_, rootsByTest_Association] := Module[
  {strm, tests = Sort[Keys[rootsByTest]], n, t, rs},
  n = Length[tests];
  strm = OpenWrite[path];
  WriteString[strm, "{\n"];
  Do[
    t = tests[[k]]; rs = rootsByTest[t];
    WriteString[strm, "  \"", ToString[t], "\": ",
      If[rs === {}, "[]",
        "[\n" <> StringRiffle[
          ("    [" <> StringRiffle[jsonNum /@ #, ","] <> "]") & /@ rs, ",\n"] <> "\n  ]"],
      If[k < n, ",\n", "\n"]],
    {k, 1, n}
  ];
  WriteString[strm, "}\n"];
  Close[strm];
];

baseDir = FileNameJoin[Join[
  {"..", suite, "results", "reduce_results", "dim" <> ToString[dim]},
  If[suite == "sparse", {"nonzero" <> ToString[nonzero]}, {}]]];
rootsDir = FileNameJoin[{baseDir, "roots"}];
CreateDirectory[rootsDir, CreateIntermediateDirectories -> True];

(* Summary files: overwritten each run, one line appended per degree *)
maxResidFile = OpenWrite[FileNameJoin[{baseDir, "max_resids.txt"}]];
avgResidFile = OpenWrite[FileNameJoin[{baseDir, "avg_resids.txt"}]];
sumResidFile = OpenWrite[FileNameJoin[{baseDir, "sum_resids.txt"}]];
avgTimesFile = OpenWrite[FileNameJoin[{baseDir, "avg_times.txt"}]];

(* ---------------- Main loop ---------------- *)
For[deg = minDeg, deg <= maxDeg, deg++,

  readFileName = coeffFile[deg];
  Print["Reading ", readFileName];
  data = Import[readFileName, "JSON"];
  If[!ListQ[data], Print["[error] Could not parse JSON: ", readFileName]; Exit[1]];

  (* Per-degree accumulators *)
  allTimes = {};
  allResids = {};
  rootsByTest = <||>;

  For[test = 1, test <= Min[numTests, Length[data]], test++,
    solveFuncs = {};

    For[i = 1, i <= dim, i++,
      polyExpr = Block[{coeffs = testCoeffs[test, i]}, Catch[useTensor[dim, {}]] ];

      (* numeric evaluator: F_i[{x1,..,xd}] -> number *)
      myFuncs[[i]][myVars] := Evaluate[polyExpr];
      solveFuncs = Append[solveFuncs, polyExpr == 0];
    ];

    {timeVal, solList} = AbsoluteTiming @ solveSystem[solveFuncs];

    solList = N[DeleteDuplicates[solList]];
    appendResiduals[solList];
    allTimes = Append[allTimes, timeVal];
    rootsByTest[test] = solList;

    If[Mod[test, 10] == 0, Print["deg ", deg, ", test ", test, "/", numTests]];
  ];

  writeRootsJSON[FileNameJoin[{rootsDir, "roots_deg" <> ToString[deg] <> ".json"}], rootsByTest];

  (* ---- Per-degree summaries over ALL equations & ALL roots ---- *)
  perDegMax  = If[allResids === {}, Indeterminate, Max @ Flatten @ allResids];
  perDegAvg  = If[allResids === {}, Indeterminate, Mean @ Flatten @ allResids];
  perDegSum  = If[allResids === {}, Indeterminate, Total @ Flatten @ allResids];
  perDegTime = If[allTimes  === {}, Indeterminate, Mean[allTimes]];

  WriteString[maxResidFile,  numFmt[perDegMax],  "\n"];
  WriteString[avgResidFile,  numFmt[perDegAvg],  "\n"];
  WriteString[sumResidFile,  numFmt[perDegSum],  "\n"];
  WriteString[avgTimesFile,  numFmt[perDegTime], "\n"];

  Print["deg ", deg, " done: average solve time ", numFmt[perDegTime], " s"];

  If[NumericQ[perDegTime] && perDegTime > maxAvgTime && deg < maxDeg,
    Print["Average solve time above ", maxAvgTime, " s; skipping degrees ",
          deg + 1, " to ", maxDeg];
    Break[]
  ];
];

Close /@ {maxResidFile, avgResidFile, sumResidFile, avgTimesFile};
Print["Program Finished"];
Exit[];
