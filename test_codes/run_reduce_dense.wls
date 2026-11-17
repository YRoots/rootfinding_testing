Off[Reduce::ratnz];
Off[Set::shape];

(* CLI: wolframscript -file run_reduce_dense.wls dim minDeg maxDeg nonzero numTests
   (nonzero is accepted for a uniform command line but unused by the dense suite)

   Writes, under ../dense/results/reduce_results/dim{dim}:
     avg_times.txt, max_resids.txt, avg_resids.txt, sum_resids.txt
                                one value per line, one line per degree
     num_roots.txt              one line per degree: numTests root counts
     roots/roots_deg{deg}.json  {test number: [roots]}, same layout as yroots/jroots
   Roots are stored as (xd, ..., x1): useTensor maps the coefficient tensor's
   first axis to the last variable (graph_all.py marks this "roots_reversed"). *)
dim      = ToExpression[$ScriptCommandLine[[2]]];
minDeg   = ToExpression[$ScriptCommandLine[[3]]];
maxDeg   = ToExpression[$ScriptCommandLine[[4]]];
nonzero  = ToExpression[$ScriptCommandLine[[5]]];
numTests = ToExpression[$ScriptCommandLine[[6]]];

Print["Parameters: dim=", dim, ", degrees=", minDeg, " to ", maxDeg,
      ", tests=", numTests];

(* Working precision used when numerically evaluating Reduce's output *)
wp = 15;

(* Variables & function symbols *)
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

(* --- Number formatting ---
   ToString[NumberForm[...]] alone writes small numbers in the two-line
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

(* {test number: [roots]}, one root per line, as run_yroots_combined.py writes *)
writeRootsJSON[path_, rootsByTest_List] := Module[{strm, n = Length[rootsByTest], rs},
  strm = OpenWrite[path];
  WriteString[strm, "{\n"];
  Do[
    rs = rootsByTest[[t]];
    WriteString[strm, "  \"", ToString[t], "\": ",
      If[rs === {}, "[]",
        "[\n" <> StringRiffle[
          ("    [" <> StringRiffle[jsonNum /@ #, ","] <> "]") & /@ rs, ",\n"] <> "\n  ]"],
      If[t < n, ",\n", "\n"]],
    {t, 1, n}
  ];
  WriteString[strm, "}\n"];
  Close[strm];
];

(* Output dirs *)
baseDir  = FileNameJoin[{"..", "dense", "results", "reduce_results",
                         "dim" <> ToString[dim]}];
rootsDir = FileNameJoin[{baseDir, "roots"}];
CreateDirectory[rootsDir, CreateIntermediateDirectories -> True];

(* Summary files: overwritten each run, one line appended per degree *)
maxResidFile = OpenWrite[FileNameJoin[{baseDir, "max_resids.txt"}]];
avgResidFile = OpenWrite[FileNameJoin[{baseDir, "avg_resids.txt"}]];
sumResidFile = OpenWrite[FileNameJoin[{baseDir, "sum_resids.txt"}]];
avgTimesFile = OpenWrite[FileNameJoin[{baseDir, "avg_times.txt"}]];
numRootsFile = OpenWrite[FileNameJoin[{baseDir, "num_roots.txt"}]];

For[deg = minDeg, deg <= maxDeg, deg++,

  readFileName = FileNameJoin[{
    "..", "dense", "coeffs_mathematica_json",
    "dim" <> ToString[dim] <> "_deg" <> ToString[deg] <> "_randn.txt"
  }];
  Print["Reading ", readFileName];

  data = Import[readFileName, "JSON"];  (* shape: {numTests, dim, deg+1, ..., deg+1} *)
  If[!ListQ[data], Print["[error] Could not parse JSON: ", readFileName]; Exit[1]];

  (* Per-degree accumulators *)
  allTimes = {};
  allResids = {};
  allNumRoots = {};
  rootsByTest = {};

  For[test = 1, test <= Min[numTests, Length[data]], test++,
    solveFuncs = {};

    For[i = 1, i <= dim, i++,
      coeffsSolve = Flatten[data[[test, i]]];

      (* Set constant coeff to 0, as every dense run does *)
      coeffsSolve[[1]] = 0;

      polyExpr = Block[{coeffs = coeffsSolve}, Catch[useTensor[dim, {}]] ];

      (* numeric evaluator: F_i[{x1,..,xd}] -> number *)
      myFuncs[[i]][myVars] := Evaluate[polyExpr];
      solveFuncs = Append[solveFuncs, polyExpr == 0];
    ];

    (* Reduce to an exact description over Reals & box (timed), then extract numeric roots *)
    {timeVal, solSet} = AbsoluteTiming @ N[
      Reduce[
        And @@ solveFuncs && And @@ Thread[-1 <= vars <= 1],
        vars, Reals,
        Backsubstitution -> True
      ],
      wp
    ];
    solList = NSolveValues[solSet, vars];
    solList = N[DeleteDuplicates[solList]];
    appendResiduals[solList];

    allTimes    = Append[allTimes, timeVal];
    allNumRoots = Append[allNumRoots, Length[solList]];
    rootsByTest = Append[rootsByTest, solList];

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
  WriteString[numRootsFile, StringRiffle[ToString /@ allNumRoots, " "], "\n"];
];

Close /@ {maxResidFile, avgResidFile, sumResidFile, avgTimesFile, numRootsFile};
Print["Program Finished"];
Exit[];
