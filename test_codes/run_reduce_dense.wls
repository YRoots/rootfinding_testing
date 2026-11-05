Off[Reduce::ratnz];
Off[Set::shape];

(* CLI: math -script run_Reduce_perdegree_onerow_numroots.wls dim minDeg maxDeg nonzero numTests *)
dim      = ToExpression[$ScriptCommandLine[[2]]];
minDeg   = ToExpression[$ScriptCommandLine[[3]]];
maxDeg   = ToExpression[$ScriptCommandLine[[4]]];
nonzero  = ToExpression[$ScriptCommandLine[[5]]];
numTests = ToExpression[$ScriptCommandLine[[6]]];

Print["Parameters: dim=", dim, ", degrees=", minDeg, " to ", maxDeg,
      ", nonzero=", nonzero, ", tests=", numTests];

(* Working precision used when numerically evaluating Reduce’s output *)
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

(* Base dir (per dim/nonzero) *)
baseDir = FileNameJoin[{
  "reduce_results",
  "dim" <> ToString[dim],
  "nonzero" <> ToString[nonzero]
}];
CreateDirectory[baseDir, CreateIntermediateDirectories -> True];

(* Roots file path: per-degree/per-test *)
rootsFileForTest[t_, d_] := Module[{degreeDir},
  degreeDir = FileNameJoin[{baseDir, "deg" <> ToString[d]}];
  CreateDirectory[degreeDir, CreateIntermediateDirectories -> True];
  FileNameJoin[{degreeDir, "roots_test" <> ToString[t] <> ".txt"}]
];

(* Summary files (append one per-degree line to each) *)
maxResidFile = OpenAppend[FileNameJoin[{baseDir, "max_resids.txt"}]];
avgResidFile = OpenAppend[FileNameJoin[{baseDir, "avg_resids.txt"}]];
sumResidFile = OpenAppend[FileNameJoin[{baseDir, "sum_resids.txt"}]];
avgTimesFile = OpenAppend[FileNameJoin[{baseDir, "avg_times.txt"}]];
numRootsFile = OpenAppend[FileNameJoin[{baseDir, "num_roots.txt"}]];  (* one line per degree: numTests ints *)

(* --- Force consistent e-notation (ASCII) --- *)
numFmt[x_ /; !NumericQ[x]] := "nan";  (* e.g. Indeterminate for a degree with no roots *)
numFmt[x_] := Module[{m, e, mm},
  {m, e} = MantissaExponent[N[x, 50]];
  mm = ToString[NumberForm[m, {Infinity, 17}]];
  mm <> "e" <> ToString[e]
];

For[deg = minDeg, deg <= maxDeg, deg++,

  readFileName = FileNameJoin[{
    "mathematica_coeffs",
    "dim" <> ToString[dim],
    "deg" <> ToString[deg],
    "num" <> ToString[nonzero] <> ".txt"
  }];
  Print["Reading ", readFileName];

  data = Import[readFileName, "JSON"];  (* shape: {numTests, dim, deg+1, deg+1} *)
  If[!ListQ[data], Print["[error] Could not parse JSON: ", readFileName]; Exit[1]];

  (* Per-degree accumulators *)
  allTimes = {};
  allResids = {};
  allNumRoots = {};

  For[test = 1, test <= numTests, test++,
    solveFuncs = {};

    For[i = 1, i <= dim, i++,
      coeffMat   = data[[test, i]];
      coeffsSolve = Flatten[coeffMat];

      polyExprOrig  = Block[{coeffs = coeffsSolve},  Catch[useTensor[dim, {}]] ];
      (* For Reduce, equations themselves are symbolic; same poly used *)
      polyExprSolve = polyExprOrig;

      (* numeric evaluator for ORIGINAL polynomial *)
      (*myFuncs[[i]] =.;*)  (* intentionally not clearing definitions *)
      myFuncs[[i]][myVars] := Evaluate[polyExprOrig];

      (* equation Fi == 0 to feed into Reduce *)
      solveFuncs = Append[solveFuncs, polyExprSolve == 0];
    ];

    (* --- Reduce to an exact/parametric description over Reals & box, then sample numeric solutions --- *)
    (* Time only the Reduce step, mirroring your prior reduce workflow *)
    {timeVal, solSet} = AbsoluteTiming @ N[
      Reduce[
        And @@ solveFuncs && And @@ Thread[-1 <= vars <= 1],
        vars, Reals,
        Backsubstitution -> True
      ],
      wp
    ];

    (* Turn Reduce output into explicit numeric solutions *)
    solList = NSolveValues[solSet, vars];
    solList = N[DeleteDuplicates[solList], wp];

    appendResiduals[solList];

    allTimes    = Append[allTimes, timeVal];
    allNumRoots = Append[allNumRoots, Length[solList]];

    (* ---- Write roots to per-degree/per-test file ---- *)
    Module[{rfile = rootsFileForTest[test, deg], strm},
      strm = OpenAppend[rfile];
      If[Length[solList] > 0,
        Scan[(WriteString[strm, StringRiffle[numFmt /@ #, " "], "\n"]) &, solList],
        WriteString[strm, "# (no roots)\n"]
      ];
      Close[strm];
    ];

    If[Mod[test, 10] == 0, Print["deg ", deg, ", test ", test, "/", numTests]];
  ];

  (* ---- Per-degree summaries ---- *)
  perDegMax  = If[allResids === {}, Indeterminate, Max @ Flatten @ allResids];
  perDegAvg  = If[allResids === {}, Indeterminate, Mean @ Flatten @ allResids];
  perDegSum  = If[allResids === {}, Indeterminate, Total @ Flatten @ allResids];
  perDegTime = If[allTimes  === {}, Indeterminate, Mean[allTimes]];

  WriteString[maxResidFile,  numFmt[perDegMax],  "\n"];
  WriteString[avgResidFile,  numFmt[perDegAvg],  "\n"];
  WriteString[sumResidFile,  numFmt[perDegSum],  "\n"];
  WriteString[avgTimesFile,  numFmt[perDegTime], "\n"];

  (* ---- num_roots: append one line with counts for this degree ---- *)
  WriteString[numRootsFile, StringRiffle[ToString /@ allNumRoots, " "], "\n"];
];

Close /@ {maxResidFile, avgResidFile, sumResidFile, avgTimesFile, numRootsFile};
Print[numFmt[perDegMax]];
Exit[];
