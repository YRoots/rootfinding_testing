Off[Reduce::ratnz];
Off[Set::shape];

(* CLI: wolframscript -file filename.wls dim minDeg maxDeg nonzero numTests *)
dim      = ToExpression[$ScriptCommandLine[[2]]];
minDeg   = ToExpression[$ScriptCommandLine[[3]]];
maxDeg   = ToExpression[$ScriptCommandLine[[4]]];
nonzero  = ToExpression[$ScriptCommandLine[[5]]];
numTests = ToExpression[$ScriptCommandLine[[6]]];

Print["Parameters: dim=", dim, ", degrees=", minDeg, " to ", maxDeg,
      ", nonzero=", nonzero, ", tests=", numTests];

(* Higher precision for improved accuracy in solves/residuals *)
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
        (* Guard against any accidental list-valued return; force scalar *)
        val = N[Abs[ myFuncs[[j]][root] ]];
        (*val = If[ListQ[val], Mean[Flatten[val]], val];*)
        allResids = Append[allResids, val],
        {j, 1, dim}
      ]
    ],
    {k, 1, Length[roots]}
  ];
];

(* Turn efficient coefficients into full coefficient matrix *)
toCoeffMatrix[data_List, dim_Integer, deg_Integer] := Module[
  {shape, tensor},

  (* create a tensor of zeros with size (deg+1)^dim *)
  shape = ConstantArray[deg + 1, dim];
  tensor = ConstantArray[0, shape];
  
  (* fill coefficients *)
  Do[
    With[{exps = term[[;; dim]], c = term[[dim + 1]]},
      tensor[[Sequence @@ (exps + 1)]] = c
    ],
    {term, data}
  ];
  
  tensor
]

(* Base dir *)
baseDir = FileNameJoin[{
  "..",
  "sparse",
  "results",
  "NSolveValues_results",
  "dim" <> ToString[dim],
  "nonzero" <> ToString[nonzero]
}];
CreateDirectory[baseDir, CreateIntermediateDirectories -> True];

rootsFileForTest[deg_, t_] := Module[{dir, file},
  dir = FileNameJoin[{baseDir, "deg" <> ToString[deg]}];
  If[!DirectoryQ[dir], CreateDirectory[dir, CreateIntermediateDirectories -> True]];
  file = FileNameJoin[{dir, "roots_test" <> ToString[t] <> ".txt"}];
  file
]

(* Summary files (append one per-degree line to each) *)
maxResidFile = OpenWrite[FileNameJoin[{baseDir, "max_resids.txt"}]];
avgResidFile = OpenWrite[FileNameJoin[{baseDir, "avg_resids.txt"}]];
sumResidFile = OpenWrite[FileNameJoin[{baseDir, "sum_resids.txt"}]];
avgTimesFile = OpenWrite[FileNameJoin[{baseDir, "avg_times.txt"}]];
numRootsFile = OpenWrite[FileNameJoin[{baseDir, "num_roots.txt"}]];  (* one line per degree: numTests ints *)

(* --- Force consistent one-line e-notation (ASCII), as in run_reduce_*.wls ---
   ToString[NumberForm[...]] alone writes small numbers in the two-line
   "mantissa x 10 / exponent" display form, which then reads back as garbage. *)
numFmt[x_ /; !NumericQ[x]] := "nan";  (* e.g. Indeterminate for a degree with no roots *)
numFmt[x_] := Module[{m, e, mm},
  {m, e} = MantissaExponent[N[x, 50]];
  mm = ToString[NumberForm[m, {Infinity, 17}]];
  mm <> "e" <> ToString[e]
];

For[deg = minDeg, deg <= maxDeg, deg++,

  readFileName = FileNameJoin[{
    "..",
    "sparse",
    "coeffs_mathematica_json",
    "dim" <> ToString[dim],
    "deg" <> ToString[deg],
    "num" <> ToString[nonzero] <> ".txt"
  }];
  Print["Reading ", readFileName];

  (* read as high-precision numbers to avoid downshifting during build *)
  data = Import[readFileName, "JSON"];  (* shape: {numTests, dim, deg+1, deg+1} *)
  If[!ListQ[data], Print["[error] Could not parse JSON: ", readFileName]; Exit[1]];

  (* Create root file that contains the roots for all the tests *)
  dir = FileNameJoin[{baseDir, "deg" <> ToString[deg]}];
  If[!DirectoryQ[dir], CreateDirectory[dir, CreateIntermediateDirectories -> True]];
  rfile = FileNameJoin[{dir, "roots_tests.txt"}];
  strm  = OpenWrite[rfile];
  Close[strm];

  (* Per-degree accumulators *)
  allTimes = {};
  allResids = {};
  allNumRoots = {};

  For[test = 1, test <= numTests, test++,
    solveFuncs = {};

    For[i = 1, i <= dim, i++,
      (* High-precision coefficients for Fi at this test *)
      efficientCoeffs = data[[test, i]];
      coeffMat = toCoeffMatrix[efficientCoeffs, dim, deg];
      coeffsSolve   = Flatten[coeffMat];

      (* Scale to improve conditioning; roots unchanged *)
      (*scale = Max[1, Max[Abs[coeffsOrig]]];
      coeffsSolve  = coeffsOrig/scale;*)

      (* Build both: original (for residuals) and scaled (for solving) *)
      polyExprOrig  = Block[{coeffs = coeffsSolve},  Catch[useTensor[dim, {}]] ];
      polyExprSolve = Block[{coeffs = coeffsSolve}, Catch[useTensor[dim, {}]] ];

      (* numeric evaluator for ORIGINAL polynomial: F_i[{x1,..,xd}] -> number *)
      (*myFuncs[[i]] =.; *) (* unset prior definitions safely *)
      myFuncs[[i]][myVars] := Evaluate[polyExprOrig];
      (* expression used in NSolveValues (SCALED) *)
      solveFuncs = Append[solveFuncs, polyExprSolve == 0];
    ];

    {timeVal, solList} = AbsoluteTiming @ NSolveValues[
      And @@ solveFuncs && And @@ Thread[-1 <= vars <= 1],
      vars, Reals,
      WorkingPrecision -> wp
    ];
    (* numeric list at high precision, then dedup *)
    solList = N[DeleteDuplicates[solList]];
    appendResiduals[solList];

    allTimes    = Append[allTimes, timeVal];
    allNumRoots = Append[allNumRoots, Length[solList]];

    (* Append roots for this test to the same file *)
    strm  = OpenAppend[rfile];
    WriteString[strm, "Test ", ToString[test], "\n"];
    If[Length[solList] > 0,
      Scan[(WriteString[strm, StringRiffle[numFmt /@ #, " "], "\n"]) &, solList]
    ];
    WriteString[strm, "\n"];
    Close[strm];

    (*
    (* Append roots for this test to different files *)
    rfile = rootsFileForTest[test];
    strm  = OpenAppend[rfile];
    WriteString[strm, "# Degree ", ToString[deg], "\n"];
    If[Length[solList] > 0,
      Scan[(WriteString[strm, StringRiffle[numFmt /@ #, " "], "\n"]) &, solList],
      WriteString[strm, "# (no roots)\n"]
    ];
    Close[strm];
    *)

    If[Mod[test, 10] == 0, Print["deg ", deg, ", test ", test, "/", numTests]];
  ];

  (* ---- Per-degree summaries ---- *)
  (* Max/avg over ALL equations & ALL roots; single numbers per degree *)
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
Print["Program Finished"];
Exit[];

