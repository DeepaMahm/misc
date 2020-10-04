solution3d[edges_, vd_, vl_, ew_] := Module[{
   uedges = UndirectedEdge @@@ edges,
   vcoords = List @@ vd,
   ew2 = KeyMap[UndirectedEdge @@ # &, ew],
   vars3d, \[Lambda], lbnd, ubnd, obj3d, err, sol, edgeLengths3d},

   Print(uedges)
   Print(ew2)

  \[Lambda] = 1/100.; lbnd = 0; ubnd = 500;
  vars3d = Array[Through[{x, y, z}@#] &, Length@vd];

  obj3d = Total[
   (EuclideanDistance[vars3d[[First[#]]], vars3d[[Last[#]]]] - ew2[#])^2 & /@ Keys[ew2]
  ] + \[Lambda]*Total[
    MapThread[EuclideanDistance[#1, #2] &, {vars3d, Values[vd]}]
  ];

  {err, sol} = Quiet[
    NMinimize[{obj3d, And @@ Thread[lbnd <= Join @@ vars3d <= ubnd]}, Flatten[vars3d]]
  ];

  edgeLengths3d = # -> EuclideanDistance[
    vars3d[[First@#]], vars3d[[Last@#]]
  ] /. sol & /@ Keys[ew2];

  Export["wolfram_output.txt", Partition[Values[sol], 3], "Table"];
  Return[<|"Error" -> err, "Solution" -> sol, "EdgeLengths" -> edgeLengths3d|>];
]
