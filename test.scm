(version 2)

(hugr my-hugr-module
  (@core/module [] []
    ((@core/func-defn @circuit) [] []
      (@core/input [] [%0])
      ((@core-f32/const 2) [] [%1])
      (@core-f32/mul [%0 %1] [%2])
      (@core/output [%2] []))))
