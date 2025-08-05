["mod" "fn" "pub" "ctr" "op" "use"] @keyword
(symbol) @function
(var) @variable
(link) @variable

["#[" "[" "]" "{" "}" "(" ")" ":" ";" ","] @punctuation

["=>" "->"] @operator

(meta name: (symbol_bare)) @function

(string) @string
(comment) @comment
(doc_comment) @comment
(literal_nat) @constant.numeric.integer
