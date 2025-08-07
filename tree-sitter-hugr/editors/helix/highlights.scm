["mod" "fn" "ctr" "op" "use"] @keyword
(pub) @keyword
(symbol_name) @function
(var_name) @variable
(link_name) @variable

["#[" "[" "]" "{" "}" "(" ")" ":" ";" ","] @punctuation

["=>" "->"] @operator

(meta name: (symbol_bare)) @function

(string) @string
(comment) @comment
(doc_comment) @comment
(nat) @constant.numeric.integer
