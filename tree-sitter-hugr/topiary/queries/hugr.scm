[
  "="
  ":"
  "->"
  "=>"
] @prepend_space @append_space

(doc_comment) @multi_line_indent_all
(doc_comment) @leaf
(doc_comment) @append_hardline

"," @append_input_softline
";" @prepend_antispace

[
  "pub"
  "fn"
  "mod"
  "ctr"
  "op"
] @append_space



(operation) @prepend_hardline @append_hardline @allow_blank_line_before

(define_function) @prepend_hardline @append_hardline @allow_blank_line_before
(define_function parameters: "(" @append_empty_softline @append_indent_start)
(define_function parameters: ")" @prepend_empty_softline @prepend_indent_end)

(declare_ctr) @prepend_hardline @append_hardline @allow_blank_line_before
(declare_ctr parameters: "(" @append_empty_softline @append_indent_start)
(declare_ctr parameters: ")" @prepend_empty_softline @prepend_indent_end)

(declare_op) @prepend_hardline @append_hardline @allow_blank_line_before
(declare_op parameters: "(" @append_empty_softline @append_indent_start)
(declare_op parameters: ")" @prepend_empty_softline @prepend_indent_end)

(meta) @append_hardline
(comment) @append_hardline
(region_dfg
  "{" @append_spaced_softline @append_indent_start
  "}" @prepend_spaced_softline @prepend_indent_end)
(module
  "{" @append_spaced_softline @append_indent_start
  "}" @prepend_spaced_softline @prepend_indent_end)

(region) @prepend_space

(operation operation: (_) @append_space)
