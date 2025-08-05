/**
 * @file HUGR Intermediate Representation
 * @author Lukas Heidemann <lukas@heidemann.me>
 * @license MIT
 */

/// <reference types="tree-sitter-cli/dsl" />
// @ts-check

function commaSep1(rule) {
  return seq(rule, repeat(seq(",", rule)));
}

function commaSep(rule) {
  return optional(commaSep1(rule));
}

function stringLiteral() {
  return seq(
    '"',
    repeat(choice(/[^"\\\n\r]+/, /\\["\\nrt]/, /\\u\{[0-9a-fA-F]+\}/)),
    '"',
  );
}

function bareId() {
  return choice(
    /[0-9]+/,
    /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*/,
    stringLiteral(),
  );
}

function sigilId(sigil) {
  return token(seq(sigil, bareId()));
}

module.exports = grammar({
  name: "hugr",

  rules: {
    // TODO: add the actual grammar rules
    source_file: ($) => repeat($.module),

    module: ($) => seq(repeat($.meta), "mod", "{", repeat($._module_item), "}"),
    _module_item: ($) =>
      choice($.symbol_import, $.symbol_function, $.symbol_ctr, $.symbol_op),

    symbol_import: ($) => seq("use", field("name", $.symbol), ";"),

    symbol_function: ($) =>
      seq(
        field("doc", optional($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "fn",
        field("name", $.symbol),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        optional(field("signature", seq(":", $.term))),
        optional(field("body", $.region)),
        ";",
      ),

    symbol_ctr: ($) =>
      seq(
        field("doc", optional($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "ctr",
        field("name", $.symbol),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        field("signature", seq(":", $.term)),
        ";",
      ),

    symbol_op: ($) =>
      seq(
        field("doc", optional($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "op",
        field("name", $.symbol),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        field("signature", seq(":", $.term)),
        ";",
      ),

    pub: ($) => "pub",
    constraints: ($) => seq("where", commaSep1($.term)),

    param: ($) => seq($.var, ":", $.term),

    operation: ($) =>
      seq(
        field("meta", repeat($.meta)),
        optional(field("outputs", seq(commaSep1($.link), "="))),
        field("operation", $.term),
        optional(field("inputs", seq(commaSep($.link)))),
        repeat($.region),
        optional($._signature),
        ";",
      ),

    _signature: ($) => seq(":", $.term),

    region: ($) => seq(choice($.region_dfg)),
    region_dfg: ($) =>
      seq(
        "{",
        field("meta", repeat($.region_meta)),
        field("sources", optional(seq(commaSep($.link), "=>"))),
        field("body", repeat($.operation)),
        field("targets", commaSep($.link)),
        "}",
      ),

    doc_comment: ($) => repeat1(seq("///", /[^\r\n]*\r?\n/)),

    term: ($) =>
      choice(
        $.term_apply,
        $.func_type,
        $.var,
        $.term_list,
        $.term_tuple,
        $.term_parens,
        $.string,
        $.literal_nat,
        $.wildcard,
      ),

    term_apply: ($) =>
      seq(
        field("symbol", $.symbol),
        optional(field("arguments", seq("(", commaSep($.term), ")"))),
      ),
    term_list: ($) => seq("[", commaSep(choice($.term, $.splice)), "]"),
    term_tuple: ($) =>
      seq(
        "(",
        optional(seq($.term, ",", commaSep(choice($.term, $.splice)))),
        ")",
      ),
    term_parens: ($) => seq("(", $.term, ")"),

    wildcard: ($) => "_",

    splice: ($) => seq("...", $.term),

    func_type: ($) =>
      prec.right(
        1,
        seq(field("inputs", $.term), "->", field("outputs", $.term)),
      ),

    symbol_bare: ($) => /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*/,
    meta: ($) =>
      seq(
        "#[",
        field("name", $.symbol_bare),
        optional(field("arguments", seq("(", commaSep1($.term), ")"))),
        "]",
      ),
    region_meta: ($) =>
      seq(
        "#![",
        $.symbol_bare,
        optional(seq("(", commaSep1($.term), ")")),
        "]",
      ),

    string: ($) => token(stringLiteral()),

    literal_nat: ($) => /([1-9][0-9]*)|0/,

    var: ($) => sigilId("?"),
    link: ($) => sigilId("%"),
    symbol: ($) => sigilId("@"),

    comment: ($) => token(seq(/\/\/[^/]/, /.*/)),
  },
  extras: ($) => [/\s/, $.comment, /\r?\n/],
});
