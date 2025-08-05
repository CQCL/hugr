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

function bareName() {
  return choice(
    /[0-9]+/,
    /[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*/,
    stringLiteral(),
  );
}

function sigilName(sigil) {
  return token(seq(sigil, bareName()));
}

module.exports = grammar({
  name: "hugr",

  rules: {
    source_file: ($) => repeat($.module),

    module: ($) => seq(repeat($.meta), "mod", "{", repeat($._module_item), "}"),
    _module_item: ($) =>
      choice($.symbol_import, $.symbol_function, $.symbol_ctr, $.symbol_op),

    symbol_import: ($) => seq("use", field("name", $.symbol_name), ";"),

    symbol_function: ($) =>
      seq(
        field("doc", repeat($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "fn",
        field("name", $.symbol_name),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        optional(field("signature", seq(":", $.term))),
        optional(field("body", $.region)),
        ";",
      ),

    symbol_ctr: ($) =>
      seq(
        field("doc", repeat($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "ctr",
        field("name", $.symbol_name),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        field("signature", seq(":", $.term)),
        ";",
      ),

    symbol_op: ($) =>
      seq(
        field("doc", repeat($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "op",
        field("name", $.symbol_name),
        optional(field("parameters", seq("(", commaSep1($.param), ")"))),
        optional(field("constraints", $.constraints)),
        field("signature", seq(":", $.term)),
        ";",
      ),

    pub: ($) => "pub",
    constraints: ($) => seq("where", commaSep1($.term)),

    param: ($) => seq($.var_name, ":", $.term),

    operation: ($) =>
      seq(
        field("meta", repeat($.meta)),
        optional(field("outputs", seq(commaSep1($.link_name), "="))),
        field("operation", $.term),
        optional(field("inputs", seq(commaSep($.link_name)))),
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
        field("sources", optional(seq(commaSep($.link_name), "=>"))),
        field("body", repeat($.operation)),
        field("targets", commaSep($.link_name)),
        "}",
      ),

    doc_comment: ($) => token(seq("///", /[^\r\n]*\r?\n/)),

    term: ($) =>
      choice(
        $.term_apply,
        $.func_type,
        $.var_name,
        $.term_list,
        $.term_tuple,
        $.term_parens,
        $.literal,
        $.wildcard,
      ),

    term_apply: ($) =>
      seq(
        field("symbol", $.symbol_name),
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

    // Literals
    literal: ($) => choice($.string, $.nat, $.bytes, $.float),
    string: ($) => token(stringLiteral()),
    nat: ($) => /([1-9][0-9]*)|0/,
    bytes: ($) => /b"[a-zA-Z0-9/+]+={0,2}"/,
    float: ($) => /[+-]?[0-9]+\.[0-9]+(e[+-]?[0-9]+)?/,

    // Names
    var_name: ($) => sigilName("?"),
    link_name: ($) => sigilName("%"),
    symbol_name: ($) => sigilName("@"),
    symbol_bare: ($) => bareName(),

    comment: ($) => token(seq(/\/\/[^/]/, /.*/)),
  },
  extras: ($) => [/\s/, $.comment, /\r?\n/],
});
