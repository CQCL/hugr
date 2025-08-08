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
        optional(field("parameters", seq("[", commaSep1($.param), "]"))),
        optional(field("signature", seq(":", $.term))),
        optional(field("constraints", $.constraints)),
        optional(field("body", choice($.region, seq("{", $.region, "}")))),
        ";",
      ),

    symbol_ctr: ($) =>
      seq(
        field("doc", repeat($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "ctr",
        field("name", $.symbol_name),
        optional(field("parameters", seq("[", commaSep1($.param), "]"))),
        field("signature", seq(":", $.term)),
        optional(field("constraints", $.constraints)),
        ";",
      ),

    symbol_op: ($) =>
      seq(
        field("doc", repeat($.doc_comment)),
        field("meta", repeat($.meta)),
        field("visibility", optional($.pub)),
        "op",
        field("name", $.symbol_name),
        optional(field("parameters", seq("[", commaSep1($.param), "]"))),
        field("signature", seq(":", $.term)),
        optional(field("constraints", $.constraints)),
        ";",
      ),

    pub: ($) => "pub",
    constraints: ($) => seq("where", commaSep1($.term)),

    param: ($) => seq($.var_name, ":", $.term),

    operation: ($) =>
      seq(
        field("meta", repeat($.meta)),
        optional(field("outputs", seq(commaSep1($.typed_link), "="))),
        field("operation", $.term),
        optional(field("inputs", commaSep($.typed_link))),
        optional(seq("{", commaSep($.region), "}")),
        ";",
      ),

    _signature: ($) => seq(":", $.term),

    region: ($) => seq(choice($.region_dfg)),
    region_dfg: ($) =>
      seq(
        repeat($.meta),
        $.sources,
        "->",
        $.targets,
        "{",
        repeat($.operation),
        "}",
      ),

    typed_link: ($) => seq($.link_name, optional(seq(":", $.term))),
    sources: ($) => seq("(", commaSep($.typed_link), ")"),
    targets: ($) => seq("(", commaSep($.typed_link), ")"),

    // region_cfg: ($) =>
    //   seq(
    //     "{|",
    //     field("meta", repeat($.region_meta)),
    //     field("sources", optional(seq(commaSep($.link_name), "=>"))),
    //     field("body", repeat($.operation)),
    //     field("targets", commaSep($.link_name)),
    //     "|}",
    //   ),

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
        optional(field("arguments", seq("[", commaSep($.term), "]"))),
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

    meta: ($) => seq("#[", field("meta", $.term), "]"),

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

    _comment: ($) => token(seq(/\/\/[^/]/, /.*/)),
  },
  extras: ($) => [/\s/, $._comment, /\r?\n/],
});
