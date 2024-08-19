@0xb4b3505e009086af;


using ParamId = UInt32;
using PortId = UInt32;
using TermId = UInt32;
using NodeId = UInt32;
using TermPtr = UInt32;

struct Node {
  # A node in the hugr graph.

  operation @0 :Operation;
  # The operation of the node.

  params @1 :List(ParamId);
  # Parameters that are passed to the operation.
  # Each parameter is an index to a term in the term table.

  inputs @2 :List(PortId);
  # Indices of the input ports of the node.

  outputs @3 :List(PortId);
  # Indices of the output ports of the node.

  children @4 :List(NodeId);
  # Children of the node.
  # Each child is an index to a node in the node table.

  meta @5 :List(MetaItem);
  # Metadata attached to the node.
}

struct Operation {
  union {
    module @0 :Void;
    input @1 :Void;
    output @2  :Void;
    dfg @3 :Void;
    cfg @4 :Void;
    block @5 :Void;
    case @6 :Void;
    exit @7 :Void;
    defineFunc @8 :DefineFunc;
    declareFunc @9 :DeclareFunc;
    callFunc @10 :CallFunc;
    loadFunc @11 :LoadFunc;
    custom @12 :Custom;
    defineAlias @13 :DefineAlias;
    declareAlias @14 :DeclareAlias;
  }

  struct DefineFunc {
    name @0 :Text;
    type @1 :Scheme;
  }

  struct DeclareFunc {
    name @0 :Text;
    type @1 :Scheme;
  }

  struct CallFunc {
    name @0 :Text;
    args @1 :List(UInt32);
  }

  struct LoadFunc {
    name @0 :Text;
  }

  struct Custom {
    name @0 :Text;
  }

  struct DefineAlias {
    name @0 :Symbol;
    value @1 :UInt32;
  }

  struct DeclareAlias {
    name @0 :Text;
    type @1 :UInt32;
  }
}

struct MetaItem {
  name @0 :Text;
  value @1 :Term;
}

struct Port {
  type @0 :UInt32;
  meta @1 :List(MetaItem);
}

struct Term {
  union {
    type @0 :Void;
    wildcard @1 :Void;
    var @2 :Text;
    named @3 :Named;
    list @4 :TypeList;
    listType @5 :ListType;
    str @6 :Text;
    strType @7 :Void;
    nat @8 :UInt64;
    natType @9 :Void;
    extSet @10 :ExtSet;
    extSetType @11 :Void;
    tuple @12 :Tuple;
    productType @13 :ProductType;
    tagged @14 :Tagged;
    sumType @15 :SumType;
    funcType @16 :FuncType;
  }

  struct Named {
    name @0 :Text;
    args @1 :List(UInt32);
  }

  struct TypeList {
    items @0 :List(UInt32);
    tail @1 :UInt32;
  }

  struct ListType {
    itemType @0 :UInt32;
  }

  struct Tuple {
    items @0 :List(UInt32);
  }

  struct ProductType {
    types @0 :UInt32;
  }

  struct FuncType {
    inputs @0 :UInt32;
    outputs @1 :UInt32;
    extensions @2 :UInt32;
  }

  struct SumType {
    types @0 :UInt32;
  }

  struct Tagged {
    tag @0 :UInt8;
    term @1 :UInt32;
  }

  struct ExtSet {
    extensions @0 :List(Text);
    rest @1 :UInt32;
  }
}

struct Scheme {
  params @0 :List(SchemeParam);
  constraints @1 :List(Constraint);
  body @2 :UInt32;
}

struct SchemeParam {
  name @0 :TermVar;
  type @1 :UInt32;
}

struct Constraint {
  name @0 :Symbol;
  args @1 :List(UInt32);
}

struct Symbol {
  name @0 :Text;
}

struct TermVar {
  name @0 :Text;
}


struct Module {
  nodes @0 :List(Node);
  ports @1 :List(Port);
  termTable @2 :List(Term);
  terms @3: List(TermPtr);
}
