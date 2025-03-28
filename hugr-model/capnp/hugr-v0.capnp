@0xe02b32c528509601;

# The id of a `Term`.
using TermId = UInt32;

# Either `0` or the id of a `Term` incremented by one.
using OptionalTermId = UInt32;

# The id of a `Region`.
using RegionId = UInt32;

# The id of a `Node`.
using NodeId = UInt32;

# The id of a `Link`.
using LinkId = UInt32;

# The index of a `Link`.
using LinkIndex = UInt32;

struct Package {
    modules @0 :List(Module);
}

struct Module {
    root @0 :RegionId;
    nodes @1 :List(Node);
    regions @2 :List(Region);
    terms @3 :List(Term);
}

struct Node {
    operation @0 :Operation;
    inputs @1 :List(LinkIndex);
    outputs @2 :List(LinkIndex);
    regions @3 :List(RegionId);
    meta @4 :List(TermId);
    signature @5 :OptionalTermId;
}

struct Operation {
    union {
        custom @0 :TermId;
        dfg @1 :Void;
        cfg @2 :Void;
        block @3 :Void;
        funcDefn @4 :Symbol;
        funcDecl @5 :Symbol;
        aliasDefn :group {
            symbol @6 :Symbol;
            value @14 :TermId;
        }
        aliasDecl @7 :Symbol;
        invalid @8 :Void;
        tailLoop @9 :Void;
        conditional @10 :Void;
        import @11 :Text;
        constructorDecl @12 :Symbol;
        operationDecl @13 :Symbol;
    }
}

struct Symbol {
    name @0 :Text;
    params @1 :List(Param);
    constraints @2 :List(TermId);
    signature @3 :TermId;
}

struct Region {
    kind @0 :RegionKind;
    sources @1 :List(LinkIndex);
    targets @2 :List(LinkIndex);
    children @3 :List(NodeId);
    meta @4 :List(TermId);
    signature @5 :OptionalTermId;
    scope @6 :RegionScope;
}

struct RegionScope {
    links @0 :UInt32;
    ports @1 :UInt32;
}

enum RegionKind {
    dataFlow @0;
    controlFlow @1;
    module @2;
}

struct Term {
    union {
        apply :group {
            symbol @0 :NodeId;
            args @1 :List(TermId);
        }
        variable :group {
            node @2 :NodeId;
            index @3 :UInt16;
        }
        list @4 :List(SeqPart);
        string @5 :Text;
        nat @6 :UInt64;
        bytes @7 :Data;
        float @8 :Float64;
        func @9 :RegionId;
        wildcard @10 :Void;
        tuple @11 :List(SeqPart);
    }

    struct SeqPart {
        union {
            item @0 :TermId;
            splice @1 :TermId;
        }
    }
}

struct Param {
    name @0 :Text;
    type @1 :TermId;
}
