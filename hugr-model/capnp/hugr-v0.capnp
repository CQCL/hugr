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

struct Module {
    root @0 :RegionId;
    nodes @1 :List(Node);
    regions @2 :List(Region);
    terms @3 :List(Term);
}

struct Node {
    union {
        invalid @0 :Void;
        instruction @1 :Instruction;
        symbol @2 :Symbol;
    }
}

struct Instruction {
    inputs @0 :List(LinkRef);
    outputs @1 :List(LinkRef);
    params @2 :List(TermId);
    regions @3 :List(RegionId);
    meta @4 :List(MetaItem);
    signature @5 :OptionalTermId;

    union {
        dfg @6 :Void;
        cfg @7 :Void;
        block @8 :Void;
        custom @9 :GlobalRef;
        customFull @10 :GlobalRef;
        tag @11 :UInt16;
        tailLoop @12 :Void;
        conditional @13 :Void;
        callFunc @14 :TermId;
        loadFunc @15 :TermId;
    }
}

struct Symbol {
    name @0 :Text;
    params @1 :List(Param);
    constraints @2 :List(TermId);
    signature @3 :OptionalTermId;
    meta @4 :List(MetaItem);

    union {
        import @5 :Void;
        funcDefn @6 :RegionId;
        funcDecl @7 :Void;
        aliasDefn @8 :TermId;
        aliasDecl @9 :Void;
        constructorDecl @10 :Void;
        operationDecl @11 :Void;
    }
}

struct Region {
    kind @0 :RegionKind;
    sources @1 :List(LinkRef);
    targets @2 :List(LinkRef);
    symbols @3 :List(NodeId);
    instructions @4 :List(NodeId);
    meta @5 :List(MetaItem);
    signature @6 :OptionalTermId;
}

enum RegionKind {
    dataFlow @0;
    controlFlow @1;
    module @2;
}

struct MetaItem {
    name @0 :Text;
    value @1 :UInt32;
}

struct LinkRef {
    union {
        id @0 :LinkId;
        named @1 :Text;
    }
}

struct GlobalRef {
    union {
        node @0 :NodeId;
        named @1 :Text;
    }
}

struct LocalRef {
    union {
        direct :group {
            index @0 :UInt16;
            node @1 :NodeId;
        }
        named @2 :Text;
    }
}

struct Term {
    union {
        wildcard @0 :Void;
        runtimeType @1 :Void;
        staticType @2 :Void;
        constraint @3 :Void;
        variable @4 :LocalRef;
        apply @5 :Apply;
        applyFull @6 :ApplyFull;
        quote @7 :TermId;
        list @8 :ListTerm;
        listType @9 :TermId;
        string @10 :Text;
        stringType @11 :Void;
        nat @12 :UInt64;
        natType @13 :Void;
        extSet @14 :ExtSet;
        extSetType @15 :Void;
        adt @16 :TermId;
        funcType @17 :FuncType;
        control @18 :TermId;
        controlType @19 :Void;
        nonLinearConstraint @20 :TermId;
    }

    struct Apply {
        global @0 :GlobalRef;
        args @1 :List(TermId);
    }

    struct ApplyFull {
        global @0 :GlobalRef;
        args @1 :List(TermId);
    }

    struct ListTerm {
        items @0 :List(ListPart);
    }

    struct ListPart {
        union {
            item @0 :TermId;
            splice @1 :TermId;
        }
    }

    struct ExtSet {
        items @0 :List(ExtSetPart);
    }

    struct ExtSetPart {
        union {
            extension @0 :Text;
            splice @1 :TermId;
        }
    }

    struct FuncType {
        inputs @0 :TermId;
        outputs @1 :TermId;
        extensions @2 :TermId;
    }
}

struct Param {
    name @0 :Text;
    type @1 :TermId;
    sort @2 :ParamSort;
}

enum ParamSort {
    implicit @0;
    explicit @1;
}
