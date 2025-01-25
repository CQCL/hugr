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
    params @3 :List(TermId);
    regions @4 :List(RegionId);
    meta @5 :List(TermId);
    signature @6 :OptionalTermId;
}

struct Operation {
    union {
        invalid @0 :Void;
        dfg @1 :Void;
        cfg @2 :Void;
        block @3 :Void;
        funcDefn @4 :FuncDecl;
        funcDecl @5 :FuncDecl;
        aliasDefn @6 :AliasDefn;
        aliasDecl @7 :AliasDecl;
        custom @8 :NodeId;
        tag @9 :UInt16;
        tailLoop @10 :Void;
        conditional @11 :Void;
        callFunc @12 :TermId;
        loadFunc @13 :TermId;
        constructorDecl @14 :ConstructorDecl;
        operationDecl @15 :OperationDecl;
        import @16 :Text;
        const @17 :TermId;
    }

    struct FuncDefn {
        name @0 :Text;
        params @1 :List(Param);
        constraints @2 :List(TermId);
        signature @3 :TermId;
    }

    struct FuncDecl {
        name @0 :Text;
        params @1 :List(Param);
        constraints @2 :List(TermId);
        signature @3 :TermId;
    }

    struct AliasDefn {
        name @0 :Text;
        params @1 :List(Param);
        type @2 :TermId;
        value @3 :TermId;
    }

    struct AliasDecl {
        name @0 :Text;
        params @1 :List(Param);
        type @2 :TermId;
    }

    struct ConstructorDecl {
        name @0 :Text;
        params @1 :List(Param);
        constraints @2 :List(TermId);
        type @3 :TermId;
    }

    struct OperationDecl {
        name @0 :Text;
        params @1 :List(Param);
        constraints @2 :List(TermId);
        type @3 :TermId;
    }
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

# Either `0` for an open scope, or the number of links in the closed scope incremented by `1`.
using LinkScope = UInt32;

enum RegionKind {
    dataFlow @0;
    controlFlow @1;
    module @2;
}

struct Term {
    union {
        wildcard @0 :Void;
        runtimeType @1 :Void;
        staticType @2 :Void;
        constraint @3 :Void;
        variable :group {
            variableNode @4 :NodeId;
            variableIndex @20 :UInt16;
        }
        apply @5 :Apply;
        const @6 :Const;
        list @7 :ListTerm;
        listType @8 :TermId;
        string @9 :Text;
        stringType @10 :Void;
        nat @11 :UInt64;
        natType @12 :Void;
        extSet @13 :ExtSet;
        extSetType @14 :Void;
        adt @15 :TermId;
        funcType @16 :FuncType;
        control @17 :TermId;
        controlType @18 :Void;
        nonLinearConstraint @19 :TermId;
        constFunc @21 :RegionId;
        constAdt @22 :ConstAdt;
        bytes @23 :Data;
        bytesType @24 :Void;
        meta @25 :Void;
        float @26 :Float64;
        floatType @27 :Void;
    }

    struct Apply {
        symbol @0 :NodeId;
        args @1 :List(TermId);
    }

    struct ApplyFull {
        symbol @0 :NodeId;
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

    struct ConstAdt {
        tag @0 :UInt16;
        values @1 :TermId;
    }

    struct FuncType {
        inputs @0 :TermId;
        outputs @1 :TermId;
        extensions @2 :TermId;
    }

    struct Const {
        type @0 :TermId;
        extensions @1 :TermId;
    }
}

struct Param {
    name @0 :Text;
    type @1 :TermId;
}
