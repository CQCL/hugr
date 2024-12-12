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
    meta @3 :List(MetaItem);
    signature @4 :OptionalTermId;
}

struct Operation {
    union {
        invalid @0 :Void;
        dfg @1 :RegionId;
        cfg @2 :RegionId;
        block @3 :RegionId;

        funcDefn :group {
            name @4 :Text;
            params @5 :List(Param);
            constraints @6 :List(TermId);
            signature @7 :TermId;
            body @8 :RegionId;
        }

        funcDecl :group {
            name @9 :Text;
            params @10 :List(Param);
            constraints @11 :List(TermId);
            signature @12 :TermId;
        }

        aliasDefn :group {
            name @13 :Text;
            params @14 :List(Param);
            type @15 :TermId;
            value @16 :TermId;
        }

        aliasDecl :group {
            name @17 :Text;
            params @18 :List(Param);
            type @19 :TermId;
        }

        custom :group {
            operation @20 :NodeId;
            params @21 :List(TermId);
        }

        customFull :group {
            operation @22 :NodeId;
            params @23 :List(TermId);
        }

        constructorDecl :group {
            name @24 :Text;
            params @25 :List(Param);
            constraints @26 :List(TermId);
            type @27 :TermId;
        }

        operationDecl :group {
            name @28 :Text;
            params @29 :List(Param);
            constraints @30 :List(TermId);
            type @31 :TermId;
        }

        tag @32 :UInt16;
        tailLoop @33 :RegionId;
        conditional @34 :List(RegionId);
        callFunc @35 :TermId;
        loadFunc @36 :TermId;
        import @37 :Text;
        const @38 :TermId;
    }
}

struct Region {
    kind @0 :RegionKind;
    sources @1 :List(LinkIndex);
    targets @2 :List(LinkIndex);
    children @3 :List(NodeId);
    meta @4 :List(MetaItem);
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

struct MetaItem {
    name @0 :Text;
    value @1 :UInt32;
}

struct Term {
    union {
        wildcard @0 :Void;
        runtimeType @1 :Void;
        staticType @2 :Void;
        constraint @3 :Void;
        variable :group {
            variableNode @4 :NodeId;
            variableIndex @5 :UInt16;
        }
        apply :group {
            symbol @6 :NodeId;
            args @7 :List(TermId);
        }
        applyFull :group {
            symbol @8 :NodeId;
            args @9 :List(TermId);
        }
        const @10 :TermId;
        list @11 :List(ListPart);
        listType @12 :TermId;
        string @13 :Text;
        stringType @14 :Void;
        nat @15 :UInt64;
        natType @16 :Void;
        extSet @17 :List(ExtSetPart);
        extSetType @18 :Void;
        adt @19 :TermId;
        funcType :group {
            inputs @20 :TermId;
            outputs @21 :TermId;
            extensions @22 :TermId;
        }
        control @23 :TermId;
        controlType @24 :Void;
        nonLinearConstraint @25 :TermId;
        constFunc @26 :RegionId;
    }

    struct ListPart {
        union {
            item @0 :TermId;
            splice @1 :TermId;
        }
    }

    struct ExtSetPart {
        union {
            extension @0 :Text;
            splice @1 :TermId;
        }
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
