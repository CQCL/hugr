@0xe02b32c528509601;

struct Module {
    root @0 :UInt32;
    nodes @1 :List(Node);
    regions @2 :List(Region);
    terms @3 :List(Term);
}

struct Node {
    operation @0 :Operation;
    inputs @1 :List(LinkRef);
    outputs @2 :List(LinkRef);
    params @3 :List(UInt32);
    regions @4 :List(UInt32);
    meta @5 :List(MetaItem);
    signature @6 :UInt32;
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
        custom @8 :GlobalRef;
        customFull @9 :GlobalRef;
        tag @10 :UInt16;
        tailLoop @11 :Void;
        conditional @12 :Void;
        callFunc @13 :UInt32;
        loadFunc @14 :UInt32;
    }

    struct FuncDefn {
        name @0 :Text;
        params @1 :List(Param);
        signature @2 :UInt32;
    }

    struct FuncDecl {
        name @0 :Text;
        params @1 :List(Param);
        signature @2 :UInt32;
    }

    struct AliasDefn {
        name @0 :Text;
        params @1 :List(Param);
        type @2 :UInt32;
        value @3 :UInt32;
    }

    struct AliasDecl {
        name @0 :Text;
        params @1 :List(Param);
        type @2 :UInt32;
    }
}

struct Region {
    kind @0 :RegionKind;
    sources @1 :List(LinkRef);
    targets @2 :List(LinkRef);
    children @3 :List(UInt32);
    meta @4 :List(MetaItem);
    signature @5 :UInt32;
}

enum RegionKind {
    dataFlow @0;
    controlFlow @1;
}

struct MetaItem {
    name @0 :Text;
    value @1 :UInt32;
}

struct LinkRef {
    union {
        id @0 :UInt32;
        named @1 :Text;
    }
}

struct GlobalRef {
    union {
        node @0 :UInt32;
        named @1 :Text;
    }
}

struct LocalRef {
    union {
        direct :group {
            index @0 :UInt16;
            node @1 :UInt32;
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
        quote @7 :UInt32;
        list @8 :ListTerm;
        listType @9 :UInt32;
        string @10 :Text;
        stringType @11 :Void;
        nat @12 :UInt64;
        natType @13 :Void;
        extSet @14 :ExtSet;
        extSetType @15 :Void;
        adt @16 :UInt32;
        funcType @17 :FuncType;
        control @18 :UInt32;
        controlType @19 :Void;
    }

    struct Apply {
        global @0 :GlobalRef;
        args @1 :List(UInt32);
    }

    struct ApplyFull {
        global @0 :GlobalRef;
        args @1 :List(UInt32);
    }

    struct ListTerm {
        items @0 :List(UInt32);
        tail @1 :UInt32;
    }

    struct ExtSet {
        extensions @0 :List(Text);
        rest @1 :UInt32;
    }

    struct FuncType {
        inputs @0 :UInt32;
        outputs @1 :UInt32;
        extensions @2 :UInt32;
    }
}

struct Param {
    union {
        implicit @0 :Implicit;
        explicit @1 :Explicit;
        constraint @2 :UInt32;
    }

    struct Implicit {
        name @0 :Text;
        type @1 :UInt32;
    }

    struct Explicit {
        name @0 :Text;
        type @1 :UInt32;
    }
}
