# HUGR design document, Draft 4

The Hierarchical Unified Graph Representation (HUGR, pronounced *hugger*
![(blue star)](images/icons/emoticons/72/1fac2.png)) is a proposed new
common internal representation used across TKET2, Tierkreis, and the L3
compiler. The HUGR project aims to give a faithful representation of
operations, that facilitates compilation and encodes complete programs,
with subprograms that may execute on different (quantum and classical)
targets.

## Motivation

Multiple compilers and tools in the Quantinuum stack use some graph-like
program representation; be it the quantum circuits encoded as DAGs in
TKET, or the higher-order executable dataflow graphs in Tierkreis.

The goal of the HUGR representation is to provide a unified structure
that can be shared between the tools, allowing for more complex
operations such as TKET optimizations across control-flow blocks, and
nested quantum and classical programs in a single graph. 
<!--
For more see
the initial proposal: [The Grand Graph
Unification](https://cqc.atlassian.net/wiki/spaces/TKET/pages/2506260512/The+Grand+Graph+Unification).
-->
The HUGR should provide a generic graph representation of a program,
where each node contains a specific kind of operation and wires
represent (typed) data or control dependencies.

### Goals

  - Modular design, allowing new operations, data types, and rewrite
    methods defined by third-parties.

  - Represent mixed quantum-classical programs, allowing for efficient
    lowering through bespoke compilation to dedicated targets.

  - Efficiently serializable. Different tools should be able to send and
    receive HUGRs via a serialized interface when sharing the in-memory
    structure is not possible.

  - Provide a common interface for rewrite operations with support for
    opaque types.  

### Non-goals

  - Translations to other representations. While the HUGR should be able
    to encode programs in languages such as QIR, the translation should
    be implemented by separate crates.

  - Execution, or any kind of interpretation of the program. The HUGR
    describes the graph representation and control flow, without fixing
    the semantics of any extension operations defined outside the core
    set in this document, which will be most in actual use.

### Main requirements

  - A directed graph structure with extensible operation types in the
    nodes and data types in the edges.

  - Indexed connection ports for each operation node, which may be
    connected to another port with the same data type or remain
    unconnected.

  - Control-flow support with ability to capture both LLVM SSACFG style
    programs and programs from future front-ends designed to target
    HUGR. These including the upcoming Python eDSL for quantum-classical
    programming, and BRAT (which already uses an internal graph-like
    representation for classical functional programs and quantum
    kernels). We expect that these front-ends will provide
    programmer-facing control flow constructs that map to the preferred
    constructs in HUGR without first having to pass through an
    LLVM/SSACFG intermediate.

  - Support for nested structures. The nodes form a tree-like hierarchy
    with nested graphs encoded as children of their containing node.

  - User-defined metadata, such as debug information, can be efficiently
    attached to nodes and queried.

  - All user-provided information can be encoded and decoded in a stable
    (versioned) efficient serialized format.

  - A type system for checking valid operation connectivity + (nice to
    have) only operations supported on specific targets are used.

  - A space efficient and user friendly specification of a subgraph and
    replacement graph, along with an efficient routine for performing
    the replacement.

  - Be suitable for targeting Quantinuum NG-RTE for “quantum”
    subprograms:
    
      - Conversion of the graph with loops into LLVM IR for final DFL
        production should be possible so that we can link and compile
        user programs on HXT and future platforms.
    
      - For supporting the various output formats for user programs
        written in different input languages when executing on hardware,
        there must be some way to transform the HUGR into print function
        calls with constant string arguments and primitive data
        (integers or floating point numbers).
    
      - Debug data should be representable as meta-data on graph nodes
        so that hardware can report operations that may fail at runtime
        (for things like div instructions that can fail).
    
      - Everything representable in the immediate future QIR profile
        (internal function definitions, declared “extern” functions,
        looping via backwards branches, external and internal function
        calls, int arithmetic instructions, floating point arithmetic
        instructions, and programs with extern values to be linked in
        later (for parameterized circuits)) should be embeddable into
        the HUGR.

## Functional description

A HUGR is a directed graph with nodes and edges. The nodes represent
processes that produce values - either statically, i.e. at compile time,
or at runtime. Each node is uniquely identified by its **node index**,
although this may not be stable under graph structure modifications.
Each node is defined by its **operation**; the possible operations are
outlined in [Node
Operations](#node-operations)
but may be [extended by
Resources](#operation-extensibility).
The edges encode relationships between nodes; there are several *kinds*
of edge for different relationships, and some edges have types:
```
EdgeKind ::= Hierarchy | Value(Locality, SimpleType) | Order | ConstE(ClassicType) | ControlFlow

Locality ::= Local | Ext | Dominator
```
A **Hierarchy** edge from node *a* to *b* encodes that *a* is the direct
parent of *b*. Only certain nodes, known as *container* nodes, may act
as parents - these are listed in
[hierarchical node relationships](#hierarchical-relationships-and-constraints).
In a valid HUGR the hierarchy edges form a tree joining all nodes of the
HUGR, with the unique
[Module](#module)
node as root.

A **sibling graph** is a subgraph of the HUGR containing all nodes with
a particular parent, plus the Order, Value and ControlFlow edges between
them.

A **Value** edge represents dataflow that happens at runtime - i.e. the
source of the edge will, at runtime, produce a value that is consumed by
the edge’s target. Value edges are from an outgoing **Port** of the
source node, to an incoming **Port** of the target node; each port may
have at most one edge, and the port types of a node are described by its
**Signature**. (In fact, each port must have exactly one edge, but ports
whose edge has not yet been specified, may be useful as an intermediate
form whilst building a HUGR.) The **Signature** may also specify a row
of `ClassicType`s for incoming `ConstE` edges. **TODO** “…and the
relevant ports must have the same type”? Does the incoming port repeat
the resource requirement of the outgoing port? Or are resources a
property of the node?

**Inport**: an incoming port

**Outport**: an outgoing port

Value edges are parameterized by the locality and type; there are three
possible localities:

  - Local: both source and target nodes must have the same parent

  - Ext: edges “in” from an ancestor, i.e. where parent(src) ==
    parent<sup>i</sup>(dest) for i\>1; see
    [inter-graph-edges](#inter-graph-value-edges).

  - Dom: edges from a dominating basic block in a control-flow graph
    that is the parent of the source; see
    [inter-graph-edges](#inter-graph-value-edges)

Note that the locality is not fixed or even specified by the signature.

Simple HUGR example

![Quantum
circuit with a Hadamard and CNOT
operation](attachments/2647818241/2647818473.svg?width=442)

In the example above, a 2-qubit circuit is described as a dataflow
region of a HUGR with one `H` operation and one `CNOT` operation. The
operations have an incoming and outgoing list of ports, with each
element identified by its offset and labelled with a type.

The signature of the `CNOT` operation is `[Qubit, Qubit] → [Qubit,
Qubit]`. Further information in the metadata may label the first qubit
as *control* and the second as *target*.

In this case, output 0 of the H operation is connected to input 0 of the
CNOT. All other ports are disconnected.

**Order** edges represent constraints on ordering that may be specified
explicitly (e.g. for operations that are stateful). These can be seen as
local value edges of unit type `()`, i.e. that pass no data, and where
the source and target nodes must have the same parent. There can be at
most one Order edge between any two nodes.

A **ConstE** edge represents dataflow that is statically knowable - i.e.
the source is a compile-time constant. (Hence, the types on these edges
do not include a resource specification.) Only a few nodes may be
sources (`def` and `const`) and targets (`call` and `load_const`) of
these edges; see
[module](#module)
and
[functions](#functions).
For a ConstE edge from *a* to *b,* we require parent(*a*) ==
parent<sup>i</sup>(*b*) for i\>=1 to satisfy valid scoping.

Finally, **ControlFlow** edges represent all possible flows of control
from one region (basic block) of the program to another. These are
always *local*, i.e. source and target have the same parent.

### Node Operations

Here we describe we define some core operations required to represent
full programs, including dataflow operations (in
[functions](#functions)).

#### Module

At the top level of the of the hierarchy is a single `module` node, the
weight attached to this node contains module level data. There may also
be additional metadata (e.g. source file, module name). The children of
a `module` correspond to "module level" operation types. Neither
`module` nor these module-level operations have signatures or value
ports, but some have constE or other edges.

Taking lots of inspiration from the MLIR
[builtin](https://mlir.llvm.org/docs/Dialects/Builtin/) and
[func](https://mlir.llvm.org/docs/Dialects/Func/) dialects, these node
operations include:

  - `constN<T>` : a static constant value of type T stored in the node
    weight (perhaps a computation of some `Graph` type represented as a
    HUGR). Has no ports, but may have any number of `ConstE<T>`
    out-edges - one for each use.

  - `def` : a function definition. The name of the function is specified
    in the metadata and function attributes (relevant for compilation)
    define the node weight. The function body is defined by its children
    (the child graph forms the body). The node has no ports but may have
    any number of `ConstE<Graph>` out-edges - one for each use.

  - `declare`: an external function declaration. Like `def`, but with no
    body, the name is used at link time to lookup definitions in linked
    modules (other hugr instances specified to the linker).

  - `alias_declare/alias_def`: analogous to `declare` and `def` but with
    type aliases. At link time `alias_declare` can be replaced with
    `alias_def`. An alias declared with `declare` is equivalent to a
    named opaque type.

Exactly which nodes are valid at this top level is dependent on the
compiler and target. Note that the operations defined can also be
defined in graphs lower in the hierarchy - this limits the scope within
which they can be used.

A **loadable HUGR** is one where all edges are connected and there are
no `declare/alias_declare` nodes.

An **executable HUGR** or **executable module** is a loadable HUGR where
the first child of the root `module` is a `def` called “main”, that is
the designated entry point. Modules that act as libraries need not be
executable.

Even non-loadable HUGRs are HUGRs so long as they satisfy (all) other
requirements such as acyclicity. (Anything not satisfying those is
not-a-HUGR.) For example, such may be processed by the linker to produce
loadable HUGRs.

In
[replacement-and-pattern-matching](#replacement-and-pattern-matching)
we describe a “partial HUGR” - this is *not* a HUGR, though it is
related.

#### Functions

Within functions the following basic dataflow operations are available,
with signatures describing their value ports (note that some operations
support many different signatures. For example, optimization may add
additional outputs to a classical copy node):

  - `Input/Output`: input/output nodes, the outputs of `Input` node are
    the inputs to the function, and the inputs to `Output` are the
    outputs of the function. In a data dependency subgraph, a valid
    ordering of operations can be achieved by topologically sorting the
    nodes starting from `Input` with respect to the Value and Order
    edges.

  - `call`: Call a function directly. There is an incoming
    `ConstE<Graph>` edge to specify the graph being called. The
    signature of the `Value` edges matches the function being called.

  - `load_constant<T>`: has an incoming `ConstE<T>` edge, and a
    `Value<*,T>` output, used to load a static constant in to the local
    dataflow graph. They also have an incoming `Order` edge connecting
    them to the `Input` node, as should all stateful operations that
    take no dataflow input, to ensure they lie in the causal cone of the
    `Input` node when traversing.

  - `copy<T, N>`: explicit copy, has a single `Value<*,T>` input, and
    `N` `Value<*,T>` outputs, where `N` \>=0. A `copy<T, 0>` is
    interpreted as a discard. A `copy<T,1>` is an identity operation and
    can be trivially removed.

  - `identity<T>`: pass-through, no operation is performed.

  - `lookup<T,N>`, where T in {Int, Nat} and N\>0. Has a `Value<*,T>`
    input, and a single `Value<*,Sum((),...,())>` output with N elements
    each of type unit `()`. The value is (1) a list of pairs of type
    `(T,Sum((),...,())` used as a lookup table on the input value, the
    first element being key and the second as the return value; and (2)
    an optional default value of the same `Sum` type.

  - `DFG`: a simply nested dataflow graph, the signature of this
    operation is the signature of the child graph. These nodes are
    parents in the hierarchy.

![](attachments/2647818241/2647818467.png)

#### Control Flow

In a dataflow graph, the evaluation semantics are simple: all nodes in
the graph are necessarily evaluated, in some order (perhaps parallel)
respecting the dataflow edges. The following operations are used to
express control flow, i.e. conditional or repeated evaluation.

##### `Conditional` nodes

These are parents to multiple `Case` nodes; the children have no edges.
The first input to the Conditional-node is of Predicate type, whose
arity matches the number of children of the Conditional-node. At runtime
the constructor (tag) selects which child to execute; the unpacked
contents of the Predicate with all remaining inputs to Conditional
appended are sent to this child, and all outputs of the child are the
outputs of the Conditional; that child is evaluated, but the others are
not. That is, Conditional-nodes act as "if-then-else" followed by a
control-flow merge.

A **Predicate(T0, T1…TN)** type is an algebraic “sum of products” type,
defined as `Sum(Tuple(#t0), Tuple(#t1), ...Tuple(#tn))` (see [type
system](#type-system)), where `#ti` is the *i*th Row defining it.

**TODO: update below diagram now that Conditional is “match”**

![](attachments/2647818241/2647818344.png)

##### `TailLoop` nodes

These provide tail-controlled loops: the data sibling graph within the
TailLoop-node computes a value of 2-ary `Predicate(#i, #o)`; the first
variant means to repeat the loop with the values of the tuple unpacked
and “fed” in at at the top; the second variant means to exit the loop
with those values unpacked. The graph may additionally take in a row
`#x` (appended to `#i`) and return the same row (appended to `#o`). The
contained graph may thus be evaluated more than once.

**Alternate TailLoop**

It is unclear whether this exact formulation of TailLoop is the most
natural or useful. It may be that compilation typically uses multiple.
Another is:

A node with type `I -> O` with three children of types `A: I -> p + F`
(the output is the row formed by extending the row `F` with the boolean
output `p`), `B: F-> O` and `C: F -> I`. This node offers similar
benefits to the option above, exchanging the machinery of variants for
having a node with 3 children rather than 1. The semantics of the node
are:

1.  Execute A, outputting the boolean `p` and some outputs `F`

2.  If `p`, execute B with inputs `F` and return the output `O`

3.  Else execute `C` with inputs `F` and then restart loop with inputs
    `I`

##### Control Flow Graphs

When Conditional and `TailLoop` are not sufficient, the HUGR allows
arbitrarily-complex (even irreducible) control flow via an explicit CFG,
expressed using `ControlFlow` edges between `BasicBlock`-nodes that are
children of a CFG-node.

  - `BasicBlock` nodes are CFG basic blocks. Edges between them are
    control-flow (as opposed to dataflow), and express traditional
    control-flow concepts of branch/merge. Each `BasicBlock` node is
    parent to a dataflow sibling graph. `BasicBlock`-nodes only exist as
    children of CFG-nodes.

  - `CFG` nodes: a dataflow node which is defined by a child control
    sibling graph. All children except the last are `BasicBlock`-nodes,
    the first of which is the entry block. The final child is an
    `ExitBlock` node, which has no children, this is the single exit
    point of the CFG and the inputs to this node match the outputs of
    the CFG-node. The inputs to the CFG-node are wired to the inputs of
    the entry block.

The first output of the DSG contained in a `BasicBlock` has type
`Predicate(#t0, #t1,...#tn)`, where the node has `N` successors, and the
remaining outputs are a row `#x`. `#ti` with `#x` appended matches the
inputs of successor `i`.

Some normalizations are possible:

  - If the entry node has no predecessors (i.e. is not a loop header),
    then its contents can be moved outside the ??-node into a containing
    DSG.

  - If the entry node’s has only one successor and that successor is the
    exit node, the CFG-node itself can be removed

**Example CFG (TODO update w/ Sum types)**

![](attachments/2647818241/2647818461.png)

#### Hierarchical Relationships and Constraints

To clarify the possible hierarchical relationships, using the operation
definitions above and also defining “*O”* to be all non-nested dataflow
operations, we can define the relationships in the following table.
**D** and **C** are useful (and intersecting) groupings of operations:
dataflow nodes and the nodes which contain them.

| **Hierarchy**             | **Edge kind**                  | **Node Operation** | **Parent**    | **Children (\>=1)**      | **Child Constraints**                    |
| ------------------------- | ------------------------------ | ------------------ | ------------- | ------------------------ | ---------------------------------------- |
| Leaf                      | **D:** Value (Data dependency) | O, `Input/Output`  | **C**         | \-                       |                                          |
| CFG container             | "                              | CFG                | "             | `BasicBlock`/`ExitBlock` | First(last) is entry(exit)               |
| Conditional               | "                              | `Conditional`      | "             | `Case`                   | No edges                                 |
| **C:** Dataflow container | "                              | `TailLoop`         | "             |  **D**                   | First(last) is `Input`(`Output`)         |
| "                         | "                              | `DFG`              | "             |  "                       | "                                        |
| "                         | Const                          | `def`              | "             |  "                       | "                                        |
| "                         | ControlFlow                    | `BasicBlock`       | CFG           |  "                       | "                                        |
| "                         | \-                             | `Case`             | `Conditional` |  "                       | "                                        |
| "                         | \-                             | `module`           | \-            |  "                       | First is main `def` for executable HUGR. |

These relationships allow to define two common varieties of sibling
graph:

**Control Flow Sibling Graph (CSG)**: where all nodes are
`BasicBlock`-nodes, and all edges are control-flow edges, which may have
cycles. The common parent is a CFG-node.

**Dataflow Sibling Graph (DSG)**: nodes are operations, `CFG`,
`Conditional`, `TailLoop` and `DFG` nodes; edges are value and order and
must be acyclic. The common parent may be a `def`, `TailLoop`, `DFG`,
`Case` or `BasicBlock` node.

In a dataflow sibling graph, the edges (value and order considered
together) must be acyclic. There is a unique Input node and Output node.
All nodes must be reachable from the Input node, and must reach the
Output node.

| **Edge Kind**  | **Hierarchical Constraints**                                                                                                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hierarchy      | Defines hierarchy; each node has \<=1 parent                                                                                                                                                            |
| Order, Control | Source + target have same parent                                                                                                                                                                        |
| Value          | For local edges, source + target have same parent, but there are [inter-graph edges](#inter-graph-value-edges) |
| ConstE         | Parent of source is ancestor of target                                                                                                                                                                  |

### Exception Handling

#### Panic

  - Any operation may panic, e.g. integer divide when denominator is
    zero

  - Panicking aborts the current graph, and recursively the container
    node also panics, etc.

  - Nodes that are independent of the panicking node may have executed
    or not, at the discretion of the runtime/compiler.

  - If there are multiple nodes that may panic where neither has
    dependences on the other (including Order edges), it is at the
    discretion of the compiler as to which one panics first

#### `ErrorType`

  - There is some type of errors, perhaps just a string, or
    `Tuple(Int,String)` with some errorcode, that is returned along with
    the fact that the graph/program panicked.

#### Catch

  - At some point we expect to add a first-order `catch` node, somewhat
    like a DFG-node. This contains a DSG, and (like a DFG node) has
    inputs matching the child DSG; but one output, of type
    `Sum(O,ErrorType)` where O is the outputs of the child DSG.
    
      - At this point L3 will have to compile potentially-panicking
        operations into an explicit check and branch to the end (exit
        block) of the nearest containing `catch`

  - There is also a higher-order `catch` operation in the Tierkreis
    resource, taking a graph argument; and `run_circuit` will return the
    same way.

#### **Inter-Graph Value Edges**

**For classical values only** we allow value edges
n<sub>1</sub>→n<sub>2</sub> where parent(n<sub>1</sub>) \!=
parent(n<sub>2</sub>) when the edge's locality is either Ext or Dom, as
follows:

Specifically, these rules allow for edges where in a given execution of
the HUGR the source of the edge executes once, but the target may
execute \>=0 times.

1.  For Ext edges, ** we require parent(n<sub>1</sub>) ==
    parent<sup>i</sup>(n<sub>2</sub>) for some i\>1 *and* there must be
    a order edge from parent(n<sub>1</sub>) to
    parent<sup>i-1</sup>(n<sub>2</sub>). The order edge records the
    ordering requirement that results, i.e. it must be possible to
    execute the entire n<sub>1</sub> node before executing
    parent<sup>i-1</sup>(n<sub>2</sub>). (Further recall that
    order+value edges together must be acyclic). We record the
    relationship between the inter-graph value edge and the
    corresponding order edge via metadata on each edge.

2.  For Dom edges, we must have that parent<sup>2</sup>(n<sub>1</sub>)
    == parent<sup>i</sup>(n<sub>2</sub>) is a CFG-node, for some i\>1,
    **and** parent(n<sub>1</sub>) strictly dominates
    parent<sup>i-1</sup>(n<sub>2</sub>) in the CFG (strictly as in
    parent(n<sub>1</sub>) \!= parent<sup>i-1</sup>(n<sub>2</sub>). (The
    i\>1 allows the node to target an arbitrarily-deep descendant of the
    dominated block, similar to an Ext edge.)

![](attachments/2647818241/2647818338.png)

This mechanism allows for some values to be passed into a block
bypassing the input/output nodes, and we expect this form to make
rewrites easier to spot. The constraints on input/output node signatures
remain as before.

HUGRs without inter-graph edges may still be useful for e.g. register
allocation, as that representation makes storage explicit. For example,
when a true/false subgraph of a Conditional-node wants a value from the
outside, we add an outport to the Input node of each subgraph, a
corresponding inport to the Conditional-node, and discard nodes to each
subgraph that doesn’t use the value. It is straightforward to turn an
edge between graphs into a combination of intra-graph edges and extra
input/output ports+nodes in such a way, but this is akin to
decompression.

Conversion from intra-graph edges to a smallest number of total edges
(using inter-graph edges to reduce their number) is much more complex,
akin to compression, as it requires elision of useless split-merge
diamonds and other patterns and will likely require computation of
(post/)dominator trees. (However this will be somewhat similar to the
analysis required to move computations out of a CFG-node into
Conditional- and TailLoop-nodes). Note that such conversion could be
done for only a subpart of the HUGR at a time.

**Example CFG (TODO update with** `Sum` **types)** the following CFG is
equivalent to the previous example. Besides the use of inter-block
edges to reduce passing of P and X, I have also used the normalization
of moving operations out of the exit-block into the surrounding graph;
this results in the qubit being passed right through so can also be
elided. Further normalization of moving F out of the entry-block into
the surrounding graph is also possible. Indeed every time a SESE region
is found within a CFG (where block *a* dominates *b*, *b* postdominates
*a*, and every loop containing either *a* or *b* contains both), it can
be normalized by moving the region bracketted by *a…b* into its own
CFG-node.

![](attachments/2647818241/2647818458.png)

### Operation Extensibility

#### Goals and constraints

The goal here is to allow the use of operations and types in the
representation that are user defined, or defined and used by extension
tooling. Here “extension tooling” can be our own, e.g. TKET2 or
Tierkreis. These operations cover various flavours:

  - Instruction sets specific to a target.

  - Operations that are best expressed in some other format that can be
    compiled in to a graph (e.g. ZX).

  - Ephemeral operations used by specific compiler passes.

A nice-to-have for this extensibility is a human-friendly format for
specifying such operations.

The key difficulty with this task is well stated in the [MLIR Operation
Definition Specification
docs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#motivation)
:

> MLIR allows pluggable dialects, and dialects contain, among others, a
> list of operations. This open and extensible ecosystem leads to the
> “stringly” type IR problem, e.g., repetitive string comparisons
> during optimization and analysis passes, unintuitive accessor methods
> (e.g., generic/error prone `getOperand(3)` vs
> self-documenting `getStride()`) with more generic return types,
> verbose and generic constructors without default arguments, verbose
> textual IR dumps, and so on. Furthermore, operation verification is:
> 
> 1\. best case: a central string-to-verification-function map
> 
> 2\. middle case: duplication of verification across the code base, or
> 
> 3\. worst case: no verification functions.
> 
> The fix is to support defining ops in a table-driven manner. Then for
> each dialect, we can have a central place that contains everything you
> need to know about each op, including its constraints, custom assembly
> form, etc. This description is also used to generate helper functions
> and classes to allow building, verification, parsing, printing,
> analysis, and many more.

As we see above MLIR's solution to this is to provide a declarative
syntax which is then used to generate C++ at MLIR compile time. This is
in fact one of the core factors that ties the use of MLIR to C++ so
tightly, as managing a new dialect necessarily involves generating,
compiling, and linking C++ code.

We can do something similar in Rust, and we wouldn't even need to parse
another format, sufficiently nice rust macros/proc\_macros should
provide a human-friendly enough definition experience.

Ultimately though, we cannot avoid the "stringly" type problem if we
want *runtime* extensibility - extensions that can be specified and used
at runtime. In many cases this is desirable.

#### Extension implementation

To strike a balance then, we implement three kinds of operation/type
definition in tooling that processes the HUGR

1.  `native`: operations and types that are native to the tool, e.g. an
    Enum of quantum gates in TKET2, or of higher order operations in
    Tierkreis. Tools which do not share natives communicate over a
    serialized interface (not necessarily binary, can just be the in
    memory form of the serialized structure). At deserialization time
    when a tool sees an operation it does not recognise, it can treat it
    as opaque (likewise any wire types it does not recognise) and store
    the [serialized definition data](#serialization): in this way
    subsequent tooling which does recognise the operation will receive
    it faithfully.

2.  `CustomOp`: new operations defined in code that implement an
    extensible interface (Rust Trait), compiler operations/extensions
    that deal with these specific ops can downcast to safely retrieve
    them at runtime from the generic object (and handle the cases where
    downcasting fails). For example, an SU4 unitary struct defined in
    matrix form. This is implemented in the TKET2 prototype.

3.  `Opdef`: a struct where the operation type is identified by the name
    it holds as a string. It also implements the `CustomOp` interface.
    The struct is backed by a declarative format (e.g. YAML) for
    defining it.

Note all of these share the same representation in serialized HUGR - it
is up to the tooling as to how to load that in to memory.

We expect most compiler passes and rewrites to deal with `native`
operations, with the other classes mostly being used at the start or end
of the compilation flow. The `CustomOp` trait allows the option for
programs that extend the core toolchain to use strict typing for their
new operations. While the `Opdef` allows users to specify extensions
with a pre-compiled binary, and provide useful information for the
compiler/runtime to use.

The exact interface that should be specified by `CustomOp` is unclear,
but should include at minimum a way to query the signature of the
operation and a fallible interface for returning an equivalent program
made of operations from some provided set of `Resources`.

These classes of extension also allow greater flexibility in future. For
instance, "header" files for both `native` or `CustomOp` operation sets
can be written in the `OpDef` format for non-Rust tooling to use (e.g.
Python front end). Or like MLIR, we can in future write code generation
tooling to generate specific `CustomOp` implementations from `Opdef`
definitions.

#### Declarative format

The declarative format needs to specify some required data that is
needed by the compiler to correctly treat the operation (the minimum
case is opaque operations that should be left untouched). However, we
wish to also leave it expressive enough to specify arbitrary extra data
that may be used by compiler extensions. This suggests a flexible
standard format such as YAML would be suitable. Here we provide an
illustrative example:

See [Type System](#type-system) for more on Resources.

```yaml
# may need some top level data, e.g. namespace?

# Import other header files to use their custom types
imports: [Quantum]

# Declare custom types
types:
- name: QubitVector
  # Opaque types can take type arguments, with specified names
  args: [size]

# Declare operations which aren't associated to a resource
operations:
- name: measure
  description: "measure a qubit"
  # We're going to implement this using ops defined in the "Quantum" resource
  resource_reqs: [Quantum] 
  inputs: [[null, Q]]
  # the first element of each pair is an optional parameter name
  outputs: [[null, Q], [measured, B]]

# Declare some resource interfaces which provide the rest of the operations
resources:
- name: MyGates
  operations:
  - name: ZZPhase
    description: "Apply a parametric ZZPhase gate"
    resource_reqs: [] # The "MyGates" resource will automatically be added as a requirement
    inputs: [[null, Q], [null, Q], [angle, Angle]]
    outputs: [[null, Q], [null, Q]]
    misc:
      # extra data that may be used by some compiler passes
      equivalent: [0, 1]
      basis: [Z, Z]
  - name: SU2
    description: "One qubit unitary matrix"
    resource_reqs: []
    inputs: [[null, Q]]
    outputs: [[null, Q]]
    args:
      - matrix: List(List(List(F64))))

- name: MyResource
  operations:
  - name: MyCustom
    description: "Custom op defined by a program"
    resource_reqs: [MyGates] # Depend on operations defined in the other module
    inputs: [[null, Q], [null, Q], [param, F64]]
    outputs: [[null, Q], [null, Q]]
```

Reading this format into Rust is made easy by `serde` and
[serde\_yaml](https://github.com/dtolnay/serde-yaml) (see the
Serialization section). It is also trivial to serialize these
definitions in to the overall HUGR serialization format.

Note the required `name`, `description`. `inputs` and `outputs` fields,
the last two defining the signature of the operation, and optional
parameter names as metadata. The optional `misc` field is used for
arbitrary YAML, which is read in as-is (into the `serde_yaml Value`
struct). The data held here can be used by compiler passes which expect
to deal with this operation (e.g. a pass can use the `basis` information
to perform commutation). The optional `args` field can be used to
specify the types of parameters to the operation - for example the
matrix needed to define an SU2 operation.

### Extensible metadata

Each node in the HUGR may have arbitrary metadata attached to it. This
is preserved during graph modifications, and copied when rewriting.
Additionally the metadata may record references to other nodes; these
references are updated along with node indices.

The metadata could either be built into the hugr itself (metadata as
node weights) or separated from it (keep a separate map from node ID to
metadata). The advantages of the first approach are:

  - just one object to have around, not two;

  - reassignment of node IDs doesn't mess with metadata.

The advantages of the second approach are:

  - Metadata should make no difference to the semantics of the hugr (by
    definition, otherwise it isn't metadata but data), so it makes sense
    to be separated from the core structure.

  - We can be more agile with the details, such as formatting and
    versioning.

The problem of reassignment can be solved by having an API function that
operates on both together atomically. We will therefore tentatively
adopt the second approach, keeping metadata and hugr in separate
structures.

For each node, the metadata is a dictionary keyed by strings. Keys are
used to identify applications or users so these do not (accidentally)
interfere with each other’s metadata; for example a reverse-DNS system
(`com.quantinuum.username....` or `com.quantinuum.tket....`). The values
are tuples of (1) any serializable struct, and (2) a list of node
indices. References from the serialized struct to other nodes should
indirect through the list of node indices stored with the struct.

TODO: Specify format, constraints, and serialization. Is YAML syntax
appropriate?

There is an API to add metadata, or extend existing metadata, or read
existing metadata, given the node ID.

TODO Examples illustrating this API.

TODO Do we want to reserve any top-level metadata keys, e.g. `Name`,
`Ports` (for port metadata) or `History` (for use by the rewrite
engine)?

**TODO** Do we allow per-port metadata (using the same mechanism?)

**TODO** What about references to ports? Should we add a list of port
indices after the list of node indices?

## Type System

The type system will resemble the tierkreis type system, but with some
extensions. Namely, the things the tierkreis type system is missing are:

  - User-defined types

  - Resource management - knowing what plugins a given graph depends on

A grammar of available types is shown on the right, which extends the
list of types which exist in Tierkreis.

SimpleTypes are the types of *values* which can be sent down wires,
except for type variables `Var`. All of the ClassicTypes can also be
sent down ConstE edges.

Function signatures are made up of *rows* (\#), which consist of an
arbitrary number of SimpleTypes, plus a resource spec.

ClassicTypes `u64, i64, Float` are all fixed-size, as are QuantumTypes.
`Sum` is a disjoint union tagged by unsigned int; `Tuple`s have
statically-known number and type of elements, as does `Array<N>` (where
N is a static constant). These types are also fixed-size if their
components are.

Container types are defined in terms of statically-known element types.
Besides `Array<N>`, `Sum` and `Tuple`, these also include variable-sized
types that have been proven to work for Tierkreis: `Graph`, `Map` and
`List` (TODO: can we leave those to the Tierkreis resource?). `NewType`
allows named newtypes to be used. Containers are linear if any of their
components are linear.

```
Type ::= [Resources]SimpleType
-- Rows are ordered lists, not sets
-- If a row contains linear types, they're first
#    ::= #(LinearType), #(ClassicType) | x⃗
#(T) ::= (T)*

Resources ::= (Resource)* -- set not list

SimpleType  ::= ClassicType | LinearType
Container(T) ::= List(T)
              | Tuple(#(T))
              | Array<u64>(T)
              | Map<ClassicType, T>
              | NewType(Name, T)
              | Sum (#(T))
ClassicType ::= u64
              | i64
              | Float
              | Var(X)
              | String
              | Graph[R](#, #)
              | Opaque(Name, #)
              | Container(ClassicType)
LinearType ::= Qubit
              | QPaque(Name, #)
              | Container(SimpleType)
```

Note: any array can be turned into an equivalent tuple, but arrays also
support dynamically-indexed `get`. (TODO: Indexed by u64, with panic if
out-of-range? Or by known-range `Sum( ()^N )`?)

**Row Types** The `#` is a *row type* which consists of zero or more
simple types. Types in the row can optionally be given names in metadata
i.e. this does not affect behaviour of the HUGR. Row types are used

  - in the signatures for `Graph` inputs and outputs, and functions

  - Tuples - the 0-ary Tuple `()` aka `unit` takes no storage

  - Sums - allowing a bounded nat: `Sum((),(),())` is a ternary value;
    the 2-ary version takes the place of a boolean type

  - Arguments to `Opaque` types - where their meaning is
    extension-defined.

**Resources** The type of `Graph` has been altered to add
*R*: a resource requirement.
The *R* here refer to a set
of [resources](#resources) which are required to produce a given type.
Graphs are annotated with the resources that they need to run and, when
run, their outputs are annotated with those resources. Keeping track of
the resource requirements of graphs allows plugin designers and backends
(like tierkreis) to control how/where a module is run.

Concretely, if a plugin writer adds a resource
*X*, then some function from
a plugin needs to provide a mechanism to convert the
*X* to some other resource
requirement before it can interface with other plugins which don’t know
about *X*.

A Tierkreis runtime could be connected to workers which provide means of
running different resources. By the same mechanism, Tierkreis can reason
about where to run different parts of the graph by inspecting their
resource requirements.

### Type Constraints

We will likely also want to add a fixed set of attributes to certain
subsets of `TYPE`. In Tierkreis these are called “type constraints”. For
example, the `Map` type can only be constructed when the type that we
map from is `Hashable`. For the Hugr, we may need this `Hashable`
constraint, as well as a `Nonlinear` constraint that the typechecker can
look for before wiring up a `copy` node. Finally there may be a
`const-able` or `serializable` constraint meaning that the value can be
put into a `const`-node: this implies the type is `Nonlinear` (but not
vice versa).

**TODO**: is this set of constraints (nonlinear, const-able, hashable)
fixed? Then Map is in the core HUGR spec.

Or, can extensions (resources) add new constraints? This is probably too
complex, but then both hashable and Map could be in the Tierkreis
resource.

(Or, can we do Map without hashable?)

### Dealing with linearity

The type system will deal with linearity the same way that Tierkreis
does. It will assume everything is linear by default (since this is
implied by the implementation of edges as “links” anyway), and allow
non-linearity via a **copy** node which most types can be passed into.

This requires some magic from the typechecker to disallow copying linear
types.

Our linear types behave like other values passed down a wire. Quantum
gates behave just like other nodes on the graph with inputs and outputs,
but adding copies to the input and output wires is disallowed. In fully
qubit-counted contexts programs take in a number of qubits as input and
return the same number, with no discarding. See
[quantum resource](#quantum-resource)
for more.

### Resources

On top of the Tierkreis type system, will be a system of Resources. A
resource is a collection of operations which are available to use in
certain graphs. The operations must be callable at the point of
execution, but otherwise only signatures need to be known. Edges contain
information on the resources that were needed to produce them, and
Functions note their resource requirements in their `Graph` type. All
operations declared as part of a resource interface implicitly have the
resource they pertain to as a requirement.

Resources can be added and used by plugin writers. We will also have
some built in resources, see
[standard library](#standard-library)

Unification will demand that resource constraints are equal and, to make
it so, we will have an operations called **lift** and **liftGraph**
which can add a resource constraints to values.

![](attachments/2647818241/2647818335.png)

**lift** - Takes as a node weight parameter the single resource
**X **which it adds to the
resource requirements of it’s argument.

![](attachments/2647818241/2647818332.png)

**liftGraph** - Like **lift**, takes a
resource X as a constant node
weight parameter. Given a graph, it will add resource
X to the requirements of the
graph.

Having these as explicit nodes on the graph allows us to search for the
point before resources were added when we want to copy graphs, allowing
us to get the version with minimal resource requirements.

Graphs which are almost alike can both be squeezed into a
Conditional-node that selects one or the other, by wrapping them in a
parent graph to correct the inputs/outputs and using the **lift**
function from below.

Note that here, any letter with vector notation refers to a variable
which stands in for a row. Hence, when checking the inputs and outputs
align, we’re introducing a *row equality constraint*, rather than the
equality constraint of `typeof(b) ~ Bool`.

### Types of built-ins

We will provide some built in modules to provide basic functionality.
I’m going to define them in terms of resources. We have the “builtin”
resource which should always be available when writing hugr plugins.
This includes Conditional and TailLoop nodes, and nodes like `call`:

![](attachments/2647818241/2647818323.png)

**call** - This operation, like **to\_const**, uses it’s constE graph as
a type parameter.

On top of that, we're definitely going to want modules which handle
graph-based control flow at runtime, arithmetic and basic quantum
circuits.

These should all be defined as a part of their own resource
inferface(s). For example, we don’t assume that we can handle arithmetic
while running a circuit, so we track its use in the Graph’s type so that
we can perform rewrites which remove the arithmetic.

We would expect standard circuits to look something like

```
GraphType[Quantum](Array(5, Q), (ms: Array(5, Qubit), results: Array(5, Bit)))
```

A circuit built using our higher-order resource to manage control flow
could then look like:

```
GraphType[Quantum, HigherOrder](Array(5, Qubit), (ms: Array(5, Qubit), results: Array(5, Bit)))
```

So we’d need to perform some graph transformation pass to turn the
graph-based control flow into a CFG node that a quantum computer could
run, which removes the `HigherOrder` resource requirement:

```
precompute :: GraphType[](GraphType[Quantum,HigherOrder](Array(5, Qubit), (ms: Array(5, Qubit), results: Array(5, Bit))),
                                         GraphType[Quantum](Array(5, Qubit), (ms: Array(5, Qubit), results: Array(5, Bit))))
```

Before we can run the circuit.

## Replacement and Pattern Matching

We wish to define an API method on the HUGR that allows replacement of a
specified subgraph with a specified replacement graph.

More ambitiously, we also wish to facilitate pattern-matching on the
HUGR.

### Replacement

#### Definitions

If n is either a DFG or a CFG node, a set S of nodes in the sibling
graph under n is *convex* (DFG-convex or CFG-convex respectively) if
every node on every path in the sibling graph that starts and ends in S
is itself in S.

The meaning of “convex” is: if A and B are nodes in the convex set S,
then any sibling node on a path from A to B is also in S.

A *partial hugr* is is a graph G satisfying all the constraints of a
hugr except that:

  - it may have unconnected input ports (the set of these is denoted
    inp(G));

  - it may have unconnected output ports (the set of these is denoted
    out(G));

  - it has no root node (the set of IDs of nodes without a parent is
    denoted top(G));

  - it may have empty container nodes (the set of IDs of these is
    denoted bot(G)).

A “partial hugr” describes a set of nodes and well-formed edges between
them that potentially occupies a region of a hugr.

Given a set S of nodes in a hugr, let S\* be the set of all nodes
descended from nodes in S, including S itself.

Call two nodes a, b in Γ *separated* if a is not in {b}\* and b is not
in {a}\* (i.e. there is no hierarchy relation between them).

#### API methods

There are the following primitive operations.

##### Replacement method

###### `Replace`

This takes as input:

  - a set S of IDs of nodes in Γ, all of which are separated;

  - a partial hugr G;

  - a map T from top(G) to IDs of container nodes in Γ\\S\*;

  - a map B from bot(G) to IDs of container nodes in S\*, such that B(x)
    is separated from B(y) unless x == y. Let X be the set of children
    of values in B, and R be S\*\\X\*.

  - a bijection μ<sub>inp</sub> between inp(G) and the set of input
    ports of nodes in R whose source is not in R;

  - a bijection μ<sub>out</sub> between out(G) and the set of output
    ports of nodes in R whose target is not in R;

  - disjoint subsets Init and Term of top(G);

The new hugr is then derived by:

1.  adding the new nodes from G;

2.  connecting the ports according to the bijections μ<sub>inp</sub> and
    μ<sub>out</sub>;

3.  for each node n in top(G), adding a hierarchy edge from t(n) to n,
    placing n in the first position among children of t(n) if n is in
    Init and in the last position if n is in Term;

4.  for each node n in bot(G), and for each child m of b(n), adding a
    hierarchy edge from n to m (replacing m’s existing parent edge)

5.  removing all nodes in R

6.  If any edges inserted in step 2 are inter-graph (i.e DFG.
    non-sibling), inserting any `Order` edges required to validate them.

##### Outlining methods

###### `OutlineDFG`

Replace a DFG-convex subgraph with a single DFG node having the original
nodes as children.

###### `OutlineCFG`

Replace a CFG-convex subgraph (of sibling BasicBlock nodes) having a
single entry node with a single BasicBlock node having a CFG node child
which has as its children the original BasicBlock nodes and an exit node
that has inputs coming from all edges of the original CFG that don’t
terminate in it.

##### Inlining methods

These are the exact inverses of the above.

###### `InlineDFG`

Given a DFG node in a DSG, inline its children into the DSG.

###### `InlineCFG`

When a BasicBlock node has a single CFG node as a child, replace it with
the children of that CFG node.

##### Identity insertion and removal methods

###### `InsertIdentity`

Given an edge between sibling nodes in a DSG, insert an `identity<T>`
node having its source as predecessor and its target as successor.

###### `RemoveIdentity`

Remove an `identity<T>` node from a DSG, wiring its predecessor to its
successor.

##### Order insertion and removal methods

###### `InsertOrder`

Insert an Order edge from `n0` to `n1` where `n0` and `n1` are distinct
siblings in a DSG such that there is no path in the DSG from `n1` to
`n0`. If there is already an order edge from `n0` to `n1` this does
nothing (but is not an error).

###### `RemoveOrder`

Given nodes `n0` and `n1`, if there is an Order edge from `n0` to `n1`,
remove it. (If there is an intergraph edge from `n0` to a descendent of
`n1`, this invalidates the hugr. TODO should this be an error?)

##### Insertion and removal of const loads

###### `InsertConstIgnore`

Given a `ConstN<T>` node `c`, and optionally a DSG `P`, add a new
`load_constant<T>` node `n` as a child of `P` with a `ConstE<T>` edge
from `c` to `n` and no outgoing edges from `n`. Also add an Order edge
from the Input node under `P` to `n`. Return the ID of `n`. If `P` is
omitted it defaults to the parent of `c` (in this case said `c` will
have to be in a DSG or CSG rather than under the Module Root.) If `P` is
provided, it must be a descendent of the parent of `c`.

###### `RemoveConstIgnore`

Given a `load_constant<T>` node `n` that has no outgoing edges, remove
it (and its incoming value and Order edges) from the hugr.

##### Insertion and removal of const nodes

###### `InsertConst`

Given a `constN<T>` node `c` and a DSG `P`, add `c` as a child of `P`,
inserting an Order edge from the Input under `P` to `c`.

###### `RemoveConst`

Given a `constN<T>` node `c` having no outgoing edges, remove `c`
together with its incoming `Order` edge.

#### Usage

Note that we can only reattach children into empty replacement
containers. This simplifies the API, and is not a serious restriction
since we can use the outlining and inlining methods to target a group of
nodes.

The most basic case – replacing a convex set of Op nodes in a DSG with
another graph of Op nodes having the same signature – is implemented by
having T map everything to the parent node, and bot(G) is empty.

If one of the nodes in the region is a complex container node that we
wish to preserve in the replacement without doing a deep copy, we can
use an empty node in the replacement and have B map this node to the old
one.

We can, for example, implement “turning a Conditional-node with known
predicate into a DFG-node” by a `Replace` where the Conditional (and its
preceding predicate) is replaced by an empty DFG and the map B specifies
the “good” child of the Conditional as the surrogate parent of the new
DFG’s children. (If the good child was just an Op, we could either
remove it and include it in the replacement, or – to avoid this overhead
– outline it in a DFG first.)

Similarly, replacement of a CFG node having a single BasicBlock child
with a DFG node can be achieved using `Replace` (specifying the
BasicBlock node as the surrogate parent for the new DFG’s children).

Arbitrary node insertion on dataflow edges can be achieved using
`InsertIdentity` followed by `Replace`. Removal of a node in a DSG
having input wires and output wires of the same type can be achieved
using `Replace` (with a set of `identity<T>` nodes) followed by
`RemoveIdentity`.

### Normalisation

We envisage that some kind of pass can be used after a rewrite or series
of rewrites to automatically apply RemoveConstIgnore for any unused
load\_constants, merging copies (and discards of copies), and other such
tidies. This might be global, or by tracking which parts of the Hugr
have been touched.

### Metadata updates on replacement

When requesting a replacement on Γ the caller may optionally provide
metadata for the nodes of Γ and Γ'. Upon replacement, the metadata for
the nodes in Γ are updated with the metadata for the nodes of Γ' that
replace them. (When child nodes are rewired, they keep their existing
metadata.)

The fate of the metadata of nodes in S depends on the policy specified,
as described below.

The caller may also specify a [basic regular
expression](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX_Basic_Regular_Expressions)
(or some other textual pattern format TBD) specifying which keys of
metadata to transfer (e.g. `Foo`, or `*` for all metadata, or `Debug_*`
for all metadata keyed by a string beginning with `Debug_`).

If no policy is specified, the default is to forget all metadata
attached to the replaced subgraph (except for rewired child nodes).

Other policies could include:

  - to each of the new nodes added, insert a piece of metadata in its
    `History` section that captures all the chosen metadata with the
    keys of the replaced nodes:
    
      - ` History: {Replaced: [{node1: old_node1_metadata, node2:
        old_node2_metadata, ...}, {...}, ...]}  `where `Replaced`
        specifies an ordered list of replacements, and the new
        replacement appends to the list (or creates a new list if
        `Replaced` doesn't yet exist);

  - to the root (module) node of Γ, attach metadata capturing a
    serialization of the replacement (both the set of nodes replaced and
    its replacement):
    
      - `History: {Replacements: [...]}`

Further policies may be defined in the future; none of these polices
(except for the default forgetful policy) form part of this
specification.

### Pattern matching

We would like to be able to find all subgraphs of a HUGR matching a
given pattern. Exactly how the pattern is specified, and the details of
the algorithm, are not discussed here; but we assume that we have an
implementation of such an algorithm that works on flat
(non-hierarchical) port-graphs.

It can be applied separately to each DSG within the HUGR, matching the
various node types within it. Starting from the root module, we can
recurse down to other DSGs within the HUGR.

It should also be possible to specify a particular DSG on which to run
the pattern matching, by supplying its parent node ID.

Patterns matching edges that traverse DSGs are also possible, but will
be implemented in terms of the above replacement operations, making use
of the child-rewiring lists.

## Serialization

### Goals

  - Fast serialization/deserialization in Rust.

  - Ability to generate and consume from Python.

  - Reasonably small sized files/payloads.

  - Ability to send over wire. Myqos will need to do things like:
    
      - Store the program in a database
    
      - Search the program(?) (Increasingly
        unlikely with larger more complicated programs)
    
      - Validate the data
    
      - **Most important:** version the data for compiler/runtime
        compatibility

### Non-goals

Human-programmability: LLVM for example has exact correspondence between
it's bitcode, in memory and human readable forms. This is quite handy
for developers to inspect and modify the human readable form directly.
Unfortunately this then requires a grammar and parsing/codegen, which is
maintenance and design overhead. We believe that for most cases,
inspecting and modifying the in-memory structure will be enough. If not,
in future we can add a human language and a standalone module for
conversion to/from the binary serialised form.

### Schema

We propose the following simple serialized structure, expressed here in
pseudocode, though we advocate MessagePack format in practice (see
[Serialization implementation](serialization.md)).
Note in particular that node and port weights are stored as separate
maps to the graph structure itself, and that hierarchical relationships
have a special encoding outside `edges`, as an optional parent field
(the first) in a node definition. `Operation` refers to serialized
payloads corresponding to arbitrary `Operations`. Metadata could also be
included as a similar map.

```rust
struct HUGR {
  nodes: [Node]
  edges: [Edge]
  node_weights: map<Int, Operation>
}

// (parent, #incoming, #outgoing)
struct Node = (Optional<Int>, Int, Int)
// ((source, offset), (target, offset)
struct Edge = ((Node, Int), (Node, Int))
```

Node and edge indices, used as keys in the weight maps and within the
definitions of nodes and indices, directly correspond to indices of the
node/edge lists. An edge is defined by the source and target nodes, and
the offset of the output/input ports within those nodes. This scheme
enforces that nodes are contiguous - a node index must always point to a
valid node - whereas in tooling implementations it may be necessary to
implement stable indexing where removing a node invalidates that index
while keeping all other indices pointing to the same node.

### Architecture

The HUGR is implemented as a Rust crate named `quantinuum-hugr`. This
crate is intended to be a common dependency for all projects, and is to
be published on the <http://crates.io> registry.

The HUGR is represented internally using structures from the `portgraph`
crate. A base PortGraph is composed with hierarchy (as an alternate
implementation of `Hierarchy` relationships) and weight components. The
implementation of this design document is available on GitHub.

<https://github.com/CQCL-DEV/hugr>

## Standard Library

`panic`: panics unconditionally; no inputs, any type of outputs (these
are never produced)

### Arithmetic Resource

The Arithmetic Resource provides types and operations for integer and
floating-point operations.

We largely adopt (a subset of) the definitions of
[WebAssembly 2.0](https://webassembly.github.io/spec/core/index.html),
including the names of the operations. Where WebAssembly specifies a
"partial" operation (i.e. when the result is not defined on certain
inputs), we use a Sum type to hold the result.

A few additonal operations not included in WebAssembly are also
specified, and there are some other small differences (highlighted
below).

The `int<N>` type is parametrized by its width `N`, which is a positive
integer.

The possible values of `N` are at least 1, 32 and 64. We could trivially
extend this list. (TODO decide.)

The `int<N>` type represents an arbitrary bit string of length `N`.
Semantics are defined by the operations. There are three possible
interpretations of a value:

  - as a bit string $(a_{N-1}, a_{N-2}, \ldots, a_0)$ where $a_i
    \in {0,1}$;

  - as an unsigned integer $\sum_{i<N}i 2^i a_i$;

  - as a signed integer $\sum_{i<N-1} 2^i a_i - 2^{N-1} a_{N-1}$.

An asterix ( \* ) in the tables below indicates that the definition
either differs from or is not part of the
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html)
specification.

Const nodes:

| Name                   | Inputs | Outputs  | Meaning                                                               |
| ---------------------- | ------ | -------- | --------------------------------------------------------------------- |
| `iconst_u<N, x>`( \* ) | none   | `int<N>` | const node producing unsigned value x (where 0 \<= x \< 2^N)          |
| `iconst_s<N, x>`( \* ) | none   | `int<N>` | const node producing signed value x (where -2^(N-1) \<= x \< 2^(N-1)) |

Casts:

| Name                   | Inputs   | Outputs                  | Meaning                                                                                      |
| ---------------------- | -------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| `iwiden_u<M,N>`( \* )  | `int<M>` | `int<N>`                 | widen an unsigned integer to a wider one with the same value (where M \<= N)                 |
| `iwiden_s<M,N>`( \* )  | `int<M>` | `int<N>`                 | widen a signed integer to a wider one with the same value (where M \<= N)                    |
| `inarrow_u<M,N>`( \* ) | `int<M>` | `Sum(int<N>, ErrorType)` | narrow an unsigned integer to a narrower one with the same value if possible (where M \>= N) |
| `inarrow_s<M,N>`( \* ) | `int<M>` | `Sum(int<N>, ErrorType)` | narrow a signed integer to a narrower one with the same value if possible (where M \>= N)    |
| `itobool` ( \* )       | `int<1>` | `bool`                   | convert to `bool` (1 is true, 0 is false)                                                    |
| `ifrombool` ( \* )     | `bool`   | `int<1>`                 | convert from `bool` (1 is true, 0 is false)                                                  |

Comparisons:

| Name       | Inputs             | Outputs | Meaning                                      |
| ---------- | ------------------ | ------- | -------------------------------------------- |
| `ieq<N>`   | `int<N>`, `int<N>` | `bool`  | equality test                                |
| `ine<N>`   | `int<N>`, `int<N>` | `bool`  | inequality test                              |
| `ilt_u<N>` | `int<N>`, `int<N>` | `bool`  | "less than" as unsigned integers             |
| `ilt_s<N>` | `int<N>`, `int<N>` | `bool`  | "less than" as signed integers               |
| `igt_u<N>` | `int<N>`, `int<N>` | `bool`  | "greater than" as unsigned integers          |
| `igt_s<N>` | `int<N>`, `int<N>` | `bool`  | "greater than" as signed integers            |
| `ile_u<N>` | `int<N>`, `int<N>` | `bool`  | "less than or equal" as unsigned integers    |
| `ile_s<N>` | `int<N>`, `int<N>` | `bool`  | "less than or equal" as signed integers      |
| `ige_u<N>` | `int<N>`, `int<N>` | `bool`  | "greater than or equal" as unsigned integers |
| `ige_s<N>` | `int<N>`, `int<N>` | `bool`  | "greater than or equal" as signed integers   |

Other operations:

| Name                   | Inputs             | Outputs                            | Meaning                                                                                                                                                  |
| ---------------------- | ------------------ | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `imax_u<N>`            | `int<N>`, `int<N>` | `int<N>`                           | maximum of unsigned integers                                                                                                                             |
| `imax_s<N>`            | `int<N>`, `int<N>` | `int<N>`                           | maximum of signed integers                                                                                                                               |
| `imin_u<N>`            | `int<N>`, `int<N>` | `int<N>`                           | minimum of unsigned integers                                                                                                                             |
| `imin_s<N>`            | `int<N>`, `int<N>` | `int<N>`                           | minimum of signed integers                                                                                                                               |
| `iadd<N>`              | `int<N>`, `int<N>` | `int<N>`                           | addition modulo 2^N (signed and unsigned versions are the same op)                                                                                       |
| `isub<N>`              | `int<N>`, `int<N>` | `int<N>`                           | subtraction modulo 2^N (signed and unsigned versions are the same op)                                                                                    |
| `ineg<N>`              | `int<M>`           | `int<N>`                           | negation modulo 2^N (signed and unsigned versions are the same op)                                                                                       |
| `imul<N>`              | `int<N>`, `int<N>` | `int<N>`                           | multiplication modulo 2^N (signed and unsigned versions are the same op)                                                                                 |
| `idivmod_u<N,M>`( \* ) | `int<N>`, `int<M>` | `Sum((int<N>, int<M>), ErrorType)` | given unsigned integers 0 \<= n \< 2^N, 0 \<= m \< 2^M, generates unsigned q, r where q\*m+r=n, 0\<=r\<m (m=0 is an error)                               |
| `idivmod_s<N,M>`( \* ) | `int<N>`, `int<M>` | `Sum((int<N>, int<M>), ErrorType)` | given signed integer -2^{N-1} \<= n \< 2^{N-1} and unsigned 0 \<= m \< 2^M, generates signed q and unsigned r where q\*m+r=n, 0\<=r\<m (m=0 is an error) |
| `idiv_u<N,M>`          | `int<N>`, `int<M>` | `Sum(int<N>, ErrorType)`           | as `idivmod_u` but discarding the second output                                                                                                          |
| `imod_u<N,M>`          | `int<N>`, `int<M>` | `Sum(int<M>, ErrorType)`           | as `idivmod_u` but discarding the first output                                                                                                           |
| `idiv_s<N,M>`( \* )    | `int<N>`, `int<M>` | `Sum(int<N>, ErrorType)`           | as `idivmod_s` but discarding the second output                                                                                                          |
| `imod_s<N,M>`( \* )    | `int<N>`, `int<M>` | `Sum(int<M>, ErrorType)`           | as `idivmod_s` but discarding the first output                                                                                                           |
| `iabs<N>`              | `int<N>`           | `int<N>`                           | convert signed to unsigned by taking absolute value                                                                                                      |
| `iand<N>`              | `int<N>`, `int<N>` | `int<N>`                           | bitwise AND                                                                                                                                              |
| `ior<N>`               | `int<N>`, `int<N>` | `int<N>`                           | bitwise OR                                                                                                                                               |
| `ixor<N>`              | `int<N>`, `int<N>` | `int<N>`                           | bitwise XOR                                                                                                                                              |
| `inot<N>`              | `int<N>`           | `int<N>`                           | bitwise NOT                                                                                                                                              |
| `ishl<N,M>`( \* )      | `int<N>`, `int<M>` | `int<N>`                           | shift first input left by k bits where k is unsigned interpretation of second input (leftmost bits dropped, rightmost bits set to zero)                  |
| `ishr<N,M>`( \* )      | `int<N>`, `int<M>` | `int<N>`                           | shift first input right by k bits where k is unsigned interpretation of second input (rightmost bits dropped, leftmost bits set to zero)                 |
| `irotl<N,M>`( \* )     | `int<N>`, `int<M>` | `int<N>`                           | rotate first input left by k bits where k is unsigned interpretation of second input (leftmost bits replace rightmost bits)                              |
| `irotr<N,M>`( \* )     | `int<N>`, `int<M>` | `int<N>`                           | rotate first input right by k bits where k is unsigned interpretation of second input (rightmost bits replace leftmost bits)                             |

The `float64` type represents IEEE 754-2019 floating-point data of 64
bits.

Floating-point operations are defined as follows. All operations below
follow
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html#floating-point-operations)
except where stated.

| Name              | Inputs               | Outputs   | Meaning                                                                  |
| ----------------- | -------------------- | --------- | ------------------------------------------------------------------------ |
| `fconst<x>`( \* ) | none                 | `float64` | const node producing a float                                             |
| `feq`( \* )       | `float64`, `float64` | `bool`    | equality test (as WASM but with 0 and 1 interpreted as `bool`)           |
| `fne`( \* )       | `float64`, `float64` | `bool`    | inequality test (as WASM but with 0 and 1 interpreted as `bool`)         |
| `flt`( \* )       | `float64`, `float64` | `bool`    | "less than" (as WASM but with 0 and 1 interpreted as `bool`)             |
| `fgt`( \* )       | `float64`, `float64` | `bool`    | "greater than" (as WASM but with 0 and 1 interpreted as `bool`)          |
| `fle`( \* )       | `float64`, `float64` | `bool`    | "less than or equal" (as WASM but with 0 and 1 interpreted as `bool`)    |
| `fge`( \* )       | `float64`, `float64` | `bool`    | "greater than or equal" (as WASM but with 0 and 1 interpreted as `bool`) |
| `fmax`            | `float64`, `float64` | `float64` | maximum                                                                  |
| `fmin`            | `float64`, `float64` | `float64` | minimum                                                                  |
| `fadd`            | `float64`, `float64` | `float64` | addition                                                                 |
| `fsub`            | `float64`, `float64` | `float64` | subtraction                                                              |
| `fneg`            | `float64`            | `float64` | negation                                                                 |
| `fabs`            | `float64`            | `float64` | absolute value                                                           |
| `fmul`            | `float64`, `float64` | `float64` | multiplication                                                           |
| `fdiv`            | `float64`, `float64` | `float64` | division                                                                 |
| `ffloor`          | `float64`            | `float64` | floor                                                                    |
| `fceil`           | `float64`            | `float64` | ceiling                                                                  |

Conversions between integers and floats:

| Name           | Inputs    | Outputs                  | Meaning               |
| -------------- | --------- | ------------------------ | --------------------- |
| `trunc_u<N>`   | `float64` | `Sum(int<N>, ErrorType)` | float to unsigned int |
| `trunc_s<N>`   | `float64` | `Sum(int<N>, ErrorType)` | float to signed int   |
| `convert_u<N>` | `int<N>`  | `float64`                | unsigned int to float |
| `convert_s<N>` | `int<N>`  | `float64`                | signed int to float   |

### Quantum Resource

This is the resource that is designed to be natively understood by
TKET2. Besides a range of quantum operations (like Hadamard, CX, etc.)
that take and return `Qubit`, we note the following operations for
allocating/deallocating `Qubit`s:

```
qalloc: () -> Qubit
qfree: Qubit -> ()
```

`qalloc` allocates a fresh, 0 state Qubit - if none is available at
runtime it panics. `qfree` loses a handle to a Qubit (may be reallocated
in future). The point at which an allocated qubit is reset may be
target/compiler specific.

Note there are also `measurez: Qubit -> (i1, Qubit)` and on supported
targets `reset: Qubit -> Qubit` operations to measure or reset a qubit
without losing a handle to it.

**Dynamic vs static allocation**

With these operations the programmer/front-end can request dynamic qubit
allocation, and the compiler can add/remove/move these operations to use
more or fewer qubits. In some use cases, that may not be desirable, and
we may instead want to guarantee only a certain number of qubits are
used by the program. For this purpose TKET2 places additional
constraints on the `main` function that are in line with TKET1 backwards
compatibility. Namely the main function takes one `Array<N, Qubit>`
input and has one output of the same type (the same statically known
size). If further the program does not contain any `qalloc` or `qfree`
operations we can state the program only uses `N` qubits.

### Higher-order (Tierkreis) Resource

In **some** contexts, notably the Tierkreis runtime, higher-order
operations allow graphs to be valid dataflow values, and be executed.
These operations allow this.

  - `call_indirect`: Call a function indirectly. Like `call`, but the
    first input is a standard dataflow graph type. This is essentially
    `eval` in Tierkreis.

  - `catch`: like `call_indirect`, the first argument is of type
    `Graph[R]<I,O>` and the rest of the arguments are of type `I`.
    However the result is not `O` but `Sum(O,ErrorType)`

  - `parallel`, `sequence`, `partial`? Note that these could be executed
    in first order graphs as straightforward (albeit expensive)
    manipulations of Graph `struct`s/protobufs\!

![](attachments/2647818241/2647818326.png)

**loop** - In order to run the *body* graph, we need the resources
R that the graph requires, so
calling the **loop** function requires those same resources. Since the
result of the body is fed into the input of the graph, it needs to have
the same resource requirements on its inputs and outputs. We require
that *v* is lifted to have resource requirement
R so that it matches the type
of input to the next iterations of the loop.

![](attachments/2647818241/2647818329.png)

**call\_indirect** - This has the same feature as **loop**: running a
graph requires it’s resources.

![](attachments/2647818241/2647818368.png)

**to\_const** - For operations which instantiate a graph (**to\_const**
and **call**) the functions are given an extra parameter at graph
construction time which corresponds to the graph type that they are
meant to instantiate. This type will be given by a typeless edge from
the graph in question to the operation, with the graph’s type added as
an edge weight.

## Glossary

  - **BasicBlock node**: A child of a CFG node (i.e. a basic block
    within a control-flow graph).

  - **call node**: TODO

  - **child node**: A child of a node is an adjacent node in the
    hierarchy that is further from the root node; equivalently, the
    target of a hierarchy edge from the current (parent) node.

  - **const edge**: TODO

  - **const node**: TODO

  - **container node**: TODO

  - **control-flow edge**: An edge between BasicBlock nodes in the same
    CFG (i.e. having the same parent CFG node).

  - **control-flow graph (CFG)**: The set of all children of a given CFG
    node, with all the edges between them. Includes exactly one entry
    and one exit node. Nodes are basic blocks, edges point to possible
    successors.

  - **copy node**: TODO

  - **Dataflow Sibling Graph (DSG)**: The set of all children of a given
    Dataflow container node, with all edges between them. Includes
    exactly one input node (unique node having no input edges) and one
    output node (unique node having no output edges). Nodes are
    processes that operate on input data and produce output data. Edges
    in a DSG are either value or order edges. The DSG must be acyclic.

  - **data-dependency node**: an input, output, operation, DFG, CFG,
    Conditional or TailLoop node. All incoming and outgoing edges are
    value edges.

  - **declare node**: child of a module, indicates that an external
    function exists but without giving a definition. May be the source
    of constE-edges to call nodes and others.

  - **def node**: child of a module node, defines a function (by being
    parent to the function’s body). May be the source of constE-edges to
    call nodes and others.

  - **DFG node**: A node representing a data-flow graph. Its children
    are all data-dependency nodes.

  - **edge kind**: There are five kinds of edge: value edge, order edge,
    control-flow edge, constE edge, and hierarchy edge.

  - **edge type:** Typing information attached to a value edge or constE
    edge (representing the data type of value that the edge carries).

  - **entry node**: The distinguished node of a CFG representing the
    point where execution begins.

  - **exit node**: The distinguished node of a CFG representing the
    point where execution ends.

  - **function:** TODO

  - **Conditional node:** TODO

  - **hierarchy**: A rooted tree whose nodes are all nodes of the HUGR,
    rooted at the module node.

  - **hierarchy edge**: An edge in the hierarchy tree. The edge is
    considered to be directed, with the source node the parent of the
    target node.

  - **input node**: The distinguished node of a DSG representing the
    point where data processing begins.

  - **input signature**: The input signature of a node is the mapping
    from identifiers of input ports to their associated edge types.

  - **inter-graph edge**: TODO

  - **CFG node**: A node representing a control-flow graph. Its children
    are all BasicBlock nodes, of which there is exactly one entry node
    and exactly one exit node.

  - **load-constant node**: TODO

  - **metadata:** TODO

  - **module**: TODO

  - **node index**: An identifier for a node that is unique within the
    HUGR.

  - **operation**: TODO

  - **output node**: The distinguished node of a DSG representing the
    point where data processing ends.

  - **output signature**: The output signature of a node is the mapping
    from identifiers of output ports to their associated edge types.

  - **parent node**: The parent of a non-root node is the adjacent node
    in the hierarchy that is nearer to the root node.

  - **port**: A notional entry or exit point from a data-dependency
    node, which has an identifier that is unique for the given node.
    Each incoming or outgoing value edge is associated with a specific
    port.

  - **port index**: An identifier for a port that is unique within the
    HUGR.

  - **replacement**: TODO

  - **resource**: TODO

  - **sibling graph**: TODO

  - **signature**: The signature of a node is the combination of its
    input and output signatures.

  - **simple type**: a quantum or classical type annotated with the
    Resources required to produce the value

  - **order edge**: An edge implying dependency of the target node on
    the source node.

  - **TailLoop node**: TODO

  - **value edge:** An edge between data-dependency nodes. Has a fixed
    edge type.

## Appendix: Rationale for Control Flow

### **Justification of the need for CFG-nodes**

  - Conditional + TailLoop are not able to express arbitrary control
    flow without introduction of extra variables (dynamic overhead, i.e.
    runtime cost) and/or code duplication (static overhead, i.e. code
    size).
    
      - Specifically, the most common case is *shortcircuit evaluation*:
        `if (P && Q) then A; else B;` where Q is only evaluated if P is
        true.

  - We *could* parse a CFG into a DSG with only Conditional-nodes and
    TailLoop-nodes by introducing extra variables, as per [Google
    paper](https://dl.acm.org/doi/pdf/10.1145/2693261), and then expect
    LLVM to remove those extra variables later. However that’s a
    significant amount of analysis and transformation, which is
    undesirable for using the HUGR as a common interchange format (e.g.
    QIR → HUGR → LLVM) when little optimization is being done (perhaps
    no cross-basic-block optimization).

  - It’s possible that maintaining support for CFGs-nodes may become a
    burden, i.e. if we find we are not using CFGs much. However, I
    believe that this burden can be kept acceptably low if we are
    willing to drop support for rewriting across basic block boundaries,
    which would be fine if we find we are not using CFGs much (e.g.
    either we rely on turning relevant CFG/fragments into
    Conditional/TailLoop-nodes first, which might constitute rewriting
    in itself; or programmers are mainly using (our) front-end tools
    that build Conditional/TailLoop-nodes directly.)

…and the converse: we want `Conditional` and `TailLoop` *as well* as
`CFG` because we believe they are much easier to work with conceptually
e.g. for authors of "rewrite rules" and other optimisations.

### **Alternative representations considered but rejected**

  - A [Google paper](https://dl.acm.org/doi/pdf/10.1145/2693261) allows
    for the introduction of extra variables into the DSG that can be
    eliminated at compile-time (ensuring no runtime cost), but only if
    stringent well-formedness conditions are maintained on the DSG, and
    there are issues with variable liveness.

  - [Lawrence's
    thesis](https://www.cl.cam.ac.uk/techreports/UCAM-CL-TR-705.pdf)
    handles some cases (e.g. shortcircuit evaluation) but cannot handle
    arbitrary (even reducible) control flow and the multi-stage approach
    makes it hard to see what amount of code duplication will be
    necessary to turn the IR back into a CFG (e.g. following a rewrite).

  - We could extend Conditional/TailLoop nodes to be more expressive
    (i.e. allowing them or others to capture *more* common cases, indeed
    arbitrary reducible code, in a DSG-like fashion), perhaps something
    like WASM. However although this would allow removing the CFG, the
    DSG nodes get more complicated, and start to behave in very
    non-DSG-like ways.
<!--
      - In the limit, we have TailLoop node for loops, plus a node that
        contains an arbitrary *acyclic* CFG\! That was [considered
        here](#) but still requires extra variables and runs into
        similar problems with liveness as the Google paper. Also [The
        fully-expressive alternative to
        θ-nodes](https://cqc.atlassian.net/wiki/spaces/TKET/pages/2623406136).
-->
  - We could use function calls to avoid code duplication (essentially
    the return address is the extra boolean variable, likely to be very
    cheap). However, I think this means pattern-matching will want to
    span across function-call boundaries; and it rules out using
    inter-graph edges for called functions. TODO are those objections
    sufficient to rule this out?

#### Comparison with MLIR

There are a lot of broad similarities here, with MLIR’s regions
providing hierarchy, and “graph” regions being like DSGs. Significant
differences include:

  - MLIR uses names everywhere, which internally are mapped to some kind
    of hyperedge; we have explicit edges in the structure (and copy
    nodes rather than hyperedges).
    
      - However, we can think of every output nodeport being a unique
        SSA/SSI name.
    
      - MLIR does not do linearity or SSI.

  - Our CFGs are Single Entry Single Exit (results defined by the output
    node of the exit block), rather than MLIR’s Single Entry Multiple
    Exit (with `return` instruction)

  - MLIR allows multiple regions inside a single operation, whereas we
    introduce extra levels of hierarchy to allow this.

  - I note re. closures that MLIR expects the enclosing scope to make
    sure any referenced values are kept ‘live’ for long enough. Not what
    we do in Tierkreis (the closure-maker copies them)\!
