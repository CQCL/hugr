# HUGR design document

The Hierarchical Unified Graph Representation (HUGR, pronounced *hugger*
🫂) is a proposed new
common internal representation used across TKET2, Tierkreis, and the L3
compiler. The HUGR project aims to give a faithful representation of
operations, that facilitates compilation and encodes complete programs,
with subprograms that may execute on different (quantum and classical)
targets.

![](/hugr/assets/hugr_logo.svg)


## Motivation

Multiple compilers and tools in the Quantinuum stack use some graph-like
program representation; be it the quantum circuits encoded as DAGs in
TKET, or the higher-order executable dataflow graphs in Tierkreis.

The goal of the HUGR representation is to provide a unified structure
that can be shared between the tools, allowing for more complex
operations such as TKET optimizations across control-flow blocks, and
nested quantum and classical programs in a single graph.
The HUGR should provide a generic graph representation of a program,
where each node contains a specific kind of operation and edges
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
  be implemented separately.
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
  HUGR. These include the [guppylang](https://github.com/quantinuum/guppylang)
  Python eDSL for quantum-classical programming,
  and BRAT (which already uses an internal graph-like
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

## Functional description

A HUGR is a directed graph. There are several different types of node, and
several different types of edge, with different semantics, described below.

A node usually has additional data associated with it, which we will
refer to as its *node weight*.

The nodes represent
processes that produce values - either statically, i.e. at compile time,
or at runtime. Each node is uniquely identified by its **node index**,
although this may not be stable under graph structure modifications.
Each node is defined by its **operation**; the possible operations are
outlined in [Node Operations](#node-operations)
but may be [extended by Extensions](#extension-system).

### Simple HUGR example

```mermaid
graph  LR
    Input -->|0:0| H
    H -->|0:0| CNOT
    Input -->|1:1| CNOT
    CNOT -->|0:0| Output
    CNOT -->|1:1| Output
```

In the example above, a 2-qubit circuit is described as a dataflow
region of a HUGR with one `H` operation and one `CNOT` operation. The
operations have an incoming and outgoing list of ports, with each
element identified by its offset and labelled with a type.
In the diagram the edge label includes the source and target port indices as `<source>:<target>`.

The signature of the `CNOT` operation is `[Qubit, Qubit] → [Qubit,
Qubit]`. Further information in the metadata may label the first qubit
as *control* and the second as *target*.

In this case, output 0 of the H operation is connected to input 0 of the
CNOT.

### Edges, ports and signatures

The edges of a HUGR encode relationships between nodes; there are several *kinds*
of edge for different relationships. Edges of a given kind are specified to
carry an edge weight:

- `Order` edges are plain directed edges, and express requirements on the
  ordering. They have no edge weight.
- `Value` edges carry typed data at runtime. They have a *port* at each end, associated
  with the source and target nodes. They have an `AnyType`as an edge weight.
- `Const` edges are similar to `Value` edges but carry static data (knowable at
  compilation time). These have as edge weight a `CopyableType`.
- `Function` edges refer to a statically-known function, but with a type scheme
  that (unlike values) may be polymorphic---see [Polymorphism](#polymorphism).
- `ControlFlow` edges represent possible flows of control from one part of the
  program to another. They have no edge weight.
- `Hierarchy` edges express the relationship between container nodes and their
  children. They have no edge weight.

It is useful to introduce some terms for broader classes of edge:
* *Static* edges are the union of the `Const` and `Function` edges
* *Dataflow* edges are the union of `Value` and Static (thus, `Value`, `Const` and `Function`)

A `Value` edge can carry data of any `AnyType`: these include the `CopyableType`s
(which can be freely copied or discarded - i.e. ordinary classical data)
as well as anything which cannot - e.g. quantum data.
A `Const` edge can only carry a `CopyableType`. For
more details see the [Type System](#type-system) section.

As well as the type, Dataflow edges are also parametrized by a
`Locality`, which declares whether the edge crosses levels in the hierarchy. See
[Edge Locality](#edge-locality) for details.

```haskell
AnyType ⊃ CopyableType

EdgeKind ::= Value(Locality, AnyType)
             | Const(Local | Ext, CopyableType) | Function(Local | Ext, PolyFuncType)
             | Hierarchy | Order | ControlFlow
```

Note that a port is associated with a node and zero or more Dataflow edges.
Incoming ports are associated with exactly one edge, or many `ControlFlow` edges.
All Dataflow edges associated with a port have the same type; thus a port has a
well defined type, matching that of its adjoining edges. The incoming and
outgoing ports of a node are each ordered independently, meaning that the first
output port will be "0" regardless of how many input ports there are.

The sequences of incoming and outgoing port types (carried on `Value` edges) of a node constitute its
*signature*.

Note that the locality is not fixed or even specified by the signature.

A source port with a `CopyableType` may have any number of edges associated with
it (including zero, which means "discard"). Any other port
must have exactly one edge associated with it. This captures the property of
linear types that the value is used exactly once.

The `Hierarchy` and `ControlFlow` edges from a node
are ordered (the children of a container node have a linear ordering, as do the
successors of a `BasicBlock` node).

#### `Hierarchy` edges

A `Hierarchy` edge from node *a* to *b* encodes that *a* is the direct parent
of *b*. Only certain nodes, known as *container* nodes, may act as parents -
these are listed in
[hierarchical node relationships](#hierarchical-relationships-and-constraints).
In a valid HUGR the hierarchy edges form a tree joining all nodes of the HUGR,
with a unique root node. The HUGR is characterized by the type of its root node.
The root node has no non-hierarchy edges (and this supersedes any other requirements on the
edges of specific node types).

A *sibling graph* is a subgraph of the HUGR containing all nodes with
a particular parent, plus any `Order`, `Value` `Static`, and `ControlFlow` edges between
them.

#### `Value` edges

A `Value` edge represents dataflow that happens at runtime - i.e. the
source of the edge will, at runtime, produce a value that is consumed by
the edge's target. Value edges are from an outgoing port of the
source node, to an incoming port of the target node.

#### Static edges (`Const` and `Function`)

A Static edge represents dataflow that is statically knowable - i.e.  the source
is a compile-time constant defined in the program. Hence, the types on these
edges are classical. Only a few nodes may be sources (`FuncDefn`, `FuncDecl` and
`Const`) and targets (`Call`, `LoadConstant`, and `LoadFunction`) of these
edges; see [operations](#node-operations).

#### `Order` edges

`Order` edges represent explicit constraints on ordering between nodes
(e.g. useful for stateful operations). These can be seen as
local value edges of unit type `()`, i.e. that pass no data, and where
the source and target nodes must have the same parent. There can be at
most one `Order` edge between any two nodes.

#### `ControlFlow` edges

`ControlFlow` edges represent all possible flows of control
from one region (basic block) of the program to another. These are
always local, i.e. source and target have the same parent.

### Node Operations

Here we define some core types of operation required to represent
full programs, including [dataflow operations](#dataflow).

#### Module

If the HUGR contains a `Module` node then it is unique and sits at the top level
of the hierarchy. In this case we call it a **module HUGR**. The weight
attached to this node contains module level data. There may also be additional
metadata (e.g. source file). The children of a `Module` correspond
to "module level" operation types. Neither `Module` nor these module-level
operations have value ports, but some have Static or other
edges. The following operations are *only* valid as immediate children of a
`Module` node.

- `FuncDecl`: an external function declaration. The name of the function,
  a list of type parameters (TypeParams, see [Type System](#type-system))
  and function attributes (relevant for compilation)
  define the node weight. The node has an outgoing `Function`
  edge for each use of the function. The function name is used at link time to
  look up definitions in linked
  modules (other hugr instances specified to the linker).
- `AliasDecl`: an external type alias declaration. At link time this can be
  replaced with the definition. An alias declared with `AliasDecl` is equivalent to a
  named opaque type.
- `FuncDefn` : a function definition. Like `FuncDecl` but with a function body.
  The function body is defined by the sibling graph formed by its children.
  At link time `FuncDecl` nodes are replaced by `FuncDefn`.
- `AliasDefn`: type alias definition. At link time `AliasDecl` can be replaced with
  `AliasDefn`.

There may also be other [scoped definitions](#scoped-definitions).

#### Scoped Definitions

The following operations are valid at the module level as well as in dataflow
regions and control-flow regions:

- `Const<T>` : a static constant value of type T stored in the node
  weight. Like `FuncDecl` and `FuncDefn` this has one `Const<T>` out-edge per use.

A **loadable HUGR** is a module HUGR where all input ports are connected and there are
no `FuncDecl/AliasDecl` nodes.

An **executable HUGR** or **executable module** is a loadable HUGR where the
root Module node has a `FuncDefn` child with function name
"main", that is the designated entry point. Modules that act as libraries need
not be executable.

#### Dataflow

Within dataflow regions, which include function definitions,
the following basic dataflow operations are available (in addition to the
[scoped definitions](#scoped-definitions)):

- `Input/Output`: input/output nodes, the outputs of `Input` node are
  the inputs to the function, and the inputs to `Output` are the
  outputs of the function.
- `Call`: Call a statically defined function. There is an incoming
  `Function` edge to specify the graph being called. The `Call`
  node specifies any type arguments to the function in the node weight,
  and the signature of the node (defined by its incoming and outgoing `Value` edges)
  matches the (type-instantiated) function being called.
- `LoadConstant<T>`: has an incoming `Const<T>` edge, where `T` is a `CopyableType`, and a
  `Value<T>` output, used to load a static constant into the local
  dataflow graph.
- `LoadFunction`: has an incoming `Function` edge and a `Value<FunctionType>`
  output. The `LoadFunction` node specifies any type arguments to the function
  in the node weight, and the `FunctionType` in the output edge matches the
  (type-instantiated) function in the incoming `Function` edge.
- `identity<T>`: pass-through, no operation is performed.
- `DFG`: A nested dataflow graph.
  These nodes are parents in the hierarchy.
  The signature of the operation comprises the output signature of the child
  Input node (as input) and the input signature of the child Output node (as
  output).
- `ExtensionOp`: an operation defined by an [Extension](#extension-system).

The example below shows two DFGs, one nested within the other. Each has an Input
and an Output node, whose outputs and inputs respectively match the inputs and
outputs of the containing DFG.

```mermaid
flowchart
    direction TB
    subgraph DFG0
        direction TB
        Input0 --> op0
        Input0 --> op1
        op0 --> op1
        subgraph DFG1
            direction TB
            Input1 --> op2
            Input1 --> op3
            op2 --> op4
            op3 --> op4
            op4 --> Output1
        end
        op0 --> DFG1
        op1 --> DFG1
        DFG1 --> Output0
    end
    A --> DFG0
    A --> DFG0
    DFG0 --> B
```

#### Control Flow

In a dataflow graph, the evaluation semantics are simple: all nodes in
the graph are necessarily evaluated, in some order (perhaps parallel)
respecting the Dataflow edges. The following operations are used to
express control flow, i.e. conditional or repeated evaluation.

##### `Conditional` nodes

These are parents to multiple `Case` nodes; the children have no edges.
The first input to the Conditional-node is of Sum type (see below), whose
arity matches the number of children of the Conditional-node. At runtime
the constructor (tag) selects which child to execute; the elements of the tagged row
of the Sum, with all remaining inputs to Conditional
appended, are sent to this child, and all outputs of the child are the
outputs of the Conditional; that child is evaluated, but the others are
not. That is, Conditional-nodes act as "if-then-else" followed by a
control-flow merge.


```mermaid
flowchart
    subgraph Conditional
        direction LR
        subgraph Case0["Case 0"]
            C0I["case 0 inputs + other inputs"] --> op0["operations"]
            op0 --> C0O["outputs"]
        end
        subgraph Case1["Case 1"]
            C1I["case 1 inputs + other inputs"] --> op1["operations"]
            op1 --> C1O["outputs"]
        end
        Case0 ~~~ Case1
    end
    Sum["case 0 inputs | case 1 inputs"] --> Conditional
    OI["other inputs"] --> Conditional
    Conditional --> outputs
```

##### `TailLoop` nodes

These provide tail-controlled loops. The dataflow sibling graph within the
TailLoop-node defines the loop body: this computes a row of outputs, whose
first element has type `Sum(#I, #O)` and the remainder is a row `#X`
(perhaps empty). Inputs to the contained graph and to the TailLoop node itself
are the row `#I:#X`, where `:` indicates row concatenation (with the row
inside the `Sum`).

Evaluation of the node begins by feeding the node inputs into the child graph
and evaluating it.  The `Sum` produced controls iteration of the loop:

- The first variant (`#I`) means that these values, along with the other
 sibling-graph outputs `#X`, are fed back into the top of the loop,
 and the body is evaluated again (thus perhaps many times)
- The second variant (`#O`) means that evaluation of the `TailLoop` node
 terminates, returning all the values produced as a row of outputs `#O:#X`.

##### Control Flow Graphs

When Conditional and `TailLoop` are not sufficient, the HUGR allows
arbitrarily-complex (even irreducible) control flow via an explicit `CFG` node:
a dataflow node defined by a child control sibling graph. This sibling
graph contains `BasicBlock` nodes (and [scoped definitions](#scoped-definitions)),
with the `BasicBock` nodes connected by `ControlFlow` edges (not dataflow).
`BasicBlock`-nodes only exist as children of `CFG`-nodes.

There are two kinds of `BasicBlock`: `DFB` (dataflow block) and `Exit`.
Each `DFB` node is parent to a dataflow sibling graph. `Exit` blocks
have only incoming control-flow edges, and no children.

The first child of the `CFG` is the entry block and must be a `DFB`,
with inputs the same as the CFG-node; the second child is an
`Exit` node, whose inputs match the outputs of the CFG-node.
The remaining children are either `DFB`s or [scoped definitions](#scoped-definitions).

The first output of the graph contained in a `DFB` has type
`Sum(\#t(0),...,#t(n-1))`, where the node has `n` successors, and the
remaining outputs are a row `#x`. `#t(i)` with `#x` appended matches the
inputs of successor `i`.

Some normalizations are possible:

- If the entry node has no predecessors (i.e. is not a loop header),
  then its contents can be moved outside the CFG node into a containing
  graph.
- If the entry node has only one successor and that successor is the
  exit node, the CFG node itself can be removed.

The CFG in the example below has three inputs: one (call it `v`) of type "P"
(not specified, but with a conversion to boolean represented by the nodes labelled "P?1" and "P?2"), one of
type "qubit" and one (call it `t`) of type "angle".

The CFG has the effect of performing an `Rz` rotation on the qubit with angle
`x`. where `x` is the constant `C` if `v` and `H(v)` are both true and `G(F(t))`
otherwise. (`H` is a function from type "P" to type "P" and `F` and `G` are
functions from type "angle" to type "angle".)

The `DFB` nodes are labelled `Entry` and `BB1` to `BB4`. Note that the first
output of each of these is a sum type, whose arity is the number of outgoing
control edges; the remaining outputs are those that are passed to all
succeeding nodes.

The three nodes labelled "Tag 0" are simply generating a 1-variant unary Sum (i.e. a Sum of one variant with empty rows) to the Output node.

```mermaid
flowchart
    subgraph CFG
        direction TB
        subgraph Entry
            direction TB
            EntryIn["Input"] -- "angle" --> F
            EntryIn -- "P" --> Entry_["P?1"]
            Entry_ -- "[|P]" --> EntryOut["Output"]
            F -- "angle" --> EntryOut
            EntryIn -- "qubit" --> EntryOut
        end
        subgraph BB1
            direction TB
            BB1In["Input"] -- "angle" --> G
            BB1_["Tag 0"] -- "[]" --> BB1Out["Output"]
            BB1In -- "qubit" --> BB1Out
            G -- "angle" --> BB1Out
        end
        subgraph BB2
            direction TB
            BB2In["Input"] -- "P" --> H -- "P" --> BB2_["P?2"]
            BB2_ -- "[angle|]" --> BB2Out["Output"]
            BB2In -- "angle" --> BB2_
            BB2In -- "qubit" --> BB2Out
        end
        subgraph BB3
            direction TB
            BB3In["Input"]
            BB3_["Tag 0"] -- "[]" --> BB3Out["Output"]
            BB3In -- "qubit" --> BB3Out
            C -- "angle" --> BB3Out
        end
        subgraph BB4
            direction TB
            BB4In["Input"] -- "qubit" --> Rz
            BB4In -- "angle" --> Rz
            BB4_["Tag 0"] -- "[]" --> BB4Out["Output"]
            Rz -- "qubit" --> BB4Out
        end
        subgraph Exit
        end
        Entry -- "0" --> BB1
        Entry -- "1" --> BB2
        BB2 -- "0" --> BB1
        BB2 -- "1" --> BB3
        BB1 -- "0" --> BB4
        BB3 -- "0" --> BB4
        BB4 -- "0" --> Exit
    end
    A -- "P" --> CFG
    A -- "qubit" --> CFG
    A -- "angle" --> CFG
    CFG -- "qubit" --> B
    linkStyle 25,26,27,28,29,30,31 stroke:#ff3,stroke-width:4px;
```

#### Hierarchical Relationships and Constraints

To clarify the possible hierarchical relationships, using the operation
definitions above and also defining "*O"* to be all non-nested dataflow
operations, we can define the relationships in the following table.
**D** and **C** are useful (and intersecting) groupings of operations:
dataflow nodes and the nodes which contain them. The "Parent" column in the
table applies unless the node in question is a root node of the HUGR (when it
has no parent).

| **Hierarchy**             | **Edge kind**                  | **Node Operation** | **Parent**    | **Children (\>=1)**      | **Child Constraints**                    |
| ------------------------- | ------------------------------ | ------------------ | ------------- | ------------------------ | ---------------------------------------- |
| Leaf                      | **D:** Value (Data dependency) | O, `Input/Output`  | **C**         | \-                       |                                          |
| CFG container             | **D**                          | CFG                | **C**         | `BasicBlock`             | First(second) is entry(exit)             |
| Conditional               | **D**                          | `Conditional`      | **C**         | `Case`                   | No edges                                 |
| **C:** Dataflow container | **D**                          | `TailLoop`         | **C**         |  **D**                   | First(second) is `Input`(`Output`)       |
| **C**                     | **D**                          | `DFG`              | **C**         |  **D**                   | First(second) is `Input`(`Output`)       |
| **C**                     | `Function`                     | `FuncDefn`         | **C**         |  **D**                   | First(second) is `Input`(`Output`)       |
| **C**                     | `ControlFlow`                  | `DFB`              | CFG           |  **D**                   | First(second) is `Input`(`Output`)       |
| **C**                     | \-                             | `Case`             | `Conditional` |  **D**                   | First(second) is `Input`(`Output`)       |
| Root                      | \-                             | `Module`           | none          |  **D**                   | Contains main `FuncDefn` for executable HUGR. |

These relationships allow to define two common varieties of sibling
graph:

**Control Flow Sibling Graph (CSG)**: where all nodes are
`BasicBlock`-nodes, and all edges are control-flow edges, which may have
cycles. The common parent is a CFG-node.

**Dataflow Sibling Graph (DSG)**: nodes are operations, `CFG`,
`Conditional`, `TailLoop` and `DFG` nodes; edges are `Value`, `Order` and `Static`, and must be acyclic.
(Thus a valid ordering of operations can be achieved by topologically sorting the
nodes.)
There is a unique Input node and Output node.
The common parent may be a `FuncDefn`, `TailLoop`, `DFG`, `Case` or `DFB` node.

| **Edge Kind**  | **Locality** |
| -------------- | ------------ |
| Hierarchy      | Defines hierarchy; each node has \<=1 parent                                                                                                                                                            |
| Order, Control | Local (Source + target have same parent) |
| Value          | Local, Ext or Dom - see [Edge Locality](#edge-locality) |
| Static         | Local or Ext - see [Edge Locality](#edge-locality) |

### Edge Locality

There are three possible `CopyableType` edge localities:

- `Local`: Source and target nodes must have the same parent.
- `Ext`: Edges "in" from a dataflow ancestor.
- `Dom`: Edges from a dominating basic block in a control-flow graph.

We allow non-local Dataflow edges
n<sub>1</sub>→n<sub>2</sub> where parent(n<sub>1</sub>) \!=
parent(n<sub>2</sub>) when the edge's locality is:

- for Value edges, Ext or Dom;
- for Static edges, Ext.

Each of these localities have additional constraints as follows:

1. For Ext edges, we require parent(n<sub>1</sub>) ==
   parent<sup>i</sup>(n<sub>2</sub>) for some i\>1, *and* for Value edges only there must be a order edge from n<sub>1</sub> to
   parent<sup>i-1</sup>(n<sub>2</sub>).

   The order edge records the
   ordering requirement that results, i.e. it must be possible to
   execute the entire n<sub>1</sub> node before executing
   parent<sup>i-1</sup>(n<sub>2</sub>). (Further recall that
   order+value edges together must be acyclic). We record the
   relationship between the Value edge and the
   corresponding order edge via metadata on each edge.

   For Static edges this order edge is not required since the source is
   guaranteed to causally precede the target.

2. For Dom edges, we must have that parent<sup>2</sup>(n<sub>1</sub>)
   == parent<sup>i</sup>(n<sub>2</sub>) is a CFG-node, for some i\>1,
   **and** parent(n<sub>1</sub>) strictly dominates
   parent<sup>i-1</sup>(n<sub>2</sub>) in the CFG (strictly as in
   parent(n<sub>1</sub>) \!= parent<sup>i-1</sup>(n<sub>2</sub>). (The
   i\>1 allows the node to target an arbitrarily-deep descendant of the
   dominated block, similar to an Ext edge.)

Specifically, these rules allow for edges where in a given execution of
the HUGR the source of the edge executes once, but the target may
execute \>=0 times.

The diagram below is equivalent to the diagram in the [Dataflow](#dataflow)
section above, but the input edge to "op3" has been replaced with a non-local
edge from the surrounding DFG (the thick arrow).

```mermaid
flowchart
    direction TB
    subgraph DFG0
        direction TB
        Input0 --> op0
        Input0 --> op1
        op0 --> op1
        subgraph DFG1
            direction TB
            Input1 --> op2
            op2 --> op4
            op3 --> op4
            op4 --> Output1
        end
        op0 --> DFG1
        DFG1 --> Output0
    end
    op1 ==> op3
    A --> DFG0
    A --> DFG0
    DFG0 --> B
```

This mechanism allows for some values to be passed into a block
bypassing the input/output nodes, and we expect this form to make
rewrites easier to spot. The constraints on input/output node signatures
remain as before.

HUGRs with only local Dataflow edges may still be useful for e.g. register
allocation, as that representation makes storage explicit. For example,
when a true/false subgraph of a Conditional-node wants a value from the
outside, we add an outgoing port to the Input node of each subgraph, a
corresponding incoming port to the Conditional-node, and discard nodes to each
subgraph that doesn't use the value. It is straightforward to turn an
edge between graphs into a combination of intra-graph edges and extra
input/output ports+nodes in such a way, but this is akin to
decompression.

Conversion from only local edges to a smallest total number of edges
(using non-local edges to reduce their number) is much more complex,
akin to compression, as it requires elision of useless split-merge
diamonds and other patterns and will likely require computation of
(post/)dominator trees. (However this will be somewhat similar to the
analysis required to move computations out of a CFG-node into
Conditional- and TailLoop-nodes). Note that such conversion could be
done for only a subpart of the HUGR at a time.

The following CFG is equivalent to the previous example. In this diagram:

- the thick arrow from "angle source" to "F" is an `Ext` edge (from an
  ancestral DFG into the CFG's entry block);
- the thick arrow from "F" to "G" is a `Dom` edge (from a dominating basic
  block);
- the `Rz` operation has been moved outside the CFG into the surrounding DFG, so
  the qubit does not need to be passed in to the CFG.

As a further normalization it would be possible to move F out of the CFG.
Alternatively, as an optimization it could be moved into the BB1 block.

Indeed every time a SESE region
is found within a CFG (where block *a* dominates *b*, *b* postdominates
*a*, and every loop containing either *a* or *b* contains both), it can
be normalized by moving the region bracketted by *a…b* into its own
CFG-node.

```mermaid
flowchart
    subgraph CFG
        direction TB
        subgraph Entry
            direction TB
            EntryIn["Input"] -- "P" --> Entry_["P?1"]
            Entry_ -- "[()|(P)]" --> EntryOut["Output"]
            F
        end
        subgraph BB1
            direction TB
            BB1In["Input"]
            BB1_["Const"] -- "[()]" --> BB1Out["Output"]
            G -- "angle" --> BB1Out
            BB1In ~~~ G
        end
        subgraph BB2
            direction TB
            BB2In["Input"] -- "P" --> H -- "P" --> BB2_["P?2"]
            BB2_ -- "[()|()]" --> BB2Out["Output"]
        end
        subgraph BB3
            direction TB
            BB3In["Input"]
            BB3_["Const"] -- "[()]" --> BB3Out["Output"]
            C -- "angle" --> BB3Out
            BB3In ~~~ C
        end
        subgraph Exit
        end
        Entry -- "0" --> BB1
        Entry -- "1" --> BB2
        BB2 -- "0" --> BB1
        BB2 -- "1" --> BB3
        BB1 -- "0" --> Exit
        BB3 -- "0" --> Exit
    end
    A -- "P" --> CFG
    A -- "qubit" --> Rz_out["Rz"]
    CFG -- "angle" --> Rz_out
    Rz_out -- "qubit" --> B
    A == "angle" ==> F
    F == "angle" ==> G
    linkStyle 12,13,14,15,16,17 stroke:#ff3,stroke-width:4px;
```

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

- A type which operations can use to indicate an error occurred.

#### Catch

- At some point we expect to add a first-order `catch` node, somewhat
  like a DFG-node. This contains a DSG, and (like a DFG node) has
  inputs matching the child DSG; but one output, of type
  `Sum(#O,#(ErrorType))` where O is the outputs of the child DSG.
- It is also possible to define a higher-order `catch` operation in an
  extension, taking a graph argument.

### Extensible metadata

Each node in the HUGR may have arbitrary metadata attached to it. This
is preserved during graph modifications, and,
[when possible](#metadata-updates-on-replacement), copied when rewriting.
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
interfere with each other's metadata; for example a reverse-DNS system
(`com.quantinuum.username....` or `com.quantinuum.tket....`). The values
are tuples of (1) any serializable struct, and (2) a list of node
indices. References from the serialized struct to other nodes should
indirect through the list of node indices stored with the struct.

**TODO**: Specify format, constraints, and serialization. Is YAML syntax
appropriate?

There is an API to add metadata, or extend existing metadata, or read
existing metadata, given the node ID.

**TODO** Examples illustrating this API.

**TODO** Do we want to reserve any top-level metadata keys, e.g. `Name`,
`Ports` (for port metadata) or `History` (for use by the rewrite
engine)?

Reserved metadata keys used by the HUGR tooling are prefixed with `core.`.
Use of this prefix by external tooling may cause issues.

#### Generator Metadata
Tooling generating HUGR can specify some reserved metadata keys to be used for debugging
purposes.

The key `core.generator` when used on the module root node is
used to specify the tooling used to generate the module.
The associated value must be an object/dictionary containing the fields `name`
and `version`, each with string values. Extra fields may be used to include
additional data about generating tooling that may be useful for debugging. Example:

```json
{
  "core.generator": { "name": "my_compiler", "version": "1.0.0" }
}
```

The key `core.used_extensions` when used on the module root node is
used to specify the names and versions of all the extensions used in the module.
Some of these may correspond to extensions packaged with the module, but they
may also be extensions the consuming tooling has pre-loaded. They can be used by the
tooling to check for extension version mismatches. The value associated with the key
must be an array of objects/dictionaries containing the keys `name` and `version`, each
with string values. Example:
```json
{
  "core.used_extensions": [{ "name": "my_ext", "version": "2.2.3" }]
}
```



**TODO** Do we allow per-port metadata (using the same mechanism?)

**TODO** What about references to ports? Should we add a list of port
indices after the list of node indices?

## Type System

There are two classes of type: `AnyType` $\supset$ `CopyableType`. Types in these
classes are distinguished by whether the runtime values of those types can be implicitly
copied or discarded (multiple or 0 links from on output port respectively):

- For the broadest class (`AnyType`), the only operation supported is the identity operation (aka no-op, or `lift` - see [Extension Tracking](#extension-tracking) below). Specifically, we do not require it to be possible to copy or discard all values, hence the requirement that outports of linear type must have exactly one edge. (That is, a type not known to be in the copyable subset).

    In fully qubit-counted contexts programs take in a number of qubits as input and return the same number, with no discarding.

- The smaller class is `CopyableType`, i.e. types holding ordinary classical
  data, where values can be copied (and discarded, the 0-ary copy). This
  allows multiple (or 0) outgoing edges from an outport; also these types can
  be sent down `Const` edges.

Note that all dataflow inputs (`Value`, `Const` and `Function`) always require a single connection, regardless of whether the type is `Linear` or `Copyable`.

**Rows** The `#` is a *row* which is a sequence of zero or more types. Types in the row can optionally be given names in metadata i.e. this does not affect behaviour of the HUGR. When writing literal types, we use `#` to distinguish between tuples and rows, e.g. `(int<1>,int<2>)` is a tuple while `Sum(#(int<1>),#(int<2>))` contains two rows.

The Hugr defines a number of type constructors, that can be instantiated into types by providing some collection of types as arguments. The constructors are given in the following grammar:

```haskell

Extensions ::= (Extension)* -- a set, not a list

Type ::= Sum([#]) -- disjoint union of rows of other types, tagged by unsigned int
       | Opaque(Name, [TypeArg]) -- a (instantiation of a) custom type defined by an extension
       | Function(#, #, Extensions) -- monomorphic function: arguments, results, and delta (see below)
       | Variable -- refers to a TypeParam bound by the nearest enclosing FuncDefn node or polymorphic type scheme
```

(We write `[Foo]` to indicate a list of Foo's.)

Tuples are represented as Sum types with a single variant. The type `(int<1>,int<2>)` is represented as `Sum([#(int<1>,int<2>)])`.

The majority of types will be Opaque ones defined by extensions including the [standard library](#standard-library). However a number of types can be constructed using only the core type constructors: for example the empty tuple type, aka `unit`, with exactly one instance (so 0 bits of data); the empty sum, with no instances; the empty Function type (taking no arguments and producing no results - `void -> void`); and compositions thereof.

Sums are `CopyableType` if all their components are; they are also fixed-size if their components are.

### Polymorphism

While function *values* passed around the graph at runtime have types that are monomorphic,
`FuncDecl` and `FuncDefn` nodes have not types but *type schemes* that are *polymorphic*---that is,
such declarations may include (bind) any number of type parameters, of kinds as follows:

```haskell
TypeParam ::= Type(Any|Copyable)
            | BoundedUSize(u64|) -- note optional bound
            | Extensions
            | String
            | Bytes
            | Float
            | List(TypeParam) -- homogeneous, any sized
            | Tuple([TypeParam]) -- heterogenous, fixed size
            | Opaque(Name, [TypeArg]) -- e.g. Opaque("Array", [5, Opaque("usize", [])])
```

The same mechanism is also used for polymorphic OpDefs, see [Extension Implementation](#extension-implementation).

Within the type of the Function node, and within the body (Hugr) of a `FuncDefn`,
types may contain "type variables" referring to those TypeParams.
The type variable is typically a type, but not necessarily, depending upon the TypeParam.

When a `FuncDefn` or `FuncDecl` is `Call`ed, the `Call` node statically provides
TypeArgs appropriate for the function's TypeParams:

```haskell
TypeArg ::= Type(Type) -- could be a variable of kind Type, or contain variable(s)
          | BoundedUSize(u64)
          | String(String)
          | Bytes([u8])
          | Float(f64)
          | Extensions(Extensions) -- may contain TypeArg's of kind Extensions
          | List([TypeArg])
          | Tuple([TypeArg])
          | Opaque(Value)
          | Variable -- refers to an enclosing TypeParam (binder) of any kind above
```

For example, a Function node declaring a `TypeParam::Opaque("Array", [5, TypeArg::Type(Type::Opaque("usize"))])`
means that any `Call` to it must statically provide a *value* that is an array of 5 `usize`s;
or a Function node declaring a `TypeParam::BoundedUSize(5)` and a `TypeParam::Type(Linear)` requires two TypeArgs,
firstly a non-negative integer less than 5, secondly a type (which might be from an extension, e.g. `usize`).

Given TypeArgs, the body of the Function node's type can be converted to a monomorphic signature by substitution,
i.e. replacing each type variable in the body with the corresponding TypeArg. This is guaranteed to produce
a valid type as long as the TypeArgs match the declared TypeParams, which can be checked in advance.

(Note that within a polymorphic type scheme, type variables of kind `List`, `Tuple` or `Opaque` will only be usable
as arguments to Opaque types---see [Extension System](#extension-system).)

#### Row Variables

Type variables of kind `TypeParam::List(TypeParam::Type(_))` are known as
"row variables" and along with type parameters of the same kinds are given special
treatment, as follows:
* A `TypeParam` of such kind may be instantiated with not just a `TypeArg::List`
  but also a single `TypeArg::Type`. (This is purely a notational convenience.)
  For example, `Type::Function(usize, unit, <exts>)` is equivalent shorthand
  for `Type::Function(#(usize), #(unit), <exts>)`.
* When a `TypeArg::List` is provided as argument for such a TypeParam, we allow
  elements to be a mixture of both types (including variables of kind
  `TypeParam::Type(_)`) and also row variables. When such variables are instantiated
  (with other `List`s) the elements of the inner `List` are spliced directly into
  the outer (concatenating their elements), eliding the inner (`List`) wrapper.

For example, a polymorphic FuncDefn might declare a row variable X of kind
`TypeParam::List(TypeParam::Type(Copyable))` and have as output a (tuple) type
`Sum([#(X, usize)])`. A call that instantiates said type-parameter with
`TypeArg::List([usize, unit])` would then have output `Sum([#(usize, unit, usize)])`.

See [Declarative Format](#declarative-format) for more examples.

Note that since a row variable does not have kind Type, it cannot be used as the type of an edge.

## Extension System

### Goals and constraints

The goal here is to allow the use of operations and types in the
representation that are user defined, or defined and used by extension
tooling. These operations cover various flavours:

- Instruction sets specific to a target.
- Operations that are best expressed in some other format that can be
  compiled into a graph (e.g. ZX).
- Ephemeral operations used by specific compiler passes.

A nice-to-have for this extensibility is a human-friendly format for
specifying such operations.

The key difficulty with this task is well stated in the [MLIR Operation
Definition Specification
docs](https://mlir.llvm.org/docs/DefiningDialects/Operations/#motivation)
:

> MLIR allows pluggable dialects, and dialects contain, among others, a
> list of operations. This open and extensible ecosystem leads to the
> "stringly" type IR problem, e.g., repetitive string comparisons
> during optimization and analysis passes, unintuitive accessor methods
> (e.g., generic/error prone `getOperand(3)` vs
> self-documenting `getStride()`) with more generic return types,
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
provide a human-friendly-enough definition experience.  However, we also
provide a declarative YAML format, below.

Ultimately though, we cannot avoid the "stringly" type problem if we
want *runtime* extensibility - extensions that can be specified and used
at runtime. In many cases this is desirable.

### Extension Implementation

To strike a balance then, every extension provides declarative structs containing
named **TypeDef**s and **OpDef**s---see [Declarative Format](#declarative-format).
These are (potentially polymorphic) definitions of types and operations, respectively---polymorphism arises because both may
declare any number of TypeParams, as per [Polymorphism](#polymorphism). To use a TypeDef as a type,
it must be instantiated with TypeArgs appropriate for its TypeParams, and similarly
to use an OpDef as a node operation: each `ExtensionOp` node stores a static-constant list of TypeArgs.

For TypeDef's, any set of TypeArgs conforming to its TypeParams, produces a valid type.
However, for OpDef's, greater flexibility is allowed: each OpDef *either*

1. Provides a polymorphic type scheme, as per [Type System](#type-system), which may declare TypeParams;
   values (TypeArgs) provided for those params will be substituted in. *Or*
2. The extension may self-register binary code (e.g. a Rust trait) providing a function
   `compute_signature` that fallibly computes a (perhaps-polymorphic) type scheme given some initial type arguments.
   The operation declares the arguments required by the `compute_signature` function as a list
   of TypeParams; if this successfully returns a type scheme, that may have additional TypeParams
   for which TypeArgs must also be provided.

For example, the TypeDef for `array` in the prelude declares two TypeParams: a `BoundedUSize`
(the array length) and a `Type`. Any valid instantiation (e.g. `array<5, usize>`) is a type.
Much the same applies for OpDef's that provide a `Function` type, but binary `compute_signature`
introduces the possibility of failure (see full details in [appendix](#appendix-3-binary-compute_signature)).

When serializing the node, we also serialize the type arguments; we can also serialize
the resulting (computed) type with the operation, and this will be useful when the type
is computed by binary code, to allow the operation to be treated opaquely by tools that
do not have the binary code available. (That is: the YAML definition, including all types
but only OpDefs that do not have binary `compute_signature`, can be sent with the HUGR).

This mechanism allows new operations to be passed through tools that do not understand
what the operations *do*---that is, new operations may be be defined independently of
any tool, but without providing any way for the tooling to treat them as anything other
than a black box. Similarly, tools may understand that operations may consume/produce
values of new types---whose *existence* is carried in the YAML---but the *semantics*
of each operation and/or type are necessarily specific to both operation *and* tool
(e.g. compiler or runtime).

However we also provide ways for extensions to provide semantics portable across tools.
For types, there is a fallback to serialized json; for operations, extensions *may* provide
either or both:

1. binary code (e.g. a Rust trait) implementing a function `try_lower`
   that takes the type arguments and a set of target extensions and may fallibly return
   a subgraph or function-body-HUGR using only those target extensions.
2. a HUGR, that declares functions implementing those operations. This
   is a simple case of the above (where the binary code is a constant function) but
   easy to pass between tools. However note this will only be possible for operations
   with sufficiently simple type (schemes), and is considered a "fallback" for use
   when a higher-performance (e.g. native HW) implementation is not available.
   Such a HUGR may itself require other extensions.

Whether a particular OpDef provides binary code for `try_lower` is independent
of whether it provides a binary `compute_signature`, but it will not generally
be possible to provide a HUGR for an operation whose type cannot be expressed
using a polymorphic type scheme.

### Declarative format

The declarative format needs to specify some required data that is
needed by the compiler to correctly treat the operation (the minimum
case is opaque operations that should be left untouched). However, we
wish to also leave it expressive enough to specify arbitrary extra data
that may be used by compiler extensions. This suggests a flexible
standard format such as YAML would be suitable. (The internal Rust structs
may also be used directly.) Here we provide an
illustrative example:

See [Type System](#type-system) for more on Extensions.

```yaml
# may need some top level data, e.g. namespace?

# Import other header files to use their custom types
  # TODO: allow qualified, and maybe locally-scoped
imports: [Quantum, Array]

extensions:
- name: MyGates
  # Declare custom types
  types:
  - name: QubitVector
    description: "A vector of qubits"
    # Opaque types can take type arguments, with specified names
    params: [["size", USize]]
  operations:
  - name: measure
    description: "measure a qubit"
    signature:
      # The first element of each pair is an optional parameter name.
      inputs: [[null, Q]]  # Q is defined in Quantum extension
      outputs: [[null, Q], ["measured", B]]
  - name: ZZPhase
    description: "Apply a parametric ZZPhase gate"
    signature:
      inputs: [[null, Q], [null, Q], ["angle", Angle]]
      outputs: [[null, Q], [null, Q]]
    misc:
      # extra data that may be used by some compiler passes
      # and is passed to try_lower and compute_signature
      equivalent: [0, 1]
      basis: [Z, Z]
  - name: SU2
    description: "One qubit unitary matrix"
    params: # per-node values passed to the type-scheme interpreter, but not used in signature
      matrix: Opaque(complex_matrix,2,2)
    signature:
      inputs: [[null, Q]]
      outputs: [[null, Q]]
  - name: MatMul
    description: "Multiply matrices of statically-known size"
    params:  # per-node values passed to type-scheme-interpreter and used in signature
      i: USize
      j: USize
      k: USize
    signature:
      inputs: [["a", Array<i>(Array<j>(F64))], ["b", Array<j>(Array<k>(F64))]]
      outputs: [[null, Array<i>(Array<k>(F64))]]
      #alternative inputs: [["a", Opaque(complex_matrix,i,j)], ["b", Opaque(complex_matrix,j,k)]]
      #alternative outputs: [[null, Opaque(complex_matrix,i,k)]]
  - name: max_float
    description: "Variable number of inputs"
    params:
      n: USize
    signature:
      # Where an element of a signature has three subelements, the third is the number of repeats
      inputs: [[null, F64, n]] # (defaulting to 1 if omitted)
      outputs: [[null, F64, 1]]
  - name: ArrayConcat
    description: "Concatenate two arrays. Extension provides a compute_signature implementation."
    params:
      t: Type  # Classic or Quantum
      i: USize
      j: USize
    # inputs could be: Array<i>(t), Array<j>(t)
    # outputs would be, in principle: Array<i+j>(t)
    # - but default type scheme interpreter does not support such addition
    # Hence, no signature block => will look up a compute_signature in registry.
  - name: TupleConcat
    description: "Concatenate two tuples"
    params:
      a: List[Type]
      b: List[Type]
    signature:
      inputs: [[null, Sum(a)], [null, Sum(b)]] # Sums with single variant are tuples
      outputs: [[null, Sum([a,b])]] # Tuple of elements of a concatenated with elements of b
  - name: GraphOp
    description: "Involves running an argument Graph. E.g. run it some variable number of times."
    params:
      - r: ExtensionSet
    signature:
      inputs: [[null, Function[r](USize -> USize)], ["arg", USize]]
      outputs: [[null, USize]]
      extensions: [r] # Indicates that running this operation also invokes extensions r
    lowering:
      file: "graph_op_hugr.bin"
      extensions: ["arithmetic.int", r] # r is the ExtensionSet in "params"
```

**Implementation note** Reading this format into Rust is made easy by `serde` and
[serde\_yaml](https://github.com/dtolnay/serde-yaml) (see the
Serialization section). It is also trivial to serialize these
definitions in to the overall HUGR serialization format.

Note the only required fields are `name` and `description`. `signature` is optional, but if present
must have children `inputs` and `outputs`, each lists, and may have `extensions`.

The optional `misc` field is used for arbitrary YAML, which is read in as-is and passed to compiler
 passes and (if no `signature` is present) the`compute_signature` function; e.g. a pass can use the `basis` information to perform commutation.

The optional `params` field can be used to specify the types of static+const arguments to each operation
---for example the matrix needed to define an SU2 operation. If `params` are not specified
then it is assumed empty.

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

The meaning of "convex" is: if A and B are nodes in the convex set S,
then any sibling node on a path from A to B is also in S.

#### API methods

There are the following primitive operations.

##### Replacement methods

###### `SimpleReplace`

This method is used for simple replacement of dataflow subgraphs consisting of
leaf nodes. It works by replacing a convex induced subgraph of the main Hugr
with a replacement Hugr having the same signature.

To this end, we specify a `SiblingSubgraph` in terms of a set $X$ of nodes of
the containing Hugr, an input signature $I$ and an output signature $O$. $I$ is
represented as a vector of vectors of input ports of nodes of $X$, where the
outer vector corresponds to the signature and each inner vector corresponds to
one or more copies of a given input into the subgraph (all connected to the
same output port of the containing Hugr). $O$ is represented as a vector of
output ports of nodes in $X$ (possibly with repeats, which correspond to
copies).

Given a `SiblingSubgraph` $S = (X, I, O)$ of a Hugr $H$, and a DFG-rooted Hugr
$H^\prime$ with an input signature matching the outer vector of $I$ and an
output signature matching $O$, we can form a new Hugr by replacing the nodes of
$X$ in $H$ with the Hugr $H^\prime$.

###### `Replace`

This is the general subgraph-replacement method. Intuitively, it takes a set of
sibling nodes to remove and replace with a new set of nodes. The new set of
nodes is itself a HUGR with some "holes" (edges and nodes that get "filled in"
by the `Replace` operation). To fully specify the operation, some further data
are needed:

- The replacement may include container nodes with no children, which adopt
  the children of removed container nodes and prevent those children being
  removed.
- All new incoming edges from the retained nodes to the new nodes, all
  outgoing edges from the new nodes to the retained nodes, and any new edges
  that bypass the replacement (going between retained nodes) must be
  specified.

Given a set $S$ of nodes in a hugr, let $S^\*$ be the set of all nodes
descended from nodes in $S$ (i.e. reachable from $S$ by following hierarchy edges),
including $S$ itself.

A `NewEdgeSpec` specifies an edge inserted between an existing node and a new node.
It contains the following fields:

- `SrcNode`: the source node of the new edge.
- `TgtNode`: the target node of the new edge.
- `EdgeKind`: may be `Value`, `Order`, `Static` or `ControlFlow`.
- `SrcPos`: for `Value` and `Static` edges, the position of the source port;
  for `ControlFlow` edges, the position among the outgoing edges.
- `TgtPos`: (for `Value` and `Static` edges only) the desired position among
  the incoming ports to the new node.

The `Replace` method takes as input:

- the ID of a container node $P$ in $\Gamma$;
- a set $S$ of IDs of nodes that are children of $P$
- a Hugr $G$ whose root is a node of the same type as $P$.
  Note this Hugr need not be valid, in that it may be missing:
  - edges to/from some ports (i.e. it may have unconnected ports)---not just Copyable dataflow outputs, which may occur even in valid Hugrs, but also incoming and/or non-Copyable dataflow ports, and ControlFlow ports,
  - all children for some container nodes strictly beneath the root (i.e. it may have container nodes with no outgoing hierarchy edges)
  - some children of the root, for container nodes that require particular children (e.g.
    $\mathtt{Input}$ and/or $\mathtt{Output}$ if $P$ is a dataflow container, the exit node
    of a CFG, the required number of children of a conditional)
- a map $B$ *from* container nodes in $G$ that have no children *to* container nodes in $S^\*$
  none of which is an ancestor of another.
  Let $X$ be the set of children of nodes in the image of $B$, and $R = S^\* \setminus X^\*$.
- a list $\mu\_\textrm{inp}$ of `NewEdgeSpec` which all have their `TgtNode`in
  $G$ and `SrcNode` in $\Gamma \setminus R$;
- a list $\mu\_\textrm{out}$ of `NewEdgeSpec` which all have their `SrcNode`in
  $G$ and `TgtNode` in $\Gamma \setminus R$, where `TgtNode` and `TgtPos` describe
  an existing incoming edge of that kind from a node in $S^\*$.
- a list $\mu\_\textrm{new}$ of `NewEdgeSpec` which all have both `SrcNode` and `TgtNode`
  in $\Gamma \setminus R$, where `TgtNode` and `TgtPos` describe an existing incoming
  edge of that kind from a node in $S^\*$.

Note that considering all three $\mu$ lists together,

- the `TgtNode` + `TgtPos`s of all `NewEdgeSpec`s with `EdgeKind` == `Value` will be unique
- and similarly for `EdgeKind` == `Static`

The well-formedness requirements of Hugr imply that $\mu\_\textrm{inp}$,
$\mu\_\textrm{out}$ and $\mu\_\textrm{new}$ may only contain `NewEdgeSpec`s with
certain `EdgeKind`s, depending on $P$:

- if $P$ is a dataflow container, `EdgeKind`s may be `Order`, `Value` or `Static` only (no `ControlFlow`)
- if $P$ is a CFG node, `EdgeKind`s may be `ControlFlow`, `Value`, or `Static` only (no `Order`)
- if $P$ is a Module node, there may be `Value` or `Static` only (no `Order`).

(in the case of $P$ being a CFG or Module node, any `Value` edges will be nonlocal, like Static edges.)

The new hugr is then derived as follows:

1. Make a copy in $\Gamma$ of all the nodes in $G$ *except the root*, and all edges except
   hierarchy edges from the root.
2. For each $\sigma\_\mathrm{inp} \in \mu\_\textrm{inp}$, insert a new edge going into the new
   copy of the `TgtNode` of $\sigma\_\mathrm{inp}$ according to the specification $\sigma\_\mathrm{inp}$.
   Where these edges are from ports that currently have edges to nodes in $R$,
   the existing edges are replaced.
3. For each $\sigma\_\mathrm{out} \in \mu\_\textrm{out}$, insert a new edge going out of the new
   copy of the `SrcNode` of $\sigma\_\mathrm{out}$ according to the specification $\sigma\_\mathrm{out}$.
   For `Value` or Static edges, the target port must have an existing edge whose source is in $R$;
   this edge is removed.
4. For each $\sigma\_\mathrm{new} \in \mu\_\textrm{new}$, insert a new edge
   between the existing `SrcNode` and `TgtNode` in $\Gamma$. For `Value` or Static edges,
   the target port must have an existing edge whose source is in $R$; this edge is removed.
5. Let $N$ be the ordered list of the copies made in $\Gamma$ of the children of the root node of $G$.
   For each child $C$ of $P$ (in order), if $C \in S$, redirect the hierarchy edge $P \rightarrow C$ to
   target the next node in $N$. Stop if there are no more nodes in $N$.
   Add any remaining nodes in $N$ to the end of $P$'s list of children.
6. For each node $(n, b = B(n))$ and for each child $m$ of $b$, replace the
   hierarchy edge from $b$ to $m$ with a hierarchy edge from the new copy of
   $n$ to $m$ (preserving the order).
7. Remove all nodes in $R$ and edges adjoining them. (Reindexing may be
   necessary after this step.)

##### Outlining methods

###### `OutlineDFG`

Replace a DFG-convex subgraph with a single DFG node having the original
nodes as children.

###### `OutlineCFG`

Replace a set of CFG sibling nodes with a single BasicBlock node having a
CFG node child which has as its children the set of BasicBlock nodes
originally specified. The set of basic blocks must satisfy constraints:

- There must be at most one node in the set with incoming (controlflow) edges
 from nodes outside the set. Specifically,
  - *either* the set includes the CFG's entry node, and any edges from outside
    the set (there may be none or more) target said entry node;
  - *or* the set does not include the CFG's entry node, but contains exactly one
    node which is the target of at least one edge(s) from outside the set.
- The set may not include the Exit block.
- There must be exactly one edge from a node in the set to a node outside it.

Situations in which multiple nodes have edges leaving the set, or where the Exit block
would be in the set, can be converted to this form by a combination of InsertIdentity
operations and one Replace. For example, rather than moving the Exit block into the nested CFG:

1. An Identity node with a single successor can be added onto each edge into the Exit
2. If there is more than one edge into the Exit, these Identity nodes can then all be combined
   by a Replace operation changing them all for a single Identity (keeping the same number
   of in-edges, but reducing to one out-edge to the Exit).
3. The single edge to the Exit node can then be used as the exiting edge.

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
`n0`. (Thus acyclicity is preserved.) If there is already an order edge from
`n0` to `n1` this does nothing (but is not an error).

###### `RemoveOrder`

Given nodes `n0` and `n1`, if there is an Order edge from `n0` to `n1`,
remove it. (If there is an non-local edge from `n0` to a descendent of
`n1`, this invalidates the hugr. **TODO** should this be an error?)

##### Insertion and removal of const loads

###### `InsertConstIgnore`

Given a `Const<T>` node `c`, and optionally `P`, a parent of a DSG, add a new
`LoadConstant<T>` node `n` as a child of `P` with a `Const<T>` edge
from `c` to `n` and no outgoing edges from `n`.  Return the ID of `n`. If `P` is
omitted it defaults to the parent of `c` (in this case said `c` will
have to be in a DSG or CSG rather than under the Module Root.) If `P` is
provided, it must be a descendent of the parent of `c`.

###### `RemoveLoadConstant`

Given a `LoadConstant<T>` node `n` that has no outgoing edges, remove
it (and its incoming Static edge and any Order edges) from the hugr.

##### Insertion and removal of const nodes

###### `InsertConst`

Given a `Const<T>` node `c` and a container node `P` (either a `Module`,
 a `CFG` node or a dataflow container), add `c` as a child of `P`.

###### `RemoveConst`

Given a `Const<T>` node `c` having no outgoing edges, remove `c`.

#### Usage

Note that we can only reattach children into empty replacement
containers. This simplifies the API, and is not a serious restriction
since we can use the outlining and inlining methods to target a group of
nodes.

The most basic case -- replacing a convex set of Op nodes in a DSG with
another graph of Op nodes having the same signature -- is implemented by
`SimpleReplace`.

If one of the nodes in the region is a complex container node that we
wish to preserve in the replacement without doing a deep copy, we can
use an empty node in the replacement and have B map this node to the old
one.

We can, for example, implement "turning a Conditional-node with known
Sum into a DFG-node" by a `Replace` where the Conditional (and its
preceding Sum) is replaced by an empty DFG and the map B specifies
the "good" child of the Conditional as the surrogate parent of the new
DFG's children. (If the good child was just an Op, we could either
remove it and include it in the replacement, or -- to avoid this overhead
-- outline it in a DFG first.)

Similarly, replacement of a CFG node having a single BasicBlock child
with a DFG node can be achieved using `Replace` (specifying the
BasicBlock node as the surrogate parent for the new DFG's children).

Arbitrary node insertion on Dataflow edges can be achieved using
`InsertIdentity` followed by `Replace`. Removal of a node in a DSG
having input wires and output wires of the same type can be achieved
using `Replace` (with a set of `identity<T>` nodes) followed by
`RemoveIdentity`.

### Normalisation

We envisage that some kind of pass can be used after a rewrite or series
of rewrites to automatically apply RemoveLoadConstant for any unused
load\_constants, and other such
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
  - `History: {Replaced: [{node1: old_node1_metadata, node2:
    old_node2_metadata, ...}, {...}, ...]}` where `Replaced`
    specifies an ordered list of replacements, and the new
    replacement appends to the list (or creates a new list if
    `Replaced` doesn't yet exist);
- to the root node of Γ, attach metadata capturing a serialization of the
 replacement (both the set of nodes replaced and its replacement):
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
various node types within it. Starting from the root node, we can
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
- Ability to send over wire. Nexus will need to do things like:
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
conversion to/from the binary serialized form.

### Schema

We propose the following simple serialized structure, expressed here in
pseudocode, though we advocate MessagePack format in practice (see
[Serialization implementation](schema/serialization.md)).
Note in particular that hierarchical relationships
have a special encoding outside `edges`, as a field `parent`
in a node definition. Nodes are identified by their position in the `nodes`
list, starting from 0. The unique root node of the HUGR reports itself as the
parent.

The other required field in a node is `op` which identifies an operation by
name, and is used as a discriminating tag in validating the remaining fields.
The other fields are defining data for the particular operation, including
`params` which specifies the arguments to the `TypeParam`s of the operation.
Metadata could also be included as a map keyed by node index.

```rust
struct HUGR {
  nodes: [Node],
  edges: [Edge],
}

struct Node{
  // parent node index
  parent: Int,
  // name of operation
  op: String
  //other op-specific fields
  ...
}
// ((source, offset), (target, offset)
struct Edge = ((Int, Optional<Int>), (Int, Optional<Int>))
```

Node indices, used within the
definitions of nodes and edges, directly correspond to positions in the
`nodes` list. An edge is defined by the source and target nodes, and
optionally the offset of the output/input ports within those nodes, if the edge
kind is one that connects to a port. This scheme
enforces that nodes are contiguous - a node index must always point to a
valid node - whereas in tooling implementations it may be necessary to
implement stable indexing where removing a node invalidates that index
while keeping all other indices pointing to the same node.

Nodes with `Input` and `Output` children are expected to appear earlier in the
list than those children, and `Input` nodes should appear before their matching
`Output` nodes.

## Architecture

The HUGR is implemented as a Rust crate named `hugr`. This
crate is intended to be a common dependency for all projects, and is published
at the [crates.io registry](https://crates.io/crates/hugr).

The HUGR is represented internally using structures from the `portgraph`
crate. A base PortGraph is composed with hierarchy (as an alternate
implementation of `Hierarchy` relationships) and weight components. The
implementation of this design document is [available on GitHub](https://github.com/quantinuum/hugr).

## Standard Library

A collection of extensions form the "standard library" for HUGR, and are defined
in this repository.

### Prelude

The prelude extension is assumed to be valid and available in all contexts, and
so must be supported by all third-party tooling.

#### Types

`usize`: a positive integer size type.

`string`: a string type.

`array<N, T>`: a known-size (N) array of type T.

`qubit`: a linear (non-copyable) qubit type.

`error`: error type. See [`ErrorType`](#errortype).

### Operations

| Name              | Inputs           | Outputs       | Meaning                                                           |
|-------------------|------------------|---------------|------------------------------------------------------------------ |
| `print`           | `string`         | -             | Append the string to the program's output stream[^1] (atomically) |
| `new_array<N, T>` | `T` x N          | `array<N, T>` | Create an array from all the inputs                               |
| `panic`           | `ErrorType`, ... | ...           | Immediately end execution and pass contents of error to context. Inputs following the `ErrorType`, and all outputs, are arbitrary; these only exist so that structural constraints such as linearity can be satisfied. |

[^1] The existence of an output stream, and the processing of it either during
or after program execution, is runtime-dependent. If no output stream exists
then the `print` function is a no-op.

### Logic Extension

The Logic Extension provides a boolean type and basic logical operations.

The boolean type `bool` is defined to be `Sum(#(),#())`, with the convention that the
first option in the sum represents "false" and the second represents "true".

The following operations are defined:

| Name     | Inputs     | Outputs | Meaning                       |
| -------- | ---------- | ------- | ----------------------------- |
| `not`    | `bool`     | `bool`  | logical "not"                 |
| `and<N>` | `bool` x N | `bool`  | N-ary logical "and" (N \>= 0) |
| `or<N>`  | `bool` x N | `bool`  | N-ary logical "or"  (N \>= 0) |

Note that an `and<0>` operation produces the constant value "true" and an
`or<0>` operation produces the constant value "false".

### Arithmetic Extensions

Types and operations for integer and
floating-point operations are provided by a collection of extensions under the
namespace `arithmetic`.

We largely adopt (a subset of) the definitions of
[WebAssembly 2.0](https://webassembly.github.io/spec/core/index.html),
including the names of the operations. Where WebAssembly specifies a
"partial" operation (i.e. when the result is not defined on certain
inputs), we use a Sum type to hold the result.

A few additional operations not included in WebAssembly are also
specified, and there are some other small differences (highlighted
below).

#### `arithmetic.int.types`

The `int<N>` type is parametrized by its width `N`, which is a positive
integer.

The possible values of `N` are 2^i for i in the range [0,6].

The `int<N>` type represents an arbitrary bit string of length `N`.
Semantics are defined by the operations. There are three possible
interpretations of a value:

- as a bit string $(a_{N-1}, a_{N-2}, \ldots, a_0)$ where
  $a_i \in \\{0,1\\}$;
- as an unsigned integer $\sum_{i \lt N} 2^i a_i$;
- as a signed integer $\sum_{i \lt N-1} 2^i a_i - 2^{N-1} a_{N-1}$.

An asterix ( \* ) in the tables below indicates that the definition
either differs from or is not part of the
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html)
specification.

#### `arithmetic.int`

This extension defines operations on the integer types.

Casts:

| Name                   | Inputs   | Outputs                  | Meaning                                                                                      |
| ---------------------- | -------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| `iwiden_u<M,N>`( \* )  | `int<M>` | `int<N>`                 | widen an unsigned integer to a wider one with the same value (where M \<= N)                 |
| `iwiden_s<M,N>`( \* )  | `int<M>` | `int<N>`                 | widen a signed integer to a wider one with the same value (where M \<= N)                    |
| `inarrow_u<M,N>`( \* ) | `int<M>` | `Sum(#(int<N>), #(ErrorType))` | narrow an unsigned integer to a narrower one with the same value if possible, and an error otherwise (where M \>= N) |
| `inarrow_s<M,N>`( \* ) | `int<M>` | `Sum(#(int<N>), #(ErrorType))` | narrow a signed integer to a narrower one with the same value if possible, and an error otherwise (where M \>= N)    |
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

| Name                         | Inputs             | Outputs                                | Meaning                                                                                                                                                      |
|------------------------------|--------------------|----------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `imax_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | maximum of unsigned integers                                                                                                                                 |
| `imax_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | maximum of signed integers                                                                                                                                   |
| `imin_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | minimum of unsigned integers                                                                                                                                 |
| `imin_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | minimum of signed integers                                                                                                                                   |
| `iadd<N>`                    | `int<N>`, `int<N>` | `int<N>`                               | addition modulo 2^N (signed and unsigned versions are the same op)                                                                                           |
| `isub<N>`                    | `int<N>`, `int<N>` | `int<N>`                               | subtraction modulo 2^N (signed and unsigned versions are the same op)                                                                                        |
| `ineg<N>`                    | `int<N>`           | `int<N>`                               | negation modulo 2^N (signed and unsigned versions are the same op)                                                                                           |
| `imul<N>`                    | `int<N>`, `int<N>` | `int<N>`                               | multiplication modulo 2^N (signed and unsigned versions are the same op)                                                                                     |
| `idivmod_checked_u<N>`( \* ) | `int<N>`, `int<N>` | `Sum(#(int<N>, int<N>), #(ErrorType))` | given unsigned integers 0 \<= n \< 2^N, 0 \<= m \< 2^N, generates unsigned q, r where q\*m+r=n, 0\<=r\<m (m=0 is an error)                                   |
| `idivmod_u<N>`               | `int<N>`, `int<N>` | `(int<N>, int<N>)`                     | given unsigned integers 0 \<= n \< 2^N, 0 \<= m \< 2^N, generates unsigned q, r where q\*m+r=n, 0\<=r\<m (m=0 will call panic)                               |
| `idivmod_checked_s<N>`( \* ) | `int<N>`, `int<N>` | `Sum(#(int<N>, int<N>), #(ErrorType))` | given signed integer -2^{N-1} \<= n \< 2^{N-1} and unsigned 0 \<= m \< 2^N, generates signed q and unsigned r where q\*m+r=n, 0\<=r\<m (m=0 is an error)     |
| `idivmod_s<N>`( \* )         | `int<N>`, `int<N>` | `(int<N>, int<N>)`                     | given signed integer -2^{N-1} \<= n \< 2^{N-1} and unsigned 0 \<= m \< 2^N, generates signed q and unsigned r where q\*m+r=n, 0\<=r\<m (m=0 will call panic) |
| `idiv_checked_u<N>` ( \* )   | `int<N>`, `int<N>` | `Sum(#(int<N>),#( ErrorType))`         | as `idivmod_checked_u` but discarding the second output                                                                                                      |
| `idiv_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | as `idivmod_u` but discarding the second output                                                                                                              |
| `imod_checked_u<N>` ( \* )   | `int<N>`, `int<N>` | `Sum(#(int<N>), #(ErrorType))`         | as `idivmod_checked_u` but discarding the first output                                                                                                       |
| `imod_u<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | as `idivmod_u` but discarding the first output                                                                                                               |
| `idiv_checked_s<N>`( \* )    | `int<N>`, `int<N>` | `Sum(#(int<N>), #(ErrorType))`         | as `idivmod_checked_s` but discarding the second output                                                                                                      |
| `idiv_s<N>`                  | `int<N>`, `int<N>` | `int<N>`                               | as `idivmod_s` but discarding the second output                                                                                                              |
| `imod_checked_s<N>`( \* )    | `int<N>`, `int<N>` | `Sum(#(int<N>), #(ErrorType))`         | as `idivmod_checked_s` but discarding the first output                                                                                                       |
| `imod_s<N>`                  | `int<N>`, `int<M>` | `int<M>`                               | as `idivmod_s` but discarding the first output                                                                                                               |
| `iabs<N>`                    | `int<N>`           | `int<N>`                               | convert signed to unsigned by taking absolute value                                                                                                          |
| `iand<N>`                    | `int<N>`, `int<N>` | `int<N>`                               | bitwise AND                                                                                                                                                  |
| `ior<N>`                     | `int<N>`, `int<N>` | `int<N>`                               | bitwise OR                                                                                                                                                   |
| `ixor<N>`                    | `int<N>`, `int<N>` | `int<N>`                               | bitwise XOR                                                                                                                                                  |
| `inot<N>`                    | `int<N>`           | `int<N>`                               | bitwise NOT                                                                                                                                                  |
| `ishl<N>`( \* )              | `int<N>`, `int<N>` | `int<N>`                               | shift first input left by k bits where k is unsigned interpretation of second input (leftmost bits dropped, rightmost bits set to zero)                      |
| `ishr<N>`( \* )              | `int<N>`, `int<N>` | `int<N>`                               | shift first input right by k bits where k is unsigned interpretation of second input (rightmost bits dropped, leftmost bits set to zero)                     |
| `irotl<N>`( \* )             | `int<N>`, `int<N>` | `int<N>`                               | rotate first input left by k bits where k is unsigned interpretation of second input (leftmost bits replace rightmost bits)                                  |
| `irotr<N>`( \* )             | `int<N>`, `int<N>` | `int<N>`                               | rotate first input right by k bits where k is unsigned interpretation of second input (rightmost bits replace leftmost bits)                                 |
| `itostring_u<N>`             | `int<N>`           | `string`                               | decimal string representation of unsigned integer                                                                                                            |
| `itostring_s<N>`             | `int<N>`           | `string`                               | decimal string representation of signed integer                                                                                                              |


#### `arithmetic.float.types`

The `float64` type represents IEEE 754-2019 floating-point data of 64
bits.

Non-finite `float64` values (i.e. NaN and ±infinity) are not allowed in `Const`
nodes.

#### `arithmetic.float`

Floating-point operations are defined as follows. All operations below
follow
[WebAssembly](https://webassembly.github.io/spec/core/exec/numerics.html#floating-point-operations)
except where stated.

| Name              | Inputs               | Outputs   | Meaning                                                                  |
| ----------------- | -------------------- | --------- | ------------------------------------------------------------------------ |
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
| `ftostring`       | `float64`            | `string`  | string representation[^1]                                                  |

[^1] The exact specification of the float-to-string conversion is
implementation-dependent.

#### `arithmetic.conversions`

Conversions between integers and floats:

| Name           | Inputs    | Outputs                  | Meaning               |
| -------------- | --------- | ------------------------ | --------------------- |
| `trunc_u<N>`   | `float64` | `Sum(#(int<N>), #(ErrorType))` | float to unsigned int, rounding towards zero. Returns an error when the float is non-finite. |
| `trunc_s<N>`   | `float64` | `Sum(#(int<N>), #(ErrorType))` | float to signed int, rounding towards zero. Returns an error when the float is non-finite. |
| `convert_u<N>` | `int<N>`  | `float64`                | unsigned int to float |
| `convert_s<N>` | `int<N>`  | `float64`                | signed int to float   |
| `bytecast_int64_to_float64` | `int<6>`  | `float64`   | reinterpret an int64 as a float64 based on its bytes, with the same endianness. |
| `bytecast_float64_to_int64` | `float64` | `int64`     | reinterpret an float64 as an int based on its bytes, with the same endianness. |

## Glossary

- **BasicBlock node**: A child of a CFG node (i.e. a basic block
  within a control-flow graph).
- **Call node**: TODO
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
- **Dataflow edge** either a `Value` edge or a Static edge; has a type,
  and runs between an output port and an input port.
- **Dataflow Sibling Graph (DSG)**: The set of all children of a given
  Dataflow container node, with all edges between them. Includes
  exactly one input node (unique node having no input edges) and one
  output node (unique node having no output edges). Nodes are
  processes that operate on input data and produce output data. Edges
  in a DSG are either value or order edges. The DSG must be acyclic.
- **data-dependency node**: an input, output, operation, DFG, CFG,
  Conditional or TailLoop node. All incoming and outgoing edges are
  value edges.
- **FuncDecl node**: child of a module, indicates that an external
  function exists but without giving a definition. May be the source
  of `Function`-edges to Call nodes.
- **FuncDefn node**: child of a module node, defines a function (by being
  parent to the function's body). May be the source of `Function`-edges
  to Call nodes.
- **DFG node**: A node representing a data-flow graph. Its children
  are all data-dependency nodes.
- **edge kind**: There are six kinds of edge: `Value` edge, order edge, hierarchy edge,
  control-flow edge, `Const` edge and `Function` edge.
- **edge type:** Typing information attached to a value edge or Static
  edge (representing the data type of value that the edge carries).
- **entry node**: The distinguished node of a CFG representing the
  point where execution begins.
- **exit node**: The distinguished node of a CFG representing the
  point where execution ends.
- **function:** TODO
- **Conditional node:** TODO
- **hierarchy**: A tree whose nodes comprise all nodes of the HUGR,
  rooted at the HUGR's root node.
- **hierarchy edge**: An edge in the hierarchy tree. The edge is considered to
  be directed, with the source node the parent of the target node.
- **input node**: The distinguished node of a DSG representing the
  point where data processing begins.
- **input signature**: The input signature of a node is the mapping
  from identifiers of input ports to their associated edge types.
- **Inter-graph Edge**: Deprecated, see *non-local edge*
- **CFG node**: A node representing a control-flow graph. Its children
  are all BasicBlock nodes, of which there is exactly one entry node
  and exactly one exit node.
- **load-constant node**: TODO
- **metadata:** TODO
- **module**: TODO
- **node index**: An identifier for a node that is unique within the
  HUGR.
- **non-local edge**: A Value or Static edge with Locality Ext,
  or a Value edge with locality Dom (i.e. not Local)
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
- **extension**: TODO
- **sibling graph**: TODO
- **signature**: The signature of a node is the combination of its
  input and output signatures.
- **simple type**: a quantum or classical type annotated with the
  Extensions required to produce the value
- **Static edge**: either a `Const` or `Function` edge
- **order edge**: An edge implying dependency of the target node on
  the source node.
- **TailLoop node**: TODO
- **value edge:** An edge between data-dependency nodes. Has a fixed
  edge type.

## Appendices

### Appendix 1: Rationale for Control Flow

#### **Justification of the need for CFG-nodes**

- Conditional + TailLoop are not able to express arbitrary control
  flow without introduction of extra variables (dynamic overhead, i.e.
  runtime cost) and/or code duplication (static overhead, i.e. code
  size).
  - Specifically, the most common case is *shortcircuit evaluation*:
    `if (P && Q) then A; else B;` where Q is only evaluated if P is
    true.
- We *could* parse a CFG into a DSG with only Conditional-nodes and
  TailLoop-nodes by introducing extra variables, as per [Google
  paper](https://doi.org/10.1145/2693261), and then expect
  LLVM to remove those extra variables later. However that's a
  significant amount of analysis and transformation, which is
  undesirable for using the HUGR as a common interchange format (e.g.
  QIR → HUGR → LLVM) when little optimization is being done (perhaps
  no cross-basic-block optimization).
- It's possible that maintaining support for CFGs-nodes may become a
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

#### **Alternative representations considered but rejected**

- A [Google paper](https://doi.org/10.1145/2693261) allows
  for the introduction of extra variables into the DSG that can be
  eliminated at compile-time (ensuring no runtime cost), but only if
  stringent well-formedness conditions are maintained on the DSG, and
  there are issues with variable liveness.
- [Lawrence's
  thesis](https://doi.org/10.48456/tr-705)
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
- We could use function calls to avoid code duplication (essentially
  the return address is the extra boolean variable, likely to be very
  cheap). However, I think this means pattern-matching will want to
  span across function-call boundaries; and it rules out using
  non-local edges for called functions. **TODO** are those objections
  sufficient to rule this out?

##### Comparison with MLIR

There are a lot of broad similarities here, with MLIR's regions
providing hierarchy, and "graph" regions being like DSGs. Significant
differences include:

- MLIR uses names everywhere, which internally are mapped to some kind
  of hyperedge; we have explicit edges in the structure.
  - However, we can think of every output nodeport being a unique
    SSA/SSI name.
  - MLIR does not do linearity or SSI.
- Our CFGs are Single Entry Single Exit (results defined by the output
  node of the exit block), rather than MLIR's Single Entry Multiple
  Exit (with `return` instruction)
- MLIR allows multiple regions inside a single operation, whereas we
  introduce extra levels of hierarchy to allow this.
- I note re. closures that MLIR expects the enclosing scope to make
  sure any referenced values are kept 'live' for long enough. Not what
  we do in Tierkreis (the closure-maker copies them)\!

### Appendix 2: Node types and their edges

The following table shows which edge kinds may adjoin each node type.

Under each edge kind, the inbound constraints are followed by the outbound
constraints. The symbol ✱ stands for "any number", while + stands for "at least
one". For example, "1, ✱" means "one edge in, any number out".

The "Root" row of the table applies to whichever node is the HUGR root,
including `Module`.

| Node type      | `Value` | `Order` | `Const` | `Function` | `ControlFlow` | `Hierarchy` | Children |
|----------------|---------|---------|---------|------------|---------------|-------------|----------|
| Root           | 0, 0    | 0, 0    | 0, 0    | 0, 0       | 0, 0          | 0, ✱        |          |
| `FuncDefn`     | 0, 0    | 0, 0    | 0, 0    | 0, ✱       | 0, 0          | 1, +        | DSG      |
| `FuncDecl`     | 0, 0    | 0, 0    | 0, 0    | 0, ✱       | 0, 0          | 1, 0        |          |
| `AliasDefn`    | 0, 0    | 0, 0    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `AliasDecl`    | 0, 0    | 0, 0    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Const`        | 0, 0    | 0, 0    | 0, ✱    | 0, 0       | 0, 0          | 1, 0        |          |
| `LoadConstant` | 0, 1    | ✱, ✱    | 1, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `LoadFunction` | 0, 1    | ✱, ✱    | 0, 0    | 1, 0       | 0, 0          | 1, 0        |          |
| `Input`        | 0, ✱    | 0, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Output`       | ✱, 0    | ✱, 0    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Call`         | ✱, ✱    | ✱, ✱    | 0, 0    | 1, 0       | 0, 0          | 1, 0        |          |
| `DFG`          | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, +        | DSG      |
| `CFG`          | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, +        | CSG      |
| `DFB`          | 0, 0    | 0, 0    | 0, 0    | 0, 0       | ✱, ✱          | 1, +        | DSG      |
| `Exit`         | 0, 0    | 0, 0    | 0, 0    | 0, 0       | +, 0          | 1, 0        |          |
| `TailLoop`     | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, +        | DSG      |
| `Conditional`  | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, +        | `Case`   |
| `Case`         | 0, 0    | 0, 0    | 0, 0    | 0, 0       | 0, 0          | 1, +        | DSG      |
| `CustomOp`     | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Noop`         | 1, 1    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `MakeTuple`    | ✱, 1    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `UnpackTuple`  | 1, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Tag`          | 1, 1    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |
| `Lift`         | ✱, ✱    | ✱, ✱    | 0, 0    | 0, 0       | 0, 0          | 1, 0        |          |

### Appendix 3: Binary `compute_signature`

When an OpDef provides a binary `compute_signature` function, and an operation node uses that OpDef:

- the node provides a list of TypeArgs, at least as many as the $n$ TypeParams declared by the OpDef
- the first $n$ of those are passed to the binary `compute_signature`
- if the binary function returns an error, the operation is invalid;
- otherwise, `compute_signature` returns a type scheme (which may itself be polymorphic)
- any remaining TypeArgs in the node (after the first $n$) are then substituted into that returned type scheme
  (the number remaining in the node must match exactly).
  **Note** this allows the binary function to use the values (TypeArgs) passed in---e.g.
  by looking inside `List` or `Opaque` TypeArgs---to determine the structure (and degree of polymorphism) of the returned type scheme.
- We require that the TypeArgs to be passed to `compute_signature` (the first $n$)
  must *not* refer to any type variables (declared by ancestor nodes in the Hugr - the nearest enclosing FuncDefn);
  these first $n$ must be static constants unaffected by substitution.
  This restriction does not apply to TypeArgs after the first $n$.
