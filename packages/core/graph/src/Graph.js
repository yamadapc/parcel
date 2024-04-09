// @flow strict-local

import {ParcelGraphImpl} from '@parcel/rust';
import {fromNodeId} from './types';
import AdjacencyList, {type SerializedAdjacencyList} from './AdjacencyList';
import type {Edge, NodeId} from './types';
import type {
  TraversalActions,
  GraphVisitor,
  GraphTraversalCallback,
} from '@parcel/types';
import {BitSet} from './BitSet';

import nullthrows from 'nullthrows';

export type NullEdgeType = 1;
export type GraphOpts<TNode, TEdgeType: number = 1> = {|
  nodes?: Array<TNode | null>,
  adjacencyList?: SerializedAdjacencyList<TEdgeType>,
  rootNodeId?: ?NodeId,
|};

export type SerializedParcelGraph<TNode, TEdgeType: number = 1> = {|
  nodes: Array<TNode | null>,
  rootNodeId: ?NodeId,
|};

export type SerializedGraph<TNode, TEdgeType: number = 1> = {|
  nodes: Array<TNode | null>,
  adjacencyList: SerializedAdjacencyList<TEdgeType>,
  rootNodeId: ?NodeId,
|};

export type AllEdgeTypes = -1;
export const ALL_EDGE_TYPES: AllEdgeTypes = -1;

/**
 * Rust backed graph structure
 */
export class ParcelGraph<TNode, TEdgeType: number = 1> {
  nodesById: {[id: number]: TNode};
  inner: ParcelGraphImpl;
  rootNodeId: ?NodeId;
  _visited: ?BitSet;

  constructor(opts: ?GraphOpts<TNode, TEdgeType>) {
    // this.nodes = opts?.nodes || [];
    this.nodesById = opts?.nodesById ?? {};
    this.inner = opts?.graph
      ? ParcelGraphImpl.deserialize(opts?.graph)
      : new ParcelGraphImpl();

    if (opts?.graph) {
      console.log(opts.graph);
    }

    this.setRootNodeId(opts?.rootNodeId);
  }

  get nodes(): {|
    entries(): [NodeId, TNode][],
  |} {
    const nodes = Object.keys(this.nodesById);
    const values = nodes.map(key => this.nodesById[Number(key)]);
    values.entries = () => {
      return nodes.map(key => {
        const node = this.nodesById[Number(key)];
        return [Number(key), node];
      });
    };
    return values;
  }

  setRootNodeId(id: ?NodeId) {
    this.rootNodeId = id;
  }

  static deserialize(
    opts: GraphOpts<TNode, TEdgeType>,
  ): ParcelGraph<TNode, TEdgeType> {
    return new this({
      nodesById: opts.nodesById,
      graph: opts.graph,
      // adjacencyList: opts.adjacencyList,
      rootNodeId: opts.rootNodeId,
    });
  }

  // TODO serialise the graph itself
  serialize(): SerializedParcelGraph<TNode, TEdgeType> {
    return {
      // nodes: this.nodes,
      graph: this.inner.serialize(),
      nodesById: this.nodesById,
      rootNodeId: this.rootNodeId,
    };
  }

  // TODO
  // Returns an iterator of all edges in the graph. This can be large, so iterating
  // the complete list can be costly in large graphs. Used when merging graphs.
  getAllEdges(): Iterator<Edge<TEdgeType | NullEdgeType>> {
    // TODO
    // return this.adjacencyList.getAllEdges();
    return this.inner.getAllEdges().map(descr => ({
      from: descr.from,
      to: descr.to,
      type: descr.weight,
    }));
  }

  addNode(node: TNode): NodeId {
    let id = this.inner.addNode(0);
    // console.log('addNode', {node, id })
    this.nodesById[id] = node;
    return id;
  }

  hasNode(id: NodeId): boolean {
    return this.nodesById[id] != null;
  }

  getNode(id: NodeId): ?TNode {
    return this.nodesById[id];
  }

  addEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType = 1,
  ): boolean {
    this.inner.addEdge(from, to, type);
    return true;
  }

  hasEdge(
    from: NodeId,
    to: NodeId,
    type?: TEdgeType | NullEdgeType,
    //  | Array<TEdgeType | NullEdgeType> = 1,
  ): boolean {
    return this.inner.hasEdge(
      from,
      to,
      Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
    );
  }

  getNodeIdsConnectedTo(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      // | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): Array<NodeId> {
    return this.inner.getNodeIdsConnectedTo(
      nodeId,
      Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
    );
  }

  getNodeIdsConnectedFrom(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      // | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): Array<NodeId> {
    return this.inner.getNodeIdsConnectedFrom(
      nodeId,
      Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
    );
  }

  // Removes node and any edges coming from or to that node
  removeNode(nodeId: NodeId) {
    this.inner.removeNode(nodeId);
    delete this.nodesById[nodeId];
  }

  removeEdges(nodeId: NodeId, type: TEdgeType | NullEdgeType = 1) {
    this.inner.removeEdges(
      nodeId,
      Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
    );
  }

  removeEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType = 1,
    // TODO: handle this?
    removeOrphans: boolean = true,
  ) {
    this.inner.removeEdge(
      from,
      to,

      Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
      removeOrphans,
    );
  }

  isOrphanedNode(nodeId: NodeId): boolean {
    return this.inner.isOrphanedNode(this.rootNodeId, nodeId);
  }

  updateNode(nodeId: NodeId, node: TNode): void {
    this._assertHasNodeId(nodeId);
    this.nodesById[nodeId] = node;
  }

  // Update a node's downstream nodes making sure to prune any orphaned branches
  replaceNodeIdsConnectedTo(
    fromNodeId: NodeId,
    toNodeIds: $ReadOnlyArray<NodeId>,
    replaceFilter?: null | (NodeId => boolean),
    type?: TEdgeType | NullEdgeType = 1,
  ): void {
    this._assertHasNodeId(fromNodeId);

    let outboundEdges = this.getNodeIdsConnectedFrom(fromNodeId, type);
    let childrenToRemove = new Set(
      replaceFilter
        ? outboundEdges.filter(toNodeId => replaceFilter(toNodeId))
        : outboundEdges,
    );
    for (let toNodeId of toNodeIds) {
      childrenToRemove.delete(toNodeId);

      if (!this.hasEdge(fromNodeId, toNodeId, type)) {
        this.addEdge(fromNodeId, toNodeId, type);
      }
    }

    for (let child of childrenToRemove) {
      this.removeEdge(fromNodeId, child, type);
    }
  }

  traverse<TContext>(
    visit: GraphVisitor<NodeId, TContext>,
    startNodeId: ?NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      // | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): ?TContext {
    const enter = typeof visit === 'function' ? visit : visit.enter;
    const traversalStartNode = nullthrows(
      startNodeId ?? this.rootNodeId,
      'A start node is required to traverse',
    );
    this._assertHasNodeId(traversalStartNode);

    if (enter) {
      this.inner.dfs(
        traversalStartNode,
        enter,
        typeof visit !== 'function' ? visit.exit ?? null : null,
        Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
        false,
      );
    }
  }

  filteredTraverse<TValue, TContext>(
    filter: (NodeId, TraversalActions) => ?TValue,
    visit: GraphVisitor<TValue, TContext>,
    startNodeId: ?NodeId,
    type?:
      | TEdgeType
      | NullEdgeType
      // | Array<TEdgeType>
      | AllEdgeTypes,
  ): ?TContext {
    return this.traverse(mapVisitor(filter, visit), startNodeId, type);
  }

  traverseAncestors<TContext>(
    startNodeId: ?NodeId,
    visit: GraphVisitor<NodeId, TContext>,
    type:
      | TEdgeType
      | NullEdgeType
      // | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): ?TContext {
    const traversalStartNode = nullthrows(
      startNodeId ?? this.rootNodeId,
      'A start node is required to traverse',
    );
    this._assertHasNodeId(traversalStartNode);
    const enter = typeof visit === 'function' ? visit : visit.enter;

    if (enter) {
      return this.inner.dfs(
        traversalStartNode,
        enter,
        typeof visit !== 'function' ? visit.exit ?? null : null,
        Array.isArray(type) ? type : type !== -1 && type != null ? [type] : [],
        true,
      );
    }
  }

  // A post-order implementation of dfsFast
  postOrderDfsFast(
    visit: GraphTraversalCallback<NodeId, TraversalActions>,
    startNodeId: ?NodeId,
  ): void {
    // TODO: implement in rust
  }

  dfs<TContext>({
    visit,
    startNodeId,
    getChildren,
  }: {|
    visit: GraphVisitor<NodeId, TContext>,
    getChildren(nodeId: NodeId): Array<NodeId>,
    startNodeId?: ?NodeId,
  |}): ?TContext {
    let enter = typeof visit === 'function' ? visit : visit.enter;

    if (enter && startNodeId != null) {
      return this.inner.dfs(
        startNodeId,
        enter,
        typeof visit !== 'function' ? visit.exit ?? null : null,
        [],
        false,
      );
    }
  }

  topoSort(): Array<NodeId> {
    // type?: TEdgeType
    return this.inner.topoSort();
  }

  findAncestor(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId {
    let res = null;
    this.traverseAncestors(nodeId, (nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res = nodeId;
        traversal.stop();
      }
    });
    return res;
  }

  findAncestors(
    nodeId: NodeId,
    fn: (nodeId: NodeId) => boolean,
  ): Array<NodeId> {
    let res = [];
    this.traverseAncestors(nodeId, (nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res.push(nodeId);
        traversal.skipChildren();
      }
    });
    return res;
  }

  findDescendant(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId {
    let res = null;
    this.traverse((nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res = nodeId;
        traversal.stop();
      }
    }, nodeId);
    return res;
  }

  findDescendants(
    nodeId: NodeId,
    fn: (nodeId: NodeId) => boolean,
  ): Array<NodeId> {
    let res = [];
    this.traverse((nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res.push(nodeId);
        traversal.skipChildren();
      }
    }, nodeId);
    return res;
  }

  _assertHasNodeId(nodeId: NodeId) {
    if (!this.hasNode(nodeId)) {
      throw new Error('Does not have node ' + fromNodeId(nodeId));
    }
  }
}

export class JSGraph<TNode, TEdgeType: number = 1> {
  nodes: Array<TNode | null>;
  adjacencyList: AdjacencyList<TEdgeType>;
  rootNodeId: ?NodeId;
  _visited: ?BitSet;

  constructor(opts: ?GraphOpts<TNode, TEdgeType>) {
    this.nodes = opts?.nodes || [];
    this.setRootNodeId(opts?.rootNodeId);

    let adjacencyList = opts?.adjacencyList;
    this.adjacencyList = adjacencyList
      ? AdjacencyList.deserialize(adjacencyList)
      : new AdjacencyList<TEdgeType>();
  }

  setRootNodeId(id: ?NodeId) {
    this.rootNodeId = id;
  }

  static deserialize(
    opts: GraphOpts<TNode, TEdgeType>,
  ): Graph<TNode, TEdgeType> {
    return new this({
      nodes: opts.nodes,
      adjacencyList: opts.adjacencyList,
      rootNodeId: opts.rootNodeId,
    });
  }

  serialize(): SerializedGraph<TNode, TEdgeType> {
    return {
      nodes: this.nodes,
      adjacencyList: this.adjacencyList.serialize(),
      rootNodeId: this.rootNodeId,
    };
  }

  // Returns an iterator of all edges in the graph. This can be large, so iterating
  // the complete list can be costly in large graphs. Used when merging graphs.
  getAllEdges(): Iterator<Edge<TEdgeType | NullEdgeType>> {
    return this.adjacencyList.getAllEdges();
  }

  addNode(node: TNode): NodeId {
    let id = this.adjacencyList.addNode();
    this.nodes.push(node);
    return id;
  }

  hasNode(id: NodeId): boolean {
    return this.nodes[id] != null;
  }

  getNode(id: NodeId): ?TNode {
    return this.nodes[id];
  }

  addEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType = 1,
  ): boolean {
    if (Number(type) === 0) {
      throw new Error(`Edge type "${type}" not allowed`);
    }

    if (this.getNode(from) == null) {
      throw new Error(`"from" node '${fromNodeId(from)}' not found`);
    }

    if (this.getNode(to) == null) {
      throw new Error(`"to" node '${fromNodeId(to)}' not found`);
    }

    return this.adjacencyList.addEdge(from, to, type);
  }

  hasEdge(
    from: NodeId,
    to: NodeId,
    type?: TEdgeType | NullEdgeType | Array<TEdgeType | NullEdgeType> = 1,
  ): boolean {
    return this.adjacencyList.hasEdge(from, to, type);
  }

  getNodeIdsConnectedTo(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): Array<NodeId> {
    this._assertHasNodeId(nodeId);

    return this.adjacencyList.getNodeIdsConnectedTo(nodeId, type);
  }

  getNodeIdsConnectedFrom(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): Array<NodeId> {
    this._assertHasNodeId(nodeId);

    return this.adjacencyList.getNodeIdsConnectedFrom(nodeId, type);
  }

  // Removes node and any edges coming from or to that node
  removeNode(nodeId: NodeId) {
    if (!this.hasNode(nodeId)) {
      return;
    }

    for (let {type, from} of this.adjacencyList.getInboundEdgesByType(nodeId)) {
      this._removeEdge(
        from,
        nodeId,
        type,
        // Do not allow orphans to be removed as this node could be one
        // and is already being removed.
        false,
      );
    }

    for (let {type, to} of this.adjacencyList.getOutboundEdgesByType(nodeId)) {
      this._removeEdge(nodeId, to, type);
    }

    this.nodes[nodeId] = null;
  }

  removeEdges(nodeId: NodeId, type: TEdgeType | NullEdgeType = 1) {
    if (!this.hasNode(nodeId)) {
      return;
    }

    for (let to of this.getNodeIdsConnectedFrom(nodeId, type)) {
      this._removeEdge(nodeId, to, type);
    }
  }

  removeEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType = 1,
    removeOrphans: boolean = true,
  ) {
    if (!this.adjacencyList.hasEdge(from, to, type)) {
      throw new Error(
        `Edge from ${fromNodeId(from)} to ${fromNodeId(to)} not found!`,
      );
    }

    this._removeEdge(from, to, type, removeOrphans);
  }

  // Removes edge and node the edge is to if the node is orphaned
  _removeEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType = 1,
    removeOrphans: boolean = true,
  ) {
    if (!this.adjacencyList.hasEdge(from, to, type)) {
      return;
    }

    this.adjacencyList.removeEdge(from, to, type);
    if (removeOrphans && this.isOrphanedNode(to)) {
      this.removeNode(to);
    }
  }

  isOrphanedNode(nodeId: NodeId): boolean {
    if (!this.hasNode(nodeId)) {
      return false;
    }

    if (this.rootNodeId == null) {
      // If the graph does not have a root, and there are inbound edges,
      // this node should not be considered orphaned.
      return !this.adjacencyList.hasInboundEdges(nodeId);
    }

    // Otherwise, attempt to traverse backwards to the root. If there is a path,
    // then this is not an orphaned node.
    let hasPathToRoot = false;
    // go back to traverseAncestors
    this.traverseAncestors(
      nodeId,
      (ancestorId, _, actions) => {
        if (ancestorId === this.rootNodeId) {
          hasPathToRoot = true;
          actions.stop();
        }
      },
      ALL_EDGE_TYPES,
    );

    if (hasPathToRoot) {
      return false;
    }

    return true;
  }

  updateNode(nodeId: NodeId, node: TNode): void {
    this._assertHasNodeId(nodeId);
    this.nodes[nodeId] = node;
  }

  // Update a node's downstream nodes making sure to prune any orphaned branches
  replaceNodeIdsConnectedTo(
    fromNodeId: NodeId,
    toNodeIds: $ReadOnlyArray<NodeId>,
    replaceFilter?: null | (NodeId => boolean),
    type?: TEdgeType | NullEdgeType = 1,
  ): void {
    this._assertHasNodeId(fromNodeId);

    let outboundEdges = this.getNodeIdsConnectedFrom(fromNodeId, type);
    let childrenToRemove = new Set(
      replaceFilter
        ? outboundEdges.filter(toNodeId => replaceFilter(toNodeId))
        : outboundEdges,
    );
    for (let toNodeId of toNodeIds) {
      childrenToRemove.delete(toNodeId);

      if (!this.hasEdge(fromNodeId, toNodeId, type)) {
        this.addEdge(fromNodeId, toNodeId, type);
      }
    }

    for (let child of childrenToRemove) {
      this._removeEdge(fromNodeId, child, type);
    }
  }

  traverse<TContext>(
    visit: GraphVisitor<NodeId, TContext>,
    startNodeId: ?NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): ?TContext {
    let enter = typeof visit === 'function' ? visit : visit.enter;
    if (
      type === ALL_EDGE_TYPES &&
      enter &&
      (typeof visit === 'function' || !visit.exit)
    ) {
      return this.dfsFast(enter, startNodeId);
    } else {
      return this.dfs({
        visit,
        startNodeId,
        getChildren: nodeId => this.getNodeIdsConnectedFrom(nodeId, type),
      });
    }
  }

  filteredTraverse<TValue, TContext>(
    filter: (NodeId, TraversalActions) => ?TValue,
    visit: GraphVisitor<TValue, TContext>,
    startNodeId: ?NodeId,
    type?: TEdgeType | Array<TEdgeType | NullEdgeType> | AllEdgeTypes,
  ): ?TContext {
    return this.traverse(mapVisitor(filter, visit), startNodeId, type);
  }

  traverseAncestors<TContext>(
    startNodeId: ?NodeId,
    visit: GraphVisitor<NodeId, TContext>,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes = 1,
  ): ?TContext {
    return this.dfs({
      visit,
      startNodeId,
      getChildren: nodeId => this.getNodeIdsConnectedTo(nodeId, type),
    });
  }

  dfsFast<TContext>(
    visit: GraphTraversalCallback<NodeId, TContext>,
    startNodeId: ?NodeId,
  ): ?TContext {
    let traversalStartNode = nullthrows(
      startNodeId ?? this.rootNodeId,
      'A start node is required to traverse',
    );
    this._assertHasNodeId(traversalStartNode);

    let visited;
    if (!this._visited || this._visited.capacity < this.nodes.length) {
      this._visited = new BitSet(this.nodes.length);
      visited = this._visited;
    } else {
      visited = this._visited;
      visited.clear();
    }
    // Take shared instance to avoid re-entrancy issues.
    this._visited = null;

    let stopped = false;
    let skipped = false;
    let actions: TraversalActions = {
      skipChildren() {
        skipped = true;
      },
      stop() {
        stopped = true;
      },
    };

    let queue = [{nodeId: traversalStartNode, context: null}];
    while (queue.length !== 0) {
      let {nodeId, context} = queue.pop();
      if (!this.hasNode(nodeId) || visited.has(nodeId)) continue;
      visited.add(nodeId);

      skipped = false;
      let newContext = visit(nodeId, context, actions);
      if (typeof newContext !== 'undefined') {
        // $FlowFixMe[reassign-const]
        context = newContext;
      }

      if (skipped) {
        continue;
      }

      if (stopped) {
        this._visited = visited;
        return context;
      }

      this.adjacencyList.forEachNodeIdConnectedFromReverse(nodeId, child => {
        if (!visited.has(child)) {
          queue.push({nodeId: child, context});
        }
        return false;
      });
    }

    this._visited = visited;
    return null;
  }

  // A post-order implementation of dfsFast
  postOrderDfsFast(
    visit: GraphTraversalCallback<NodeId, TraversalActions>,
    startNodeId: ?NodeId,
  ): void {
    let traversalStartNode = nullthrows(
      startNodeId ?? this.rootNodeId,
      'A start node is required to traverse',
    );
    this._assertHasNodeId(traversalStartNode);

    let visited;
    if (!this._visited || this._visited.capacity < this.nodes.length) {
      this._visited = new BitSet(this.nodes.length);
      visited = this._visited;
    } else {
      visited = this._visited;
      visited.clear();
    }
    this._visited = null;

    let stopped = false;
    let actions: TraversalActions = {
      stop() {
        stopped = true;
      },
      skipChildren() {
        throw new Error(
          'Calling skipChildren inside a post-order traversal is not allowed',
        );
      },
    };

    let queue = [traversalStartNode];
    while (queue.length !== 0) {
      let nodeId = queue[queue.length - 1];

      if (!visited.has(nodeId)) {
        visited.add(nodeId);

        this.adjacencyList.forEachNodeIdConnectedFromReverse(nodeId, child => {
          if (!visited.has(child)) {
            queue.push(child);
          }
          return false;
        });
      } else {
        queue.pop();
        visit(nodeId, null, actions);

        if (stopped) {
          this._visited = visited;
          return;
        }
      }
    }

    this._visited = visited;
    return;
  }

  dfs<TContext>({
    visit,
    startNodeId,
    getChildren,
  }: {|
    visit: GraphVisitor<NodeId, TContext>,
    getChildren(nodeId: NodeId): Array<NodeId>,
    startNodeId?: ?NodeId,
  |}): ?TContext {
    let traversalStartNode = nullthrows(
      startNodeId ?? this.rootNodeId,
      'A start node is required to traverse',
    );
    this._assertHasNodeId(traversalStartNode);

    let visited;
    if (!this._visited || this._visited.capacity < this.nodes.length) {
      this._visited = new BitSet(this.nodes.length);
      visited = this._visited;
    } else {
      visited = this._visited;
      visited.clear();
    }
    // Take shared instance to avoid re-entrancy issues.
    this._visited = null;

    let stopped = false;
    let skipped = false;
    let actions: TraversalActions = {
      skipChildren() {
        skipped = true;
      },
      stop() {
        stopped = true;
      },
    };

    let walk = (nodeId, context: ?TContext) => {
      if (!this.hasNode(nodeId)) return;
      visited.add(nodeId);

      skipped = false;
      let enter = typeof visit === 'function' ? visit : visit.enter;
      if (enter) {
        let newContext = enter(nodeId, context, actions);
        if (typeof newContext !== 'undefined') {
          // $FlowFixMe[reassign-const]
          context = newContext;
        }
      }

      if (skipped) {
        return;
      }

      if (stopped) {
        return context;
      }

      for (let child of getChildren(nodeId)) {
        if (visited.has(child)) {
          continue;
        }

        visited.add(child);
        let result = walk(child, context);
        if (stopped) {
          return result;
        }
      }

      if (
        typeof visit !== 'function' &&
        visit.exit &&
        // Make sure the graph still has the node: it may have been removed between enter and exit
        this.hasNode(nodeId)
      ) {
        let newContext = visit.exit(nodeId, context, actions);
        if (typeof newContext !== 'undefined') {
          // $FlowFixMe[reassign-const]
          context = newContext;
        }
      }

      if (skipped) {
        return;
      }

      if (stopped) {
        return context;
      }
    };

    let result = walk(traversalStartNode);
    this._visited = visited;
    return result;
  }

  bfs(visit: (nodeId: NodeId) => ?boolean): ?NodeId {
    let rootNodeId = nullthrows(
      this.rootNodeId,
      'A root node is required to traverse',
    );

    let queue: Array<NodeId> = [rootNodeId];
    let visited = new Set<NodeId>([rootNodeId]);

    while (queue.length > 0) {
      let node = queue.shift();
      let stop = visit(rootNodeId);
      if (stop === true) {
        return node;
      }

      for (let child of this.getNodeIdsConnectedFrom(node)) {
        if (!visited.has(child)) {
          visited.add(child);
          queue.push(child);
        }
      }
    }

    return null;
  }

  topoSort(type?: TEdgeType): Array<NodeId> {
    let sorted: Array<NodeId> = [];
    this.traverse(
      {
        exit: nodeId => {
          sorted.push(nodeId);
        },
      },
      null,
      type,
    );
    return sorted.reverse();
  }

  findAncestor(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId {
    let res = null;
    this.traverseAncestors(nodeId, (nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res = nodeId;
        traversal.stop();
      }
    });
    return res;
  }

  findAncestors(
    nodeId: NodeId,
    fn: (nodeId: NodeId) => boolean,
  ): Array<NodeId> {
    let res = [];
    this.traverseAncestors(nodeId, (nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res.push(nodeId);
        traversal.skipChildren();
      }
    });
    return res;
  }

  findDescendant(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId {
    let res = null;
    this.traverse((nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res = nodeId;
        traversal.stop();
      }
    }, nodeId);
    return res;
  }

  findDescendants(
    nodeId: NodeId,
    fn: (nodeId: NodeId) => boolean,
  ): Array<NodeId> {
    let res = [];
    this.traverse((nodeId, ctx, traversal) => {
      if (fn(nodeId)) {
        res.push(nodeId);
        traversal.skipChildren();
      }
    }, nodeId);
    return res;
  }

  _assertHasNodeId(nodeId: NodeId) {
    if (!this.hasNode(nodeId)) {
      throw new Error('Does not have node ' + fromNodeId(nodeId));
    }
  }
}

export default ParcelGraph;
// export default JSGraph;

export function mapVisitor<NodeId, TValue, TContext>(
  filter: (NodeId, TraversalActions) => ?TValue,
  visit: GraphVisitor<TValue, TContext>,
): GraphVisitor<NodeId, TContext> {
  function makeEnter(visit) {
    return function mappedEnter(nodeId, context, actions) {
      let value = filter(nodeId, actions);
      if (value != null) {
        return visit(value, context, actions);
      }
    };
  }

  if (typeof visit === 'function') {
    return makeEnter(visit);
  }

  let mapped = {};
  if (visit.enter != null) {
    mapped.enter = makeEnter(visit.enter);
  }

  if (visit.exit != null) {
    mapped.exit = function mappedExit(nodeId, context, actions) {
      let exit = visit.exit;
      if (!exit) {
        return;
      }

      let value = filter(nodeId, actions);
      if (value != null) {
        return exit(value, context, actions);
      }
    };
  }

  return mapped;
}
