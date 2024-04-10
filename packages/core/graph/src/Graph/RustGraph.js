// @flow strict-local

import {ParcelGraphImpl} from '@parcel/rust';
import type {Edge, NodeId} from '../types';
import {fromNodeId} from '../types';
import {BitSet} from '../BitSet';
import type {
  AllEdgeTypes,
  GraphOpts,
  NullEdgeType,
  SerializedParcelGraph,
} from './common';
import {mapVisitor} from './common';
import type {
  GraphTraversalCallback,
  GraphVisitor,
  TraversalActions,
} from '@parcel/types';
import nullthrows from 'nullthrows';

/**
 * Rust backed graph structure
 */
export class RustGraph<TNode, TEdgeType: number = 1> {
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

    this.setRootNodeId(opts?.rootNodeId);
  }

  /**
   * Entries iterator for backwards compatibility.
   *
   * Ideally this should be removed and the consumers should use higher-level
   * APIs rather than listing nodes directly.
   */
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

  /**
   * Set the root node id for the graph. This will be used as the traversal
   * starting point and is used to determine disconnected nodes.
   */
  setRootNodeId(id: ?NodeId) {
    this.rootNodeId = id;
  }

  static deserialize(
    opts: GraphOpts<TNode, TEdgeType>,
  ): RustGraph<TNode, TEdgeType> {
    return new this({
      nodesById: opts.nodesById,
      graph: opts.graph,
      // adjacencyList: opts.adjacencyList,
      rootNodeId: opts.rootNodeId,
    });
  }

  serialize(): SerializedParcelGraph<TNode, TEdgeType> {
    return {
      // nodes: this.nodes,
      graph: this.inner.serialize(),
      nodesById: this.nodesById,
      rootNodeId: this.rootNodeId,
    };
  }

  // Returns an iterator of all edges in the graph. This can be large, so iterating
  // the complete list can be costly in large graphs. Used when merging graphs.
  getAllEdges(): Iterator<Edge<TEdgeType | NullEdgeType>> {
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

  // TODO: do not call this on removal as it is slow; move to rust
  cleanUp() {
    const nodes = this.inner.getUnreachableNodes(this.rootNodeId);
    nodes.forEach(nodeId => {
      this.removeNode(nodeId);
    });
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
    this.inner.postOrderDfs(startNodeId, visit);
  }

  dfs<TContext>({
    visit,
    startNodeId,
    // TODO: getChildren is not handled in rust
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
