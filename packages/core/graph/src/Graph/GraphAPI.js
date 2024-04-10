// @flow strict-local

import type {AllEdgeTypes, NullEdgeType} from './common';
import type {Edge, NodeId} from '../types';
import type {
  GraphTraversalCallback,
  GraphVisitor,
  TraversalActions,
} from '@parcel/types';

// constructor(opts: ?GraphOpts<TNode, TEdgeType>): this;
export interface GraphAPI<TNode, TEdgeType: number> {
  setRootNodeId(id: ?NodeId): void;

  getAllEdges(): Iterator<Edge<TEdgeType | NullEdgeType>>;

  addNode(node: TNode): NodeId;

  hasNode(id: NodeId): boolean;

  getNode(id: NodeId): ?TNode;

  addEdge(from: NodeId, to: NodeId, type: TEdgeType | NullEdgeType): boolean;

  hasEdge(
    from: NodeId,
    to: NodeId,
    type?: TEdgeType | NullEdgeType | Array<TEdgeType | NullEdgeType>,
  ): boolean;

  getNodeIdsConnectedTo(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes,
  ): Array<NodeId>;

  getNodeIdsConnectedFrom(
    nodeId: NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes,
  ): Array<NodeId>;

  removeNode(nodeId: NodeId): void;

  removeEdges(nodeId: NodeId, type: TEdgeType | NullEdgeType): void;

  removeEdge(
    from: NodeId,
    to: NodeId,
    type: TEdgeType | NullEdgeType,
    removeOrphans: boolean,
  ): void;

  isOrphanedNode(nodeId: NodeId): boolean;

  updateNode(nodeId: NodeId, node: TNode): void;

  replaceNodeIdsConnectedTo(
    fromNodeId: NodeId,
    toNodeIds: $ReadOnlyArray<NodeId>,
    replaceFilter?: null | (NodeId => boolean),
    type?: TEdgeType | NullEdgeType,
  ): void;

  traverse<TContext>(
    visit: GraphVisitor<NodeId, TContext>,
    startNodeId: ?NodeId,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes,
  ): ?TContext;

  filteredTraverse<TValue, TContext>(
    filter: (NodeId, TraversalActions) => ?TValue,
    visit: GraphVisitor<TValue, TContext>,
    startNodeId: ?NodeId,
    type?: TEdgeType | Array<TEdgeType | NullEdgeType> | AllEdgeTypes,
  ): ?TContext;

  traverseAncestors<TContext>(
    startNodeId: ?NodeId,
    visit: GraphVisitor<NodeId, TContext>,
    type:
      | TEdgeType
      | NullEdgeType
      | Array<TEdgeType | NullEdgeType>
      | AllEdgeTypes,
  ): ?TContext;

  dfsFast<TContext>(
    visit: GraphTraversalCallback<NodeId, TContext>,
    startNodeId: ?NodeId,
  ): ?TContext;

  postOrderDfsFast(
    visit: GraphTraversalCallback<NodeId, TraversalActions>,
    startNodeId: ?NodeId,
  ): void;

  dfs<TContext>(dfsParams: {|
    visit: GraphVisitor<NodeId, TContext>,
    getChildren(nodeId: NodeId): Array<NodeId>,
    startNodeId?: ?NodeId,
  |}): ?TContext;

  bfs(visit: (nodeId: NodeId) => ?boolean): ?NodeId;

  topoSort(type?: TEdgeType): Array<NodeId>;

  findAncestor(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId;

  findAncestors(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): Array<NodeId>;

  findDescendant(nodeId: NodeId, fn: (nodeId: NodeId) => boolean): ?NodeId;

  findDescendants(
    nodeId: NodeId,
    fn: (nodeId: NodeId) => boolean,
  ): Array<NodeId>;
}
