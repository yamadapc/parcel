// @flow strict-local

import type {NodeId} from '../types';
import type {GraphVisitor, TraversalActions} from '@parcel/types';
import type {SerializedAdjacencyList} from '../AdjacencyList';

export type NullEdgeType = 1;

export type GraphOpts<TNode, TEdgeType: number = 1> = {|
  nodes?: Array<TNode | null>,
  adjacencyList?: SerializedAdjacencyList<TEdgeType>,
  rootNodeId?: ?NodeId,
|};

export type SerializedParcelGraph<TNode> = {|
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
 * Decorate a visitor so that it'll only visit nodes that match a predicate
 * function.
 */
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
