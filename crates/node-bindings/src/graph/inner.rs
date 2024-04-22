use napi::{Env, JsFunction, JsUnknown};
use napi_derive::napi;
use petgraph::graph::{Edge, EdgeIndex, EdgeReference, NodeIndex};
use petgraph::prelude::{Dfs, DfsPostOrder, EdgeRef};
use petgraph::visit::{NodeRef, Reversed};
use petgraph::{Directed, Direction, Graph};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::rc::Rc;

use crate::graph::dfs_actions::DFSActions;
use crate::graph::{EdgeDescriptor, EdgeWeight, JSNodeIndex, NodeWeight};

struct RemovalPlan {
  /// This node is being removed
  cause: NodeIndex,
  /// These nodes are being removed because they are orphaned after this change
  orphans: Vec<NodeIndex>,
}

pub struct OrphanedNodes(Vec<NodeIndex>);

pub struct GraphInner {
  inner: Graph<NodeWeight, EdgeWeight, Directed, u32>,
}

impl GraphInner {
  pub fn new() -> Self {
    GraphInner {
      inner: Graph::new(),
    }
  }

  pub fn add_node(&mut self) -> NodeIndex {
    self.inner.add_node(0)
  }

  pub fn node_weight(&self, node: NodeIndex) -> Option<&NodeWeight> {
    self.inner.node_weight(node)
  }

  pub fn node_count(&self) -> usize {
    self.inner.node_count()
  }

  pub fn add_edge(
    &mut self,
    source: NodeIndex,
    target: NodeIndex,
    weight: EdgeWeight,
  ) -> EdgeIndex {
    self.inner.add_edge(source, target, weight)
  }

  /// Return true if a node exists in the Graph
  ///
  /// O(1)
  pub fn has_node(&self, node_index: NodeIndex) -> bool {
    self.inner.node_weight(node_index).is_some()
  }

  pub fn remove_node(&mut self, node: NodeIndex, root_node: NodeIndex) -> OrphanedNodes {
    let get_edges = |inner: &Graph<NodeWeight, EdgeWeight>, direction| {
      inner
        .edges_directed(node, direction)
        .map(|edge| (edge.id(), edge.target()))
        .collect::<Vec<(EdgeIndex, NodeIndex)>>()
    };

    let outgoing_edges = get_edges(&self.inner, Direction::Outgoing);
    let rewritten_node_index = self.inner.node_count() - 1;
    self.inner.remove_node(node);

    for (_, target) in outgoing_edges {
      let target = if target.index() == rewritten_node_index {
        node
      } else {
        target
      };

      if is_orphaned_node(&self.inner, root_node, target) {
        self.remove_node(target, root_node);
      }
    }

    OrphanedNodes(vec![])
  }

  pub fn has_edge(&self, from: NodeIndex, to: NodeIndex, maybe_weight: Vec<EdgeWeight>) -> bool {
    let weight_is_empty = maybe_weight.is_empty();
    self.inner.edges_connecting(from, to).any(|item| {
      if !weight_is_empty {
        maybe_weight.contains(item.weight())
      } else {
        true
      }
    })
  }

  pub fn raw_edges(&self) -> &[Edge<EdgeWeight>] {
    self.inner.raw_edges()
  }

  pub fn get_all_edges(&self) -> Vec<EdgeDescriptor> {
    let edges = self.inner.raw_edges();
    edges
      .iter()
      .map(|edge| EdgeDescriptor {
        from: edge.source().index() as u32,
        to: edge.target().index() as u32,
        weight: edge.weight,
      })
      .collect()
  }

  // todo
  pub fn remove_edge(&mut self, edge: EdgeIndex) {
    self.inner.remove_edge(edge);
  }

  pub fn remove_edges(&mut self, node_index: NodeIndex, maybe_weight: Vec<EdgeWeight>) {
    let indexes_to_remove = self
      .inner
      .edges_directed(node_index, Direction::Outgoing)
      .filter(|edge| {
        if maybe_weight.is_empty() {
          maybe_weight.contains(edge.weight())
        } else {
          true
        }
      })
      .map(|edge| edge.id())
      .collect::<Vec<EdgeIndex>>();

    for idx in indexes_to_remove {
      self.inner.remove_edge(idx);
    }
  }

  pub fn get_node_ids_connected_to(
    &self,
    node_index: NodeIndex,
    edge_weight: Vec<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(node_index, Direction::Incoming)
      .filter(|edge| {
        if !edge_weight.is_empty() {
          edge_weight.contains(edge.weight())
        } else {
          true
        }
      })
      .map(|edge| edge.source().index() as u32)
      .collect()
  }

  pub fn get_node_ids_connected_from(
    &self,
    node_index: NodeIndex,
    edge_weight: Vec<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(node_index, Direction::Outgoing)
      .filter(|edge| {
        if !edge_weight.is_empty() {
          edge_weight.contains(edge.weight())
        } else {
          true
        }
      })
      .map(|edge| edge.target().index() as u32)
      .collect()
  }

  pub fn topo_sort(&self) -> Vec<NodeIndex> {
    petgraph::algo::toposort(&self.inner, None).unwrap_or_else(|_| vec![])
  }

  pub fn get_unreachable_nodes(&self, root_index: NodeIndex) -> Vec<NodeIndex> {
    get_unreachable_nodes(&self.inner, root_index)
  }

  pub fn post_order_dfs<E>(
    &self,
    start_node: NodeIndex,
    visit: impl Fn(NodeIndex) -> Result<(), E>,
  ) -> Result<(), E> {
    let mut dfs = DfsPostOrder::new(&self.inner, start_node);

    while let Some(node) = dfs.next(&self.inner) {
      visit(node)?;
    }

    Ok(())
  }

  pub fn dfs<R, ExitFn, E>(
    &self,
    start_node: NodeIndex,
    enter: impl Fn(NodeIndex) -> Result<R, E>,
    exit: Option<ExitFn>,
    maybe_edge_weight: Vec<EdgeWeight>,
    traverse_up: bool,
  ) -> Result<Option<R>, E>
  where
    ExitFn: Fn(NodeIndex) -> Result<R, E>,
  {
    let exit = exit.map(Rc::new);
    let direction = if traverse_up {
      Direction::Incoming
    } else {
      Direction::Outgoing
    };

    enum QueueCommand<R, ExitFn> {
      Item {
        idx: NodeIndex,
        context: Rc<Option<R>>,
      },
      Exit {
        f: Rc<ExitFn>,
        idx: NodeIndex,
      },
    }

    let mut queue: Vec<QueueCommand<R, ExitFn>> = vec![];
    let mut visited = HashSet::<NodeIndex>::new();
    let initial_context: Option<R> = None;

    queue.push(QueueCommand::Item {
      idx: start_node,
      context: Rc::new(initial_context),
    });

    let actions = DFSActions::new();
    // let js_actions = actions.to_js(&env)?;

    while let Some(command) = queue.pop() {
      match command {
        QueueCommand::Item { idx, context } => {
          if visited.contains(&idx) {
            continue;
          }

          visited.insert(idx);
          // let js_node_idx = env.create_int64(idx.index() as i64)?;

          actions.reset();

          // Visit
          let new_context = enter(idx)?;
          // enter.call(None, &[&js_node_idx.into_unknown(), &context, &js_actions])?;

          if actions.is_skipped() {
            continue;
          }

          if actions.is_stopped() {
            return Ok(Some(new_context));
          }
          let new_context = Rc::new(Some(new_context));

          if let Some(exit) = &exit {
            queue.push(QueueCommand::Exit {
              f: exit.clone(),
              idx,
            });
          }
          for child in self.inner.edges_directed(idx, direction) {
            let matches_target_weight =
              maybe_edge_weight.is_empty() || maybe_edge_weight.contains(child.weight());
            if !matches_target_weight {
              continue;
            }

            let other_id = if traverse_up {
              child.source().id()
            } else {
              child.target().id()
            };
            if visited.contains(&other_id) {
              continue;
            }

            queue.push(QueueCommand::Item {
              idx: other_id,
              context: new_context.clone(),
            })
          }
        }
        QueueCommand::Exit { f, idx } => {
          // let js_node_idx = env.create_int64(idx.index() as i64)?;
          // f.call(None, &[js_node_idx])?;
          f(idx)?;
        }
      }
    }

    Ok(None)
  }

  pub fn is_orphaned_node(&self, root_index: NodeIndex, node_index: NodeIndex) -> bool {
    is_orphaned_node(&self.inner, root_index, node_index)
  }

  // pub fn remove_node(&mut self, node: NodeIndex, root_node: NodeIndex) -> OrphanedNodes {
  //   let get_edges = |inner: &Graph<NodeWeight, EdgeWeight>, direction| {
  //     inner
  //       .edges_directed(node, direction)
  //       .map(|edge| (edge.id(), edge.target()))
  //       .collect::<Vec<(EdgeIndex, NodeIndex)>>()
  //   };
  //
  //   let outgoing_edges = get_edges(&self.inner, Direction::Outgoing);
  //   let rewritten_node_index = self.inner.node_count() - 1;
  //   self.inner.remove_node(node);
  //
  //   for (_, target) in outgoing_edges {
  //     let target = if target.index() == rewritten_node_index {
  //       node
  //     } else {
  //       target
  //     };
  //
  //     // TODO: We don't want to do this here as it's the wrong side-effect and messes-up complexity
  //     let js_target = target.index() as u32;
  //     if is_orphaned_node(&self.inner, root_node, target) {
  //       self.remove_node(env, js_target, js_root_node)?;
  //     }
  //   }
  //
  //   OrphanedNodes(vec![])
  // }
}

/// Given a graph and root index, get the list of nodes that are disconnected from the
/// root and unreachable.
///
/// O(n) with respect to the number of nodes in the graph.
fn get_unreachable_nodes<N, E>(graph: &Graph<N, E>, root: NodeIndex) -> Vec<NodeIndex> {
  let mut dfs = Dfs::new(&graph, root);
  let mut reachable_nodes = HashSet::new();

  while let Some(node) = dfs.next(&graph) {
    reachable_nodes.insert(node);
  }

  graph
    .node_indices()
    .collect::<Vec<NodeIndex>>()
    .into_iter()
    .filter(|node| !reachable_nodes.contains(node))
    .collect()
}

/// Returns true if a node is 'orphaned' and unreachable from the root node.
///
/// ATTENTION: This traverses the graph and is worst case O(n) with respect to
/// the number of nodes. When possible, prefer to use "get_unreachable_nodes" to
/// get all the unreachable nodes at once.
fn is_orphaned_node<N, E>(
  graph: &Graph<N, E>,
  root_index: NodeIndex,
  node_index: NodeIndex,
) -> bool {
  if graph.node_weight(node_index).is_none() {
    return true;
  }
  let reversed_graph = Reversed(&graph);
  let mut dfs = Dfs::new(reversed_graph, node_index);

  while let Some(node) = dfs.next(&reversed_graph) {
    if node == root_index {
      return false;
    }
  }

  true
}

/// The graph is serialized to a list of nodes and a list of edges.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct SerializedGraph {
  nodes: Vec<u32>,
  edges: Vec<(u32, u32, EdgeWeight)>,
}

impl From<&GraphInner> for SerializedGraph {
  fn from(value: &GraphInner) -> Self {
    SerializedGraph {
      nodes: value
        .inner
        .raw_nodes()
        .iter()
        .map(|item| item.weight)
        .collect(),
      edges: value
        .inner
        .raw_edges()
        .iter()
        .map(|item| {
          (
            item.source().index() as u32,
            item.target().index() as u32,
            item.weight,
          )
        })
        .collect(),
    }
  }
}

impl GraphInner {
  pub fn from_serialized(value: &SerializedGraph) -> napi::Result<Self> {
    let mut output: GraphInner = Self::new();
    for node in value.nodes.iter() {
      output.inner.add_node(*node);
    }
    for edge in value.edges.iter() {
      output.inner.add_edge(edge.0.into(), edge.1.into(), edge.2);
    }
    Ok(output)
  }
}

#[cfg(test)]
mod test {
  use super::*;

  /// Assert two graphs are equal.
  fn assert_graph_is_equal(graph: &GraphInner, other: &GraphInner) {
    assert_eq!(graph.node_count(), other.node_count());
    assert_eq!(
      graph
        .inner
        .raw_nodes()
        .iter()
        .map(|node| node.weight)
        .collect::<Vec<NodeWeight>>(),
      other
        .inner
        .raw_nodes()
        .iter()
        .map(|node| node.weight)
        .collect::<Vec<NodeWeight>>()
    );
    assert_eq!(
      graph
        .inner
        .raw_edges()
        .iter()
        .map(|node| node.weight)
        .collect::<Vec<NodeWeight>>(),
      other
        .inner
        .raw_edges()
        .iter()
        .map(|node| node.weight)
        .collect::<Vec<NodeWeight>>()
    );
  }

  #[test]
  fn test_serialize_graph() {
    let mut graph: GraphInner = GraphInner::new();
    let root = graph.inner.add_node(0);
    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);
    let idx3 = graph.inner.add_node(0);
    graph.remove_node(idx3, root);

    graph.inner.add_edge(root, idx1, 0);
    graph.inner.add_edge(idx1, idx2, 0);

    let serialized = SerializedGraph::from(&graph);
    let deserialized = GraphInner::from_serialized(&serialized).unwrap();
    assert_graph_is_equal(&graph, &deserialized);
  }

  #[test]
  fn test_add_node() {
    let mut graph = GraphInner::new();
    let idx1 = graph.add_node();
    let idx2 = graph.add_node();
    let idx3 = graph.add_node();
    assert_eq!(graph.node_count(), 3);
    assert_eq!(idx1.index(), 0);
    assert_eq!(idx2.index(), 1);
    assert_eq!(idx3.index(), 2);
  }

  #[test]
  fn test_has_node() {
    let mut graph = GraphInner::new();
    let idx1 = graph.add_node();
    let idx2 = graph.add_node();
    assert!(graph.has_node(idx1));
    assert!(graph.has_node(idx2));
    assert!(!graph.has_node(3.into()));
  }

  #[test]
  fn test_node_weight() {
    let mut graph = GraphInner::new();
    let idx1 = graph.add_node();
    let idx2 = graph.add_node();
    assert_eq!(graph.node_weight(idx1), Some(42).as_ref());
    assert_eq!(graph.node_weight(idx2), Some(43).as_ref());
    assert_eq!(graph.node_weight(3.into()), None);
  }

  #[test]
  fn test_remove_node() {
    let mut graph = GraphInner::new();
    let idx1 = graph.add_node();
    let idx2 = graph.add_node();
    let idx3 = graph.add_node();
    assert_eq!(graph.node_count(), 3);
    graph.remove_node(idx2, idx1);
    assert_eq!(graph.node_count(), 2);
    assert!(graph.has_node(idx1));
    assert!(!graph.has_node(idx2));
    assert!(graph.has_node(idx3));
  }

  // #[test]
  // fn test_remove_edge_makes_edge_none() {
  //   let mut graph = GraphInner::new();
  //   let idx1 = graph.add_node();
  //   let idx2 = graph.add_node();
  //   let _edge = graph.add_edge(idx1, idx2, 0);
  //   assert!(graph.has_edge(idx1, idx2, vec![]));
  //   graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
  //   assert!(!graph.has_edge(idx1, idx2, vec![]));
  // }

  // #[test]
  // fn test_removed_edge_is_not_returned_anymore() {
  //   let mut graph = GraphInner::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let _edge = graph.add_edge(idx1, idx2, 0).unwrap();
  //   graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
  //   assert!(!graph.has_edge(idx1, idx2, vec![]));
  //   assert_eq!(graph.get_all_edges(), vec![]);
  // }

  #[test]
  fn test_remove_edge_should_prune_graph_at_that_edge() {
    let mut graph = GraphInner::new();
    let root = graph.add_node(0);
    let idx2 = graph.add_node(0);
    let idx3 = graph.add_node(0);
    let idx4 = graph.add_node(0);
    graph.add_edge(root, idx2, 0).unwrap();
    graph.add_edge(root, idx4, 0).unwrap();
    graph.add_edge(idx2, idx3, 0).unwrap();
    graph.add_edge(idx2, idx4, 0).unwrap();
    graph.remove_edge(root, idx2, vec![], true, root).unwrap();
    assert!(!graph.has_edge(idx3, idx4, vec![]));
    assert!(graph.has_node(root));
    assert!(!graph.has_node(idx2));
    assert!(!graph.has_node(idx3));
    assert!(graph.has_node(idx4));
    assert_eq!(
      graph.get_all_edges(),
      vec![EdgeDescriptor {
        from: root,
        to: idx4,
        weight: 0
      }]
    )
  }

  #[test]
  fn test_node_count_increases_when_node_is_added() {
    let mut graph = GraphInner::new();
    assert_eq!(graph.node_count(), 0);
    graph.add_node(0);
    assert_eq!(graph.node_count(), 1);
  }

  #[test]
  fn test_node_count_decreases_when_node_is_removed() {
    let mut graph = GraphInner::new();
    let idx = graph.add_node(0);
    assert_eq!(graph.node_count(), 1);
    graph.remove_node(idx, idx);
    assert_eq!(graph.node_count(), 0);
  }

  #[test]
  fn test_is_orphaned_node() {
    let mut graph = GraphInner::new();
    let root = graph.inner.add_node(0);

    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);
    let idx3 = graph.inner.add_node(0);

    assert!(is_orphaned_node(&graph.inner, root, idx1));
    assert!(is_orphaned_node(&graph.inner, root, idx2));
    assert!(is_orphaned_node(&graph.inner, root, idx3));

    graph.inner.add_edge(root, idx1, 0);
    assert!(!is_orphaned_node(&graph.inner, root, idx1));
    assert!(is_orphaned_node(&graph.inner, root, idx2));
    assert!(is_orphaned_node(&graph.inner, root, idx3));

    graph.inner.add_edge(idx2, idx3, 0);
    assert!(!is_orphaned_node(&graph.inner, root, idx1));
    assert!(is_orphaned_node(&graph.inner, root, idx2));
    assert!(is_orphaned_node(&graph.inner, root, idx3));

    graph.inner.add_edge(idx1, idx2, 0);
    assert!(!is_orphaned_node(&graph.inner, root, idx1));
    assert!(!is_orphaned_node(&graph.inner, root, idx2));
    assert!(!is_orphaned_node(&graph.inner, root, idx3));
  }

  #[test]
  fn test_get_unreachable_nodes_on_disconnected_graph() {
    let mut graph = GraphInner::new();
    let root = graph.inner.add_node(0);

    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);
    let idx3 = graph.inner.add_node(0);

    let unreachable = get_unreachable_nodes(&graph.inner, root);
    assert_eq!(unreachable.len(), 3);
    assert_eq!(unreachable, vec![idx1, idx2, idx3]);
  }

  #[test]
  fn test_get_unreachable_nodes_with_direct_root_connection() {
    let mut graph = GraphInner::new();
    let root = graph.inner.add_node(0);

    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);

    graph.inner.add_edge(root, idx1, 0);

    let unreachable = get_unreachable_nodes(&graph.inner, root);
    assert_eq!(unreachable.len(), 1);
    assert_eq!(unreachable, vec![idx2]);
  }

  #[test]
  fn test_get_unreachable_nodes_with_indirect_root_connection() {
    let mut graph = GraphInner::new();
    let root = graph.inner.add_node(0);

    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);
    let idx3 = graph.inner.add_node(0);

    graph.inner.add_edge(root, idx1, 0);
    graph.inner.add_edge(idx1, idx2, 0);

    let unreachable = get_unreachable_nodes(&graph.inner, root);
    assert_eq!(unreachable.len(), 1);
    assert_eq!(unreachable, vec![idx3]);

    graph.inner.add_edge(idx2, idx3, 0);
    let unreachable = get_unreachable_nodes(&graph.inner, root);
    assert_eq!(unreachable.len(), 0);
    assert_eq!(unreachable, vec![]);
  }
}
