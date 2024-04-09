use std::collections::HashSet;
use std::rc::Rc;

use napi::bindgen_prelude::Buffer;
use napi::{Env, JsFunction, JsUnknown};
use napi_derive::napi;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{Dfs, EdgeRef, NodeRef, Reversed};
use petgraph::{Directed, Direction, Graph};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};

use dfs_actions::DFSActions;

mod dfs_actions;

type JSNodeIndex = u32;
type JSEdgeIndex = u32;
type NodeWeight = u32;
type EdgeWeight = u32;

#[napi(object)]
pub struct EdgeDescriptor {
  pub from: JSNodeIndex,
  pub to: JSNodeIndex,
  pub weight: EdgeWeight,
}

/// Internal graph used for Parcel bundle/asset/request tracking, wraps petgraph
/// Edges and nodes have number weights
#[napi]
pub struct ParcelGraphImpl {
  inner: Graph<NodeWeight, EdgeWeight, Directed, u32>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SerializedGraph {
  nodes: Vec<u32>,
  edges: Vec<(u32, u32, EdgeWeight)>,
}

// MARK: JavaScript API
#[napi]
impl ParcelGraphImpl {
  /// Create a new graph instance. This is currently a `petgraph` adjacency list graph.
  /// The graph is directed and has u32 weights for both nodes and edges.
  ///
  /// JavaScript owns the graph instance.
  #[napi(constructor)]
  pub fn new() -> Self {
    Self {
      inner: Graph::new(),
    }
  }

  /// Deserialize a graph from a buffer.
  #[napi(factory)]
  pub fn deserialize(serialized: Buffer) -> napi::Result<Self> {
    let mut output = Self::new();
    let value = serialized.as_ref();
    let archive: SerializedGraph =
      from_bytes(value).map_err(|_| napi::Error::from_reason("Failed to deserialize"))?;
    for node in archive.nodes.iter() {
      output.add_node(*node);
    }
    for edge in archive.edges.iter() {
      output.add_edge(edge.0, edge.1, edge.2)?;
    }
    Ok(output)
  }

  /// Serialize the graph to a buffer.
  #[napi]
  pub fn serialize(&self, _env: Env) -> napi::Result<Buffer> {
    let serialized = SerializedGraph {
      nodes: self
        .inner
        .raw_nodes()
        .iter()
        .map(|item| item.weight)
        .collect(),
      edges: self
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
    };
    let serialized =
      to_allocvec(&serialized).map_err(|_err| napi::Error::from_reason("Failed to serialize"))?;
    Ok(Buffer::from(serialized.as_ref()))
  }

  #[napi]
  pub fn add_node(&mut self, weight: NodeWeight) -> JSNodeIndex {
    self.inner.add_node(weight).index() as u32
  }

  #[napi]
  pub fn has_node(&self, node_index: JSNodeIndex) -> bool {
    self
      .inner
      .node_weight(NodeIndex::new(node_index as usize))
      .is_some()
  }

  #[napi]
  pub fn node_weight(&self, node_index: JSNodeIndex) -> Option<NodeWeight> {
    self
      .inner
      .node_weight(NodeIndex::new(node_index as usize))
      .cloned()
  }

  #[napi]
  pub fn remove_node(&mut self, node_index: JSNodeIndex) {
    self.inner.remove_node(NodeIndex::new(node_index as usize));
  }

  #[napi]
  pub fn add_edge(
    &mut self,
    js_from: JSNodeIndex,
    js_to: JSNodeIndex,
    weight: EdgeWeight,
  ) -> napi::Result<JSEdgeIndex> {
    let from = NodeIndex::new(js_from as usize);
    let to = NodeIndex::new(js_to as usize);
    if self.inner.node_weight(from).is_none() {
      return Err(napi::Error::from_reason(format!(
        "\"from\" node '{js_from}' not found"
      )));
    }
    if self.inner.node_weight(to).is_none() {
      return Err(napi::Error::from_reason(format!(
        "\"to\" node '{js_to}' not found"
      )));
    }

    let edge = self.inner.add_edge(from, to, weight).index() as u32;
    Ok(edge)
  }

  #[napi]
  pub fn has_edge(
    &self,
    from: JSNodeIndex,
    to: JSNodeIndex,
    maybe_weight: Vec<EdgeWeight>,
  ) -> bool {
    self
      .inner
      .edges_connecting(NodeIndex::new(from as usize), NodeIndex::new(to as usize))
      .any(|item| {
        if !maybe_weight.is_empty() {
          maybe_weight.contains(item.weight())
        } else {
          true
        }
      })
  }

  #[napi]
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

  #[napi]
  pub fn remove_edge(
    &mut self,
    from: JSNodeIndex,
    to: JSNodeIndex,
    maybe_weight: Vec<EdgeWeight>,
    _remove_orphans: bool,
  ) -> napi::Result<()> {
    if !self.has_edge(from, to, maybe_weight.clone()) {
      return Err(napi::Error::from_reason(format!(
        "Edge from {from} to {to} not found!"
      )));
    }

    let edges = self
      .inner
      .edges_connecting(NodeIndex::new(from as usize), NodeIndex::new(to as usize));
    let edges_to_remove: Vec<EdgeIndex> = edges
      .filter(|edge| {
        if !maybe_weight.is_empty() {
          maybe_weight.contains(edge.weight())
        } else {
          true
        }
      })
      .map(|edge| edge.id())
      .collect();
    for edge_id in edges_to_remove {
      self.inner.remove_edge(edge_id);
    }

    Ok(())
  }

  #[napi]
  pub fn remove_edges(&mut self, node_index: JSNodeIndex, maybe_weight: Vec<EdgeWeight>) {
    let indexes_to_remove = self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Outgoing)
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

  #[napi]
  pub fn get_node_ids_connected_to(
    &self,
    node_index: JSNodeIndex,
    edge_weight: Vec<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Incoming)
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

  #[napi]
  pub fn get_node_ids_connected_from(
    &self,
    node_index: JSNodeIndex,
    edge_weight: Vec<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Outgoing)
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

  /// Will return an empty vec if the graph has a cycle
  #[napi]
  pub fn topo_sort(&self) -> Vec<JSNodeIndex> {
    let result = petgraph::algo::toposort(&self.inner, None).unwrap_or_else(|_| vec![]);
    result.iter().map(|node| node.index() as u32).collect()
  }

  /// Custom DFS visitor for JS callbacks.
  #[napi]
  pub fn dfs(
    &self,
    env: Env,
    start_node: JSNodeIndex,
    enter: JsFunction,
    exit: Option<JsFunction>,
    maybe_edge_weight: Vec<EdgeWeight>,
    // If true we will traverse up from start-node to all its incoming connections
    traverse_up: bool,
  ) -> napi::Result<JsUnknown> {
    let exit = exit.map(Rc::new);
    let direction = if traverse_up {
      Direction::Incoming
    } else {
      Direction::Outgoing
    };

    enum QueueCommand {
      Item {
        idx: NodeIndex,
        context: Rc<JsUnknown>,
      },
      Exit {
        f: Rc<JsFunction>,
        idx: NodeIndex,
      },
    }

    let mut queue: Vec<QueueCommand> = vec![];
    let mut visited = HashSet::<NodeIndex>::new();
    let initial_context = env.get_undefined()?.into_unknown();

    queue.push(QueueCommand::Item {
      idx: NodeIndex::new(start_node as usize),
      context: Rc::new(initial_context),
    });

    let actions = DFSActions::new();
    let js_actions = actions.to_js(&env)?;

    while let Some(command) = queue.pop() {
      match command {
        QueueCommand::Item { idx, context } => {
          visited.insert(idx);
          let js_node_idx = env.create_int64(idx.index() as i64)?;

          actions.reset();

          // Visit
          let new_context =
            enter.call(None, &[&js_node_idx.into_unknown(), &context, &js_actions])?;

          if actions.is_skipped() {
            continue;
          }

          if actions.is_stopped() {
            return Ok(new_context);
          }
          let new_context = Rc::new(new_context);

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

            if visited.contains(&child.target().id()) {
              continue;
            }
            queue.push(QueueCommand::Item {
              idx: child.target().id(),
              context: new_context.clone(),
            })
          }
        }
        QueueCommand::Exit { f, idx } => {
          let js_node_idx = env.create_int64(idx.index() as i64)?;
          f.call(None, &[js_node_idx])?;
        }
      }
    }

    Ok(env.get_undefined()?.into_unknown())
  }

  #[napi]
  pub fn get_unreachable_nodes(&self, root_index: JSNodeIndex) -> Vec<JSNodeIndex> {
    let root_index = NodeIndex::new(root_index as usize);
    get_unreachable_nodes(&self.inner, root_index)
      .into_iter()
      .map(|node| node.index() as u32)
      .collect()
  }

  #[napi]
  pub fn is_orphaned_node(&self, root_index: JSNodeIndex, node_index: JSNodeIndex) -> bool {
    let root_index = NodeIndex::new(root_index as usize);
    let node_index = NodeIndex::new(node_index as usize);

    is_orphaned_node(&self.inner, root_index, node_index)
  }
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
  let reversed_graph = Reversed(&graph);
  let mut dfs = Dfs::new(reversed_graph, node_index);

  while let Some(node) = dfs.next(&reversed_graph) {
    if node == root_index {
      return false;
    }
  }

  true
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_is_orphaned_node() {
    let mut graph = ParcelGraphImpl::new();
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
    let mut graph = ParcelGraphImpl::new();
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
    let mut graph = ParcelGraphImpl::new();
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
    let mut graph = ParcelGraphImpl::new();
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
