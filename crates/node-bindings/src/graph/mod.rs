use std::collections::HashSet;
use std::rc::Rc;

use napi::bindgen_prelude::Buffer;
use napi::{Env, JsFunction, JsUnknown};
use napi_derive::napi;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{Dfs, DfsPostOrder, EdgeRef, NodeRef, Reversed};
use petgraph::{Directed, Direction, Graph};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};

use dfs_actions::DFSActions;

mod dfs_actions;

/// All node indexes are represented by a u32. This allows the least expensive translation with JS.
type JSNodeIndex = u32;
/// All edge indexes are represented by a u32. This allows the least expensive translation with JS.
type JSEdgeIndex = u32;
/// All node weights are a u32. This is not necessary and unused. We could use a unit type here.
type NodeWeight = u32;
/// All edge weights are a u32. This is not necessary and unused. We could use a unit type here.
type EdgeWeight = u32;

/// Edge object when JavaScript lists edges.
/// This incurs copying so it's not optimal. Ideally the nº of listed edges is not very sensitive.
/// Ideally we will take the code that lists edges and move it across so we return less data.
#[napi(object)]
#[derive(Debug, PartialEq)]
pub struct EdgeDescriptor {
  pub from: JSNodeIndex,
  pub to: JSNodeIndex,
  pub weight: EdgeWeight,
}

/// Internal graph used for Parcel bundle/asset/request tracking, wraps petgraph
/// Edges and nodes have number weights
#[napi]
#[derive(Debug, Clone)]
pub struct ParcelGraphImpl {
  inner: Graph<NodeWeight, EdgeWeight, Directed, u32>,
  /// See `remove_node`
  removed_nodes: HashSet<NodeIndex>,
}

// MARK: JavaScript API
#[napi]
impl ParcelGraphImpl {
  /// Create a new graph instance. This is currently a `petgraph` adjacency list graph.
  /// The graph is directed and has u32 weights for both nodes and edges.
  ///
  /// JavaScript owns the graph instance.
  ///
  /// NOTE: Not using `napi(constructor)` because that breaks RustRover
  /// https://youtrack.jetbrains.com/issue/RUST-11565
  #[napi]
  pub fn new() -> Self {
    Self {
      inner: Graph::new(),
      removed_nodes: HashSet::new(),
    }
  }

  /// Deserialize a graph from a buffer.
  ///
  /// NOTE: Not using `napi(factory)` because that breaks RustRover
  /// https://youtrack.jetbrains.com/issue/RUST-11565
  #[napi]
  pub fn deserialize(serialized: Buffer) -> napi::Result<Self> {
    let value = serialized.as_ref();
    let serialized_graph: SerializedGraph =
      from_bytes(value).map_err(|_| napi::Error::from_reason("Failed to deserialize"))?;
    Ok(Self::from(&serialized_graph))
  }

  /// Serialize the graph to a buffer. This copies the Graph data into a `SerializedGraph`,
  /// but the buffer is not copied into JavaScript. So we can optimise this quite a bit further.
  #[napi]
  pub fn serialize(&self, _env: Env) -> napi::Result<Buffer> {
    let serialized = SerializedGraph::from(self);
    let serialized =
      to_allocvec(&serialized).map_err(|_err| napi::Error::from_reason("Failed to serialize"))?;
    Ok(Buffer::from(serialized.as_ref()))
  }

  /// Add a node and return its index
  ///
  /// O(1) amortized ; but might resize internal vectors
  #[napi]
  pub fn add_node(&mut self, weight: NodeWeight) -> JSNodeIndex {
    self.inner.add_node(weight).index() as u32
  }

  /// Return true if a node exists in the Graph
  ///
  /// O(1)
  #[napi]
  pub fn has_node(&self, node_index: JSNodeIndex) -> bool {
    let node_index = NodeIndex::new(node_index as usize);
    if self.removed_nodes.contains(&node_index) {
      return false;
    }

    self.inner.node_weight(node_index).is_some()
  }

  /// Query the weight of a node.
  ///
  /// O(1)
  #[napi]
  pub fn node_weight(&self, node_index: JSNodeIndex) -> Option<NodeWeight> {
    if self
      .removed_nodes
      .contains(&NodeIndex::new(node_index as usize))
    {
      return None;
    }

    self
      .inner
      .node_weight(NodeIndex::new(node_index as usize))
      .cloned()
  }

  /// Mark node as removed.
  /// petgraph removal will invalidate the last node index since it will be moved.
  ///
  /// Ideally we would remove the node and fix the JS side to make sure it's okay with
  /// indexes changing. This is much better as otherwise the Graph never shrinks.
  #[napi]
  pub fn remove_node(&mut self, js_node_index: JSNodeIndex, js_root_node: JSNodeIndex) {
    // petgraph node removal will invalidate the last node index since it will be moved
    // to the removed node index.
    // Because of this, we will not remove the node, but instead mark it as removed.
    // self.inner.remove_node(NodeIndex::new(node_index as usize));
    self.removed_nodes.insert(js_node_index.into());

    let get_edges = |inner: &Graph<NodeWeight, EdgeWeight>, direction| {
      inner
        .edges_directed(NodeIndex::new(js_node_index as usize), direction)
        .map(|edge| (edge.id(), edge.target()))
        .collect::<Vec<(EdgeIndex, NodeIndex)>>()
    };

    for (edge, _) in get_edges(&self.inner, Direction::Incoming) {
      self.inner.remove_edge(edge);
    }

    for (edge, target) in get_edges(&self.inner, Direction::Outgoing) {
      self.inner.remove_edge(edge);

      // Orphan clean-up
      // TODO: We don't want to do this here as it's the wrong side-effect and messes-up complexity
      let root_node = NodeIndex::new(js_root_node as usize);
      let js_target = target.index() as u32;
      if is_orphaned_node(&self.inner, root_node, target) {
        self.remove_node(js_target, js_root_node);
      }
    }
  }

  /// Count the number of nodes on the graph.
  #[napi]
  pub fn node_count(&self) -> u32 {
    self.inner.node_count() as u32 - self.removed_nodes.len() as u32
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
    if !self.has_node(js_from) {
      return Err(napi::Error::from_reason(format!(
        "\"from\" node '{js_from}' not found"
      )));
    }
    if !self.has_node(js_to) {
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
    if !self.has_node(from) || !self.has_node(to) {
      return false;
    }

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

  /// Remove an edge from the graph.
  ///
  /// Due to how petgraph is implemented, removal will relocate edges. This means that
  /// the last edge on the graph will change IDs after edges are removed.
  ///
  /// Also, clean-up unreachable nodes. Without the clean-up step this would be O(1)
  /// but due to the clean-up step this is worst case O(n).
  #[napi]
  pub fn remove_edge(
    &mut self,
    js_from: JSNodeIndex,
    js_to: JSNodeIndex,
    maybe_weight: Vec<EdgeWeight>,
    remove_orphans: bool,
    js_root_node: JSNodeIndex,
  ) -> napi::Result<()> {
    if !self.has_edge(js_from, js_to, maybe_weight.clone()) {
      return Err(napi::Error::from_reason(format!(
        "Edge from {js_from} to {js_to} not found!"
      )));
    }

    let from = NodeIndex::new(js_from as usize);
    let to = NodeIndex::new(js_to as usize);
    let root_node = NodeIndex::new(js_root_node as usize);

    let edges = self.inner.edges_connecting(from, to);
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

    // TODO: We don't want to do this here as it's the wrong side-effect and messes-up complexity
    if remove_orphans {
      if is_orphaned_node(&self.inner, root_node, to) {
        self.remove_node(js_to, js_root_node);
      }
      if is_orphaned_node(&self.inner, root_node, from) {
        self.remove_node(js_from, js_root_node);
      }
    }

    Ok(())
  }

  /// Remove a list of edges. Does not run clean-up
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

  /// Get the node indexes connected incoming into a node.
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

  /// Get the node indexes connected outgoing from a node.
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

  /// Run DFS but in post-order.
  #[napi]
  pub fn post_order_dfs(
    &self,
    env: Env,
    start_node: JSNodeIndex,
    visit: JsFunction,
  ) -> napi::Result<()> {
    let start_node = NodeIndex::new(start_node as usize);
    let mut dfs = DfsPostOrder::new(&self.inner, start_node);

    while let Some(node) = dfs.next(&self.inner) {
      let js_node_idx = env.create_int64(node.index() as i64)?;
      visit
        .call(None, &[&js_node_idx.into_unknown()])?
        .into_unknown();
    }

    Ok(())
  }

  #[napi]
  pub fn is_orphaned_node(&self, root_index: JSNodeIndex, node_index: JSNodeIndex) -> bool {
    let root_index = NodeIndex::new(root_index as usize);
    let node_index = NodeIndex::new(node_index as usize);

    is_orphaned_node(&self.inner, root_index, node_index)
  }
}

// MARK: Serialized Graph and conversion functions

/// The graph is serialized to a list of nodes and a list of edges.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct SerializedGraph {
  nodes: Vec<u32>,
  edges: Vec<(u32, u32, EdgeWeight)>,
  removed_nodes: Vec<u32>,
}

impl From<&ParcelGraphImpl> for SerializedGraph {
  fn from(value: &ParcelGraphImpl) -> Self {
    SerializedGraph {
      removed_nodes: value
        .removed_nodes
        .iter()
        .map(|item| item.index() as u32)
        .collect(),
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

impl From<&SerializedGraph> for ParcelGraphImpl {
  fn from(value: &SerializedGraph) -> Self {
    let mut output: ParcelGraphImpl = Self::new();
    output.removed_nodes = value
      .removed_nodes
      .iter()
      .map(|index| NodeIndex::new(*index as usize))
      .collect();
    for node in value.nodes.iter() {
      output.inner.add_node(*node);
    }
    for edge in value.edges.iter() {
      output.inner.add_edge(edge.0.into(), edge.1.into(), edge.2);
    }
    output
  }
}

// MARK: Algorithm helpers

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

  /// Assert two graphs are equal.
  fn assert_graph_is_equal(graph: &ParcelGraphImpl, other: &ParcelGraphImpl) {
    assert_eq!(graph.node_count(), other.node_count());
    assert_eq!(graph.removed_nodes, other.removed_nodes);
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
    let mut graph: ParcelGraphImpl = ParcelGraphImpl::new();
    let root = graph.inner.add_node(0);
    let idx1 = graph.inner.add_node(0);
    let idx2 = graph.inner.add_node(0);
    let idx3 = graph.inner.add_node(0);
    graph.remove_node(idx3.index() as JSNodeIndex, root.index() as JSNodeIndex);

    graph.inner.add_edge(root, idx1, 0);
    graph.inner.add_edge(idx1, idx2, 0);

    let serialized = SerializedGraph::from(&graph);
    let deserialized = ParcelGraphImpl::from(&serialized);
    assert_graph_is_equal(&graph, &deserialized);
  }

  #[test]
  fn test_add_node() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(0);
    let idx2 = graph.add_node(0);
    let idx3 = graph.add_node(0);
    assert_eq!(graph.node_count(), 3);
    assert_eq!(idx1, 0);
    assert_eq!(idx2, 1);
    assert_eq!(idx3, 2);
  }

  #[test]
  fn test_has_node() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(0);
    let idx2 = graph.add_node(0);
    assert!(graph.has_node(idx1));
    assert!(graph.has_node(idx2));
    assert!(!graph.has_node(3));
  }

  #[test]
  fn test_node_weight() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(42);
    let idx2 = graph.add_node(43);
    assert_eq!(graph.node_weight(idx1), Some(42));
    assert_eq!(graph.node_weight(idx2), Some(43));
    assert_eq!(graph.node_weight(3), None);
  }

  #[test]
  fn test_remove_node() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(0);
    let idx2 = graph.add_node(0);
    let idx3 = graph.add_node(0);
    assert_eq!(graph.node_count(), 3);

    graph.remove_node(idx2, idx1);
    assert_eq!(graph.node_count(), 2);
    assert!(graph.has_node(idx1));
    assert!(!graph.has_node(idx2));
    assert!(graph.has_node(idx3));
  }

  #[test]
  fn test_remove_edge_makes_edge_none() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(0);
    let idx2 = graph.add_node(0);
    let _edge = graph.add_edge(idx1, idx2, 0).unwrap();
    assert!(graph.has_edge(idx1, idx2, vec![]));
    graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
    assert!(!graph.has_edge(idx1, idx2, vec![]));
  }

  #[test]
  fn test_removed_edge_is_not_returned_anymore() {
    let mut graph = ParcelGraphImpl::new();
    let idx1 = graph.add_node(0);
    let idx2 = graph.add_node(0);
    let _edge = graph.add_edge(idx1, idx2, 0).unwrap();
    graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
    assert!(!graph.has_edge(idx1, idx2, vec![]));

    assert_eq!(graph.get_all_edges(), vec![]);
  }

  #[test]
  fn test_remove_edge_should_prune_graph_at_that_edge() {
    let mut graph = ParcelGraphImpl::new();
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
    let mut graph = ParcelGraphImpl::new();
    assert_eq!(graph.node_count(), 0);
    graph.add_node(0);
    assert_eq!(graph.node_count(), 1);
  }

  #[test]
  fn test_node_count_decreases_when_node_is_removed() {
    let mut graph = ParcelGraphImpl::new();
    let idx = graph.add_node(0);
    assert_eq!(graph.node_count(), 1);
    graph.remove_node(idx, idx);
    assert_eq!(graph.node_count(), 0);
  }

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
