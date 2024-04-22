use std::collections::HashSet;

use napi::bindgen_prelude::{Array, Buffer, FromNapiValue, ObjectFinalize};
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, Ref};
use napi_derive::napi;
use petgraph::graph::NodeIndex;
use petgraph::visit::{Dfs, EdgeRef, NodeRef, Reversed};
use petgraph::Graph;
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};

use inner::GraphInner;

mod dfs_actions;
mod inner;

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
#[napi(custom_finalize)]
pub struct ParcelGraphImpl {
  inner: GraphInner,
  nodes: Ref<()>,
}

impl ObjectFinalize for ParcelGraphImpl {
  fn finalize(mut self, env: Env) -> napi::Result<()> {
    self.nodes.unref(env)?;
    Ok(())
  }
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
  pub fn new(env: Env) -> napi::Result<Self> {
    let nodes_array = env.create_array(0)?;
    let nodes_ref = env.create_reference(nodes_array.coerce_to_object()?)?;
    Ok(Self {
      inner: GraphInner::new(),
      nodes: nodes_ref,
    })
  }

  /// Deserialize a graph from a buffer.
  ///
  /// NOTE: Not using `napi(factory)` because that breaks RustRover
  /// https://youtrack.jetbrains.com/issue/RUST-11565
  #[napi]
  pub fn deserialize(env: Env, serialized: Buffer) -> napi::Result<Self> {
    let value = serialized.as_ref();
    let serialized_graph: inner::SerializedGraph =
      from_bytes(value).map_err(|_| napi::Error::from_reason("Failed to deserialize"))?;
    let nodes_array = env.create_array(0)?;
    let nodes_ref = env.create_reference(nodes_array.coerce_to_object()?)?;

    Ok(Self {
      inner: GraphInner::from_serialized(&serialized_graph)?,
      nodes: nodes_ref,
    })
  }

  /// Serialize the graph to a buffer. This copies the Graph data into a `SerializedGraph`,
  /// but the buffer is not copied into JavaScript. So we can optimise this quite a bit further.
  #[napi]
  pub fn serialize(&self, _env: Env) -> napi::Result<Buffer> {
    let serialized = inner::SerializedGraph::from(&self.inner);
    let serialized =
      to_allocvec(&serialized).map_err(|_err| napi::Error::from_reason("Failed to serialize"))?;
    Ok(Buffer::from(serialized.as_ref()))
  }

  /// Add a node and return its index
  ///
  /// O(1) amortized ; but might resize internal vectors
  #[napi]
  pub fn add_node(&mut self, env: Env, weight: JsUnknown) -> napi::Result<JSNodeIndex> {
    // let weight = env.create_reference(weight)?;
    let node_index = self.inner.add_node();
    let nodes = &self.nodes;
    let mut nodes_array: Array = get_array_reference(env, &nodes)?;
    nodes_array.insert(weight)?;
    Ok(node_index.index() as u32)
  }

  /// Return true if a node exists in the Graph
  ///
  /// O(1)
  #[napi]
  pub fn has_node(&self, node_index: JSNodeIndex) -> bool {
    let node_index = NodeIndex::new(node_index as usize);
    self.inner.node_weight(node_index).is_some()
  }

  /// Query the weight of a node.
  ///
  /// O(1)
  #[napi]
  pub fn node_weight(&self, node_index: JSNodeIndex) -> Option<NodeWeight> {
    self
      .inner
      .node_weight(NodeIndex::new(node_index as usize))
      .cloned()
  }

  #[napi]
  pub fn get_nodes(&self, env: Env) -> napi::Result<JsObject> {
    let nodes = &self.nodes;
    let js_nodes = env.get_reference_value(&nodes)?;
    Ok(js_nodes)
  }

  #[napi]
  pub fn get_node(&self, env: Env, node_index: JSNodeIndex) -> napi::Result<Option<JsUnknown>> {
    let node_index: NodeIndex = NodeIndex::new(node_index as usize);

    let nodes = &self.nodes;
    let nodes_array: Array = get_array_reference(env, &nodes)?;
    nodes_array.get::<JsUnknown>(node_index.index() as u32)
  }

  /// Mark node as removed.
  /// petgraph removal will invalidate the last node index since it will be moved.
  ///
  /// Ideally we would remove the node and fix the JS side to make sure it's okay with
  /// indexes changing. This is much better as otherwise the Graph never shrinks.
  #[napi]
  pub fn remove_node(
    &mut self,
    env: Env,
    js_node_index: JSNodeIndex,
    js_root_node: JSNodeIndex,
  ) -> napi::Result<()> {
    // petgraph node removal will invalidate the last node index since it will be moved
    // to the removed node index.
    // Because of this, we will not remove the node, but instead mark it as removed.

    let nodes = &self.nodes;
    let mut nodes_array = get_array_reference(env, &nodes)?;
    let new_length = nodes_array.len() - 1;
    nodes_array.set(js_node_index, nodes_array.get::<JsUnknown>(new_length))?;
    let mut nodes_obj = nodes_array.coerce_to_object()?;
    nodes_obj.set("length", new_length)?;

    // let get_edges = |inner: &Graph<NodeWeight, EdgeWeight>, direction| {
    //   inner
    //     .edges_directed(NodeIndex::new(js_node_index as usize), direction)
    //     .map(|edge| (edge.id(), edge.target()))
    //     .collect::<Vec<(EdgeIndex, NodeIndex)>>()
    // };
    //
    // let outgoing_edges = get_edges(&self.inner, Direction::Outgoing);
    // let rewritten_node_index = self.inner.node_count() - 1;
    // self
    //   .inner
    //   .remove_node(NodeIndex::new(js_node_index as usize));
    //
    // for (_, target) in outgoing_edges {
    //   let target = if target.index() == rewritten_node_index {
    //     NodeIndex::new(js_node_index as usize)
    //   } else {
    //     target
    //   };
    //
    //   // Orphan clean-up
    //   // TODO: We don't want to do this here as it's the wrong side-effect and messes-up complexity
    //   let root_node = NodeIndex::new(js_root_node as usize);
    //   let js_target = target.index() as u32;
    //   if is_orphaned_node(&self.inner, root_node, target) {
    //     self.remove_node(env, js_target, js_root_node)?;
    //   }
    // }

    Ok(())
  }

  /// Count the number of nodes on the graph.
  #[napi]
  pub fn node_count(&self) -> u32 {
    self.inner.node_count() as u32
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
    self.inner.has_edge(
      NodeIndex::new(from as usize),
      NodeIndex::new(to as usize),
      maybe_weight,
    )
  }

  #[napi]
  pub fn get_all_edges(&self) -> Vec<EdgeDescriptor> {
    self.inner.get_all_edges()
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
    env: Env,
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

    // let from = NodeIndex::new(js_from as usize);
    // let to = NodeIndex::new(js_to as usize);
    // let root_node = NodeIndex::new(js_root_node as usize);
    //
    // let edges = self.inner.edges_connecting(from, to);
    // let edges_to_remove: Vec<EdgeIndex> = edges
    //   .filter(|edge| {
    //     if !maybe_weight.is_empty() {
    //       maybe_weight.contains(edge.weight())
    //     } else {
    //       true
    //     }
    //   })
    //   .map(|edge| edge.id())
    //   .collect();
    // for edge_id in edges_to_remove {
    //   self.inner.remove_edge(edge_id);
    // }
    //
    // // TODO: We don't want to do this here as it's the wrong side-effect and messes-up complexity
    // if remove_orphans {
    //   if is_orphaned_node(&self.inner, root_node, to) {
    //     self.remove_node(env, js_to, js_root_node)?;
    //   }
    //   if is_orphaned_node(&self.inner, root_node, from) {
    //     self.remove_node(env, js_from, js_root_node)?;
    //   }
    // }

    Ok(())
  }

  /// Remove a list of edges. Does not run clean-up
  #[napi]
  pub fn remove_edges(&mut self, node_index: JSNodeIndex, maybe_weight: Vec<EdgeWeight>) {
    self
      .inner
      .remove_edges(NodeIndex::new(node_index as usize), maybe_weight)
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
      .get_node_ids_connected_to(NodeIndex::new(node_index as usize), edge_weight)
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
      .get_node_ids_connected_from(NodeIndex::new(node_index as usize), edge_weight)
  }

  /// Will return an empty vec if the graph has a cycle
  #[napi]
  pub fn topo_sort(&self) -> Vec<JSNodeIndex> {
    let result = self.inner.topo_sort();
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
    let result = self.inner.dfs(
      NodeIndex::new(start_node as usize),
      |node: NodeIndex| -> napi::Result<JsUnknown> {
        let js_node_idx = env.create_int64(node.index() as i64)?;
        enter.call(None, &[&js_node_idx.into_unknown()])
      },
      exit.map(|exit| {
        move |node: NodeIndex| -> napi::Result<JsUnknown> {
          let js_node_idx = env.create_int64(node.index() as i64)?;
          exit.call(None, &[js_node_idx.into_unknown()])
        }
      }),
      maybe_edge_weight,
      traverse_up,
    )?;

    let js_undefined = env.get_undefined()?.into_unknown();
    Ok(result.unwrap_or_else(|| js_undefined))

    // let exit = exit.map(Rc::new);
    // let direction = if traverse_up {
    //   Direction::Incoming
    // } else {
    //   Direction::Outgoing
    // };
    //
    // enum QueueCommand {
    //   Item {
    //     idx: NodeIndex,
    //     context: Rc<JsUnknown>,
    //   },
    //   Exit {
    //     f: Rc<JsFunction>,
    //     idx: NodeIndex,
    //   },
    // }
    //
    // let mut queue: Vec<QueueCommand> = vec![];
    // let mut visited = HashSet::<NodeIndex>::new();
    // let initial_context = env.get_undefined()?.into_unknown();
    //
    // let start_node = NodeIndex::new(start_node as usize);
    // queue.push(QueueCommand::Item {
    //   idx: start_node,
    //   context: Rc::new(initial_context),
    // });
    //
    // let actions = DFSActions::new();
    // let js_actions = actions.to_js(&env)?;
    //
    // while let Some(command) = queue.pop() {
    //   match command {
    //     QueueCommand::Item { idx, context } => {
    //       if visited.contains(&idx) {
    //         continue;
    //       }
    //
    //       visited.insert(idx);
    //       let js_node_idx = env.create_int64(idx.index() as i64)?;
    //
    //       actions.reset();
    //
    //       // Visit
    //       let new_context =
    //         enter.call(None, &[&js_node_idx.into_unknown(), &context, &js_actions])?;
    //
    //       if actions.is_skipped() {
    //         continue;
    //       }
    //
    //       if actions.is_stopped() {
    //         return Ok(new_context);
    //       }
    //       let new_context = Rc::new(new_context);
    //
    //       if let Some(exit) = &exit {
    //         queue.push(QueueCommand::Exit {
    //           f: exit.clone(),
    //           idx,
    //         });
    //       }
    //       for child in self.inner.edges_directed(idx, direction) {
    //         let matches_target_weight =
    //           maybe_edge_weight.is_empty() || maybe_edge_weight.contains(child.weight());
    //         if !matches_target_weight {
    //           continue;
    //         }
    //
    //         let other_id = if traverse_up {
    //           child.source().id()
    //         } else {
    //           child.target().id()
    //         };
    //         if visited.contains(&other_id) {
    //           continue;
    //         }
    //
    //         queue.push(QueueCommand::Item {
    //           idx: other_id,
    //           context: new_context.clone(),
    //         })
    //       }
    //     }
    //     QueueCommand::Exit { f, idx } => {
    //       let js_node_idx = env.create_int64(idx.index() as i64)?;
    //       f.call(None, &[js_node_idx])?;
    //     }
    //   }
    // }
    //
    // Ok(env.get_undefined()?.into_unknown())
  }

  #[napi]
  pub fn get_unreachable_nodes(&self, root_index: JSNodeIndex) -> Vec<JSNodeIndex> {
    self
      .inner
      .get_unreachable_nodes(NodeIndex::new(root_index as usize))
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
    self.inner.post_order_dfs(
      NodeIndex::new(start_node as usize),
      |node: NodeIndex| -> napi::Result<()> {
        let js_node_idx = env.create_int64(node.index() as i64)?;
        let _: JsUnknown = visit.call(None, &[&js_node_idx.into_unknown()])?;
        Ok(())
      },
    )
  }

  #[napi]
  pub fn is_orphaned_node(&self, root_index: JSNodeIndex, node_index: JSNodeIndex) -> bool {
    let root_index = NodeIndex::new(root_index as usize);
    let node_index = NodeIndex::new(node_index as usize);

    self.inner.is_orphaned_node(root_index, node_index)
  }
}

// MARK: Serialized Graph and conversion functions
//
// /// The graph is serialized to a list of nodes and a list of edges.
// #[derive(Serialize, Deserialize, Debug, PartialEq)]
// struct SerializedGraph {
//   nodes: Vec<u32>,
//   edges: Vec<(u32, u32, EdgeWeight)>,
// }
//
// impl From<&ParcelGraphImpl> for SerializedGraph {
//   fn from(value: &ParcelGraphImpl) -> Self {
//     SerializedGraph {
//       nodes: value
//         .inner
//         .raw_nodes()
//         .iter()
//         .map(|item| item.weight)
//         .collect(),
//       edges: value
//         .inner
//         .raw_edges()
//         .iter()
//         .map(|item| {
//           (
//             item.source().index() as u32,
//             item.target().index() as u32,
//             item.weight,
//           )
//         })
//         .collect(),
//     }
//   }
// }
//
// impl ParcelGraphImpl {
//   fn from_serialized(env: Env, value: &SerializedGraph) -> napi::Result<Self> {
//     let mut output: ParcelGraphImpl = Self::new(env)?;
//     for node in value.nodes.iter() {
//       output.inner.add_node(*node);
//     }
//     for edge in value.edges.iter() {
//       output.inner.add_edge(edge.0.into(), edge.1.into(), edge.2);
//     }
//     Ok(output)
//   }
// }
//
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

/// Get an array from a NAPI reference.
///
/// This is unsafe because NAPI does not implement conversion wrappers for Arrays.
///
/// We do a runtime check to make sure the reference is an object and that the object is an
/// array.
///
/// ## Safety
///
/// This should be safe if:
///
/// * the reference is valid ; we are not using it after free (after ref count is 0)
/// * the environment is valid
fn get_array_reference(env: Env, nodes: &Ref<()>) -> napi::Result<Array> {
  let object = env.get_reference_value::<JsObject>(&nodes)?;
  if object.is_array()? != true {
    return Err(napi::Error::from_reason("Expected array"));
  }

  unsafe { Array::from_napi_value(env.raw(), object.raw()) }
}

#[cfg(test)]
mod test {
  // use super::*;
  //
  // /// Assert two graphs are equal.
  // fn assert_graph_is_equal(graph: &ParcelGraphImpl, other: &ParcelGraphImpl) {
  //   assert_eq!(graph.node_count(), other.node_count());
  //   assert_eq!(
  //     graph
  //       .inner
  //       .raw_nodes()
  //       .iter()
  //       .map(|node| node.weight)
  //       .collect::<Vec<NodeWeight>>(),
  //     other
  //       .inner
  //       .raw_nodes()
  //       .iter()
  //       .map(|node| node.weight)
  //       .collect::<Vec<NodeWeight>>()
  //   );
  //   assert_eq!(
  //     graph
  //       .inner
  //       .raw_edges()
  //       .iter()
  //       .map(|node| node.weight)
  //       .collect::<Vec<NodeWeight>>(),
  //     other
  //       .inner
  //       .raw_edges()
  //       .iter()
  //       .map(|node| node.weight)
  //       .collect::<Vec<NodeWeight>>()
  //   );
  // }
  //
  // #[test]
  // fn test_serialize_graph() {
  //   let mut graph: ParcelGraphImpl = ParcelGraphImpl::new();
  //   let root = graph.inner.add_node(0);
  //   let idx1 = graph.inner.add_node(0);
  //   let idx2 = graph.inner.add_node(0);
  //   let idx3 = graph.inner.add_node(0);
  //   graph.remove_node(idx3.index() as JSNodeIndex, root.index() as JSNodeIndex);
  //
  //   graph.inner.add_edge(root, idx1, 0);
  //   graph.inner.add_edge(idx1, idx2, 0);
  //
  //   let serialized = SerializedGraph::from(&graph);
  //   let deserialized = ParcelGraphImpl::from(&serialized);
  //   assert_graph_is_equal(&graph, &deserialized);
  // }
  //
  // #[test]
  // fn test_add_node() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let idx3 = graph.add_node(0);
  //   assert_eq!(graph.node_count(), 3);
  //   assert_eq!(idx1, 0);
  //   assert_eq!(idx2, 1);
  //   assert_eq!(idx3, 2);
  // }
  //
  // #[test]
  // fn test_has_node() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   assert!(graph.has_node(idx1));
  //   assert!(graph.has_node(idx2));
  //   assert!(!graph.has_node(3));
  // }
  //
  // #[test]
  // fn test_node_weight() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(42);
  //   let idx2 = graph.add_node(43);
  //   assert_eq!(graph.node_weight(idx1), Some(42));
  //   assert_eq!(graph.node_weight(idx2), Some(43));
  //   assert_eq!(graph.node_weight(3), None);
  // }
  //
  // #[test]
  // fn test_remove_node() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let idx3 = graph.add_node(0);
  //   assert_eq!(graph.node_count(), 3);
  //   graph.remove_node(idx2, idx1);
  //   assert_eq!(graph.node_count(), 2);
  //   assert!(graph.has_node(idx1));
  //   assert!(!graph.has_node(idx2));
  //   assert!(graph.has_node(idx3));
  // }
  //
  // #[test]
  // fn test_remove_edge_makes_edge_none() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let _edge = graph.add_edge(idx1, idx2, 0).unwrap();
  //   assert!(graph.has_edge(idx1, idx2, vec![]));
  //   graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
  //   assert!(!graph.has_edge(idx1, idx2, vec![]));
  // }
  //
  // #[test]
  // fn test_removed_edge_is_not_returned_anymore() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx1 = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let _edge = graph.add_edge(idx1, idx2, 0).unwrap();
  //   graph.remove_edge(idx1, idx2, vec![], true, idx1).unwrap();
  //   assert!(!graph.has_edge(idx1, idx2, vec![]));
  //   assert_eq!(graph.get_all_edges(), vec![]);
  // }
  //
  // #[test]
  // fn test_remove_edge_should_prune_graph_at_that_edge() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let root = graph.add_node(0);
  //   let idx2 = graph.add_node(0);
  //   let idx3 = graph.add_node(0);
  //   let idx4 = graph.add_node(0);
  //   graph.add_edge(root, idx2, 0).unwrap();
  //   graph.add_edge(root, idx4, 0).unwrap();
  //   graph.add_edge(idx2, idx3, 0).unwrap();
  //   graph.add_edge(idx2, idx4, 0).unwrap();
  //   graph.remove_edge(root, idx2, vec![], true, root).unwrap();
  //   assert!(!graph.has_edge(idx3, idx4, vec![]));
  //   assert!(graph.has_node(root));
  //   assert!(!graph.has_node(idx2));
  //   assert!(!graph.has_node(idx3));
  //   assert!(graph.has_node(idx4));
  //   assert_eq!(
  //     graph.get_all_edges(),
  //     vec![EdgeDescriptor {
  //       from: root,
  //       to: idx4,
  //       weight: 0
  //     }]
  //   )
  // }
  //
  // #[test]
  // fn test_node_count_increases_when_node_is_added() {
  //   let mut graph = ParcelGraphImpl::new();
  //   assert_eq!(graph.node_count(), 0);
  //   graph.add_node(0);
  //   assert_eq!(graph.node_count(), 1);
  // }
  //
  // #[test]
  // fn test_node_count_decreases_when_node_is_removed() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let idx = graph.add_node(0);
  //   assert_eq!(graph.node_count(), 1);
  //   graph.remove_node(idx, idx);
  //   assert_eq!(graph.node_count(), 0);
  // }
  //
  // #[test]
  // fn test_is_orphaned_node() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let root = graph.inner.add_node(0);
  //
  //   let idx1 = graph.inner.add_node(0);
  //   let idx2 = graph.inner.add_node(0);
  //   let idx3 = graph.inner.add_node(0);
  //
  //   assert!(is_orphaned_node(&graph.inner, root, idx1));
  //   assert!(is_orphaned_node(&graph.inner, root, idx2));
  //   assert!(is_orphaned_node(&graph.inner, root, idx3));
  //
  //   graph.inner.add_edge(root, idx1, 0);
  //   assert!(!is_orphaned_node(&graph.inner, root, idx1));
  //   assert!(is_orphaned_node(&graph.inner, root, idx2));
  //   assert!(is_orphaned_node(&graph.inner, root, idx3));
  //
  //   graph.inner.add_edge(idx2, idx3, 0);
  //   assert!(!is_orphaned_node(&graph.inner, root, idx1));
  //   assert!(is_orphaned_node(&graph.inner, root, idx2));
  //   assert!(is_orphaned_node(&graph.inner, root, idx3));
  //
  //   graph.inner.add_edge(idx1, idx2, 0);
  //   assert!(!is_orphaned_node(&graph.inner, root, idx1));
  //   assert!(!is_orphaned_node(&graph.inner, root, idx2));
  //   assert!(!is_orphaned_node(&graph.inner, root, idx3));
  // }
  //
  // #[test]
  // fn test_get_unreachable_nodes_on_disconnected_graph() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let root = graph.inner.add_node(0);
  //
  //   let idx1 = graph.inner.add_node(0);
  //   let idx2 = graph.inner.add_node(0);
  //   let idx3 = graph.inner.add_node(0);
  //
  //   let unreachable = get_unreachable_nodes(&graph.inner, root);
  //   assert_eq!(unreachable.len(), 3);
  //   assert_eq!(unreachable, vec![idx1, idx2, idx3]);
  // }
  //
  // #[test]
  // fn test_get_unreachable_nodes_with_direct_root_connection() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let root = graph.inner.add_node(0);
  //
  //   let idx1 = graph.inner.add_node(0);
  //   let idx2 = graph.inner.add_node(0);
  //
  //   graph.inner.add_edge(root, idx1, 0);
  //
  //   let unreachable = get_unreachable_nodes(&graph.inner, root);
  //   assert_eq!(unreachable.len(), 1);
  //   assert_eq!(unreachable, vec![idx2]);
  // }
  //
  // #[test]
  // fn test_get_unreachable_nodes_with_indirect_root_connection() {
  //   let mut graph = ParcelGraphImpl::new();
  //   let root = graph.inner.add_node(0);
  //
  //   let idx1 = graph.inner.add_node(0);
  //   let idx2 = graph.inner.add_node(0);
  //   let idx3 = graph.inner.add_node(0);
  //
  //   graph.inner.add_edge(root, idx1, 0);
  //   graph.inner.add_edge(idx1, idx2, 0);
  //
  //   let unreachable = get_unreachable_nodes(&graph.inner, root);
  //   assert_eq!(unreachable.len(), 1);
  //   assert_eq!(unreachable, vec![idx3]);
  //
  //   graph.inner.add_edge(idx2, idx3, 0);
  //   let unreachable = get_unreachable_nodes(&graph.inner, root);
  //   assert_eq!(unreachable.len(), 0);
  //   assert_eq!(unreachable, vec![]);
  // }

  #[test]
  fn test_compiles() {}
}
