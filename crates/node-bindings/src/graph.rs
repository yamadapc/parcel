use std::collections::HashSet;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};

use napi::bindgen_prelude::{Buffer, FromNapiValue};
use napi::{Env, JsFunction, JsUnknown};
use napi_derive::napi;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, NodeRef};
use petgraph::{Directed, Direction, Graph};
use postcard::{from_bytes, to_allocvec};
use serde::{Deserialize, Serialize};

type JSNodeIndex = u32;
type JSEdgeIndex = u32;
type NodeWeight = u32;
type EdgeWeight = u32;

/// Helper for visitor API of Graph traversal
struct DFSActions {
  /// If JS flips this on a parent we will skip its children
  skipped: Rc<AtomicBool>,
  /// If JS flips this on a parent we will return immediately
  stopped: Rc<AtomicBool>,
}

impl DFSActions {
  fn new() -> Self {
    Self {
      skipped: Rc::new(AtomicBool::new(false)),
      stopped: Rc::new(AtomicBool::new(false)),
    }
  }

  fn set_skipped(&self, value: bool) {
    self.skipped.store(value, Ordering::Relaxed);
  }

  fn is_skipped(&self) -> bool {
    self.skipped.load(Ordering::Relaxed)
  }

  fn is_stopped(&self) -> bool {
    self.stopped.load(Ordering::Relaxed)
  }

  fn to_js(&self, env: &Env) -> napi::Result<JsUnknown> {
    let mut js_actions = env.create_object()?;
    js_actions.set(
      "skipChildren",
      env.create_function_from_closure("skipChildren", {
        let skipped = self.skipped.clone();
        move |_| {
          skipped.store(true, Ordering::Relaxed);
          Ok(())
        }
      })?,
    )?;
    js_actions.set(
      "stop",
      env.create_function_from_closure("stop", {
        let stopped = self.stopped.clone();
        move |_| {
          stopped.store(true, Ordering::Relaxed);
          Ok(())
        }
      })?,
    )?;
    Ok(js_actions.into_unknown())
  }
}

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

#[napi]
impl ParcelGraphImpl {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self {
      inner: Graph::new(),
    }
  }

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
      output.add_edge(edge.0, edge.1, edge.2);
    }
    Ok(output)
  }

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
    from: JSNodeIndex,
    to: JSNodeIndex,
    weight: EdgeWeight,
  ) -> JSEdgeIndex {
    self
      .inner
      .add_edge(
        NodeIndex::new(from as usize),
        NodeIndex::new(to as usize),
        weight,
      )
      .index() as u32
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
  pub fn remove_edge(&mut self, from: JSNodeIndex, to: JSNodeIndex, maybe_weight: Vec<EdgeWeight>) {
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

          actions.set_skipped(false);

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

  // #[napi]
  // pub fn ancestors(&self, node_index: JSNodeIndex) -> Vec<JSNodeIndex> {
  //   self.inner
  // }
  // #[napi]
  // pub fn descendants(&self, node_index: JSNodeIndex) -> Vec<JSNodeIndex> {
  // }
}

#[cfg(test)]
mod test {
  #[test]
  fn test_compiles() {}
}
