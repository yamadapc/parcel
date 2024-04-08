use std::collections::HashSet;
use std::rc::Rc;

use napi::{Env, JsFunction, JsUnknown};
use napi_derive::napi;
use petgraph::graph::{EdgeIndex, NodeIndex};
use petgraph::visit::{EdgeRef, NodeRef};
use petgraph::{Directed, Direction, Graph};

type JSNodeIndex = u32;
type JSEdgeIndex = u32;
type NodeWeight = u32;
type EdgeWeight = u32;

/// Internal graph used for Parcel bundle/asset/request tracking, wraps petgraph
/// Edges and nodes have number weights
#[napi]
pub struct ParcelGraphImpl {
  inner: Graph<NodeWeight, EdgeWeight, Directed, u32>,
}

#[napi]
impl ParcelGraphImpl {
  #[napi(constructor)]
  pub fn new() -> Self {
    Self {
      inner: Graph::new(),
    }
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
    maybe_weight: Option<EdgeWeight>,
  ) -> bool {
    self
      .inner
      .edges_connecting(NodeIndex::new(from as usize), NodeIndex::new(to as usize))
      .any(|item| {
        if let Some(weight) = maybe_weight {
          *item.weight() == weight
        } else {
          true
        }
      })
  }

  #[napi]
  pub fn remove_edge(
    &mut self,
    from: JSNodeIndex,
    to: JSNodeIndex,
    maybe_weight: Option<EdgeWeight>,
  ) {
    let edges = self
      .inner
      .edges_connecting(NodeIndex::new(from as usize), NodeIndex::new(to as usize));
    let edges_to_remove: Vec<EdgeIndex> = edges
      .filter(|edge| {
        if let Some(weight) = maybe_weight {
          *edge.weight() == weight
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
  pub fn remove_edges(&mut self, node_index: JSNodeIndex, maybe_weight: Option<EdgeWeight>) {
    let indexes_to_remove = self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Outgoing)
      .filter(|edge| {
        if let Some(weight) = maybe_weight {
          *edge.weight() == weight
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
    edge_weight: Option<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Incoming)
      .filter(|edge| {
        if let Some(target_weight) = edge_weight {
          *edge.weight() == target_weight
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
    edge_weight: Option<EdgeWeight>,
  ) -> Vec<JSNodeIndex> {
    self
      .inner
      .edges_directed(NodeIndex::new(node_index as usize), Direction::Outgoing)
      .filter(|edge| {
        if let Some(target_weight) = edge_weight {
          *edge.weight() == target_weight
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
    maybe_edge_weight: Option<EdgeWeight>,
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
      Item { idx: NodeIndex },
      Exit { f: Rc<JsFunction>, idx: NodeIndex },
    }

    let mut queue: Vec<QueueCommand> = vec![];
    let mut visited = HashSet::<NodeIndex>::new();
    let mut context = env.get_undefined()?.into_unknown();

    queue.push(QueueCommand::Item {
      idx: NodeIndex::new(start_node as usize),
    });

    while let Some(command) = queue.pop() {
      match command {
        QueueCommand::Item { idx } => {
          visited.insert(idx);
          let js_node_idx = env.create_int64(idx.index() as i64)?;
          context = enter.call(None, &[&js_node_idx.into_unknown(), &context])?;

          if let Some(exit) = &exit {
            queue.push(QueueCommand::Exit {
              f: exit.clone(),
              idx,
            });
          }
          for child in self.inner.edges_directed(idx, direction) {
            let matches_target_weight = maybe_edge_weight
              .map(|weight| weight != *child.weight())
              .unwrap_or(true);
            if !matches_target_weight {
              continue;
            }

            queue.push(QueueCommand::Item {
              idx: child.target().id(),
            })
          }
        }
        QueueCommand::Exit { f, idx } => {
          let js_node_idx = env.create_int64(idx.index() as i64)?;
          f.call(None, &[js_node_idx])?;
        }
      }
    }

    Ok(context)
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
