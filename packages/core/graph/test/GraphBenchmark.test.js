// @flow strict-local

import {JSGraph} from '../src/Graph/JSGraph';
import {RustGraph} from '../src/Graph/RustGraph';
import assert from 'assert';

function range(n: number): number[] {
  const result = [];
  for (let i = 0; i < n; i++) {
    result.push(i);
  }
  return result;
}

function benchmark<T>(name: string, setup: () => T, fn: (value: T) => void) {
  it(name, () => {
    console.log('setup');
    const setupStart = Date.now();
    const iterations = 10;
    const ctxs = range(iterations).map(() => setup());
    const setupEnd = Date.now();
    const setupDuration = setupEnd - setupStart;
    console.log(`done setting-up duration: ${setupDuration / iterations}ms`);

    {
      const start = Date.now();
      for (let i = 0; i < iterations; i++) {
        fn(ctxs[i]);
      }
      const end = Date.now();
      const duration = end - start;
      console.log(`${name} duration: ${duration / iterations}ms`);
    }
  });
}

describe(`RustGraph benchmarks`, function () {
  this.timeout(500000);

  benchmark(
    'RustGraph can insert nodes',
    () => {
      return new RustGraph();
    },
    graph => {
      graph.addNode(0);
    },
  );

  benchmark(
    'RustGraph can traverse nodes',
    () => {
      const graph = new RustGraph();
      let currentNode = graph.addNode(0);
      for (let i = 0; i < 1e6; i++) {
        const newNode = graph.addNode(0);
        graph.addEdge(currentNode, newNode);
        currentNode = newNode;
      }
      return graph;
    },
    graph => {
      let count = 0;
      graph.dfs({
        visit: () => count++,
        startNodeId: 0,
      });
      assert(count === 1e6 + 1);
    },
  );
});

describe(`JSGraph benchmarks`, () => {
  beforeEach(function () {
    this.timeout(600000);
  });

  benchmark(
    'JSGraph can insert nodes',
    () => {
      return new JSGraph();
    },
    graph => {
      graph.addNode(0);
    },
  );

  benchmark(
    'JSGraph can traverse nodes',
    () => {
      const graph = new RustGraph();
      let currentNode = graph.addNode(0);
      for (let i = 0; i < 1e6; i++) {
        const newNode = graph.addNode(0);
        graph.addEdge(currentNode, newNode);
        currentNode = newNode;
      }
      return graph;
    },
    graph => {
      let count = 0;
      graph.dfs({
        visit: () => count++,
        startNodeId: 0,
      });
      assert(count === 1e6 + 1);
    },
  );
});
