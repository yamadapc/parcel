// @flow strict-local

import {JSGraph} from './Graph/JSGraph';
import {RustGraph} from './Graph/RustGraph';

const Graph = RustGraph; // JSGraph;
// const Graph = JSGraph;

export {Graph};
export default Graph;
