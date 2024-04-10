// @flow strict-local

import {getFeatureFlag} from '@parcel/feature-flags/src';
import {JSGraph} from './Graph/JSGraph';
import {RustGraph} from './Graph/RustGraph';

// const Graph = getFeatureFlag('rustBackedGraph') ? RustGraph : JSGraph;
const Graph = JSGraph;

export {Graph};
export default Graph;
