// @flow strict

export type FeatureFlagKey =
  // This feature flag mostly exists to test the feature flag system, and doesn't have any build/runtime effect
  | 'exampleFeature'
  /**
   * Enables content hash based invalidation for config keys used in plugins.
   * This allows Assets not to be invalidated when using
   * `config.getConfigFrom(..., {packageKey: '...'})` and the value itself hasn't changed.
   */
  | 'configKeyInvalidation'
  /**
   * Moves the @parcel/graph internal implementation to rust.
   */
  | 'rustBackedGraph';

export type FeatureFlags = {|
  [FeatureFlagKey]: boolean,
|};
