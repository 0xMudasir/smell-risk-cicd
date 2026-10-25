// Default (Section 8.4): the live Pages site calls a local Flask API, so Predict
// works on the presenter's machine and shows the offline state elsewhere. Point
// this at a hosted API if one is deployed (ask the owner first).
export const environment = {
  production: true,
  apiBaseUrl: 'http://localhost:5000',
};
