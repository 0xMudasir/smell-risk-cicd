// Default build info. CD (deploy.yml) overwrites this file with the real commit
// SHA, Actions run number, and UTC build time before building for GitHub Pages.
export const buildInfo = {
  sha: 'local',
  runNumber: 0,
  builtAt: '',
};
