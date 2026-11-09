module.exports = {
  run: [
    {
      // Python environment created by install.js.
      method: 'fs.rm',
      params: {
        path: '.env',
        recursive: true,
        force: true
      }
    },
    {
      // Model and audio caches populated at runtime.
      method: 'fs.rm',
      params: {
        path: 'cache',
        recursive: true,
        force: true
      }
    },
    {
      // Pinned mamba checkout and pure-Python build.
      method: 'fs.rm',
      params: {
        path: 'mamba',
        recursive: true,
        force: true
      }
    },
    {
      // Legacy cleanup only: older installers cloned causal-conv1d.
      // The current installer does not create this folder.
      method: 'fs.rm',
      params: {
        path: 'causal-conv1d',
        recursive: true,
        force: true
      }
    },
    {
      method: 'input',
      params: {
        title: 'Reset completed',
        description: 'The Python environment, model cache, and mamba build were removed. Your output/ and logs/ folders were kept.'
      }
    }
  ]
};
