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
      // LiteRT graphs downloaded by install.js.
      method: 'fs.rm',
      params: {
        path: 'models_litert',
        recursive: true,
        force: true
      }
    },
    {
      // Phase 2 model pack for the native MLX / Core ML backend.
      method: 'fs.rm',
      params: {
        path: 'ModelPack',
        recursive: true,
        force: true
      }
    },
    {
      // Phase 2 Swift SDK extracted beside the model pack.
      method: 'fs.rm',
      params: {
        path: 'ReuseKit',
        recursive: true,
        force: true
      }
    },
    {
      method: 'input',
      params: {
        title: 'Reset completed',
        description: 'The Python environment, model cache, and downloaded model packs were removed. Your output/ and logs/ folders were kept.'
      }
    }
  ]
};
