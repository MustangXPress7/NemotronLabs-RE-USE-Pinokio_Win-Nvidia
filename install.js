module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      // Fail before downloading anything on an unsupported host.
      when: "{{platform !== 'darwin' || arch !== 'arm64'}}",
      method: 'shell.run',
      params: {
        message: "echo RE-USE for Apple Silicon requires macOS on arm64. Pinokio detected platform={{platform}} arch={{arch}} gpu={{gpu}}. See README_MAC.md.",
        on: [{
          event: "/(RE-USE for Apple Silicon requires)/",
          break: true
        }]
      }
    },
    {
      // Python 3.12 matches the environment used to validate ai-edge-litert 2.3.0.
      method: 'shell.run',
      params: {
        message: 'python --version',
        conda: {
          path: '.env',
          python: 'python=3.12'
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'uv pip install -r requirements_mac.txt',
        conda: {
          path: '.env'
        }
      }
    },
    {
      // LiteRT graphs for the 30-block standard model.
      method: 'hf.download',
      params: {
        "_": ["litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT", "reuse_16k_2s_fastload_fp32.tflite"],
        "local-dir": "models_litert"
      }
    },
    {
      method: 'hf.download',
      params: {
        "_": ["litert-community/NVIDIA-NemotronLabs-RE-USE-LiteRT", "reuse_16k_2s_fp32.tflite"],
        "local-dir": "models_litert"
      }
    },
    {
      // Same graph with the laughter-preserving weights.
      method: 'hf.download',
      params: {
        "_": ["litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT", "reuse_laughter_16k_2s_fastload_fp32.tflite"],
        "local-dir": "models_litert"
      }
    },
    {
      method: 'hf.download',
      params: {
        "_": ["litert-community/NVIDIA-NemotronLabs-RE-USE-laughter-LiteRT", "reuse_laughter_16k_2s_fp32.tflite"],
        "local-dir": "models_litert"
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'python verify_mac.py',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'input',
      params: {
        title: 'Installation verified',
        description: 'LiteRT/Metal graphs and dependencies are ready. Return to the dashboard to start RE-USE.'
      }
    }
  ]
};
