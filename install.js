module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      // Fail before downloading anything when the machine cannot run this app.
      // The app hardcodes CUDA, so a CPU-only install would only crash later.
      when: "{{gpu !== 'nvidia' || (platform !== 'win32' && platform !== 'linux')}}",
      method: 'shell.run',
      params: {
        message: "echo RE-USE requires an NVIDIA GPU on Windows or Linux. Pinokio detected gpu={{gpu}} platform={{platform}}. See README.md for supported configurations.",
        on: [{
          event: "/(RE-USE requires an NVIDIA GPU)/",
          break: true
        }]
      }
    },
    {
      // Creates .env with Python 3.10 on first install, activates it afterwards.
      method: 'shell.run',
      params: {
        message: 'python --version',
        conda: {
          path: '.env',
          python: 'python=3.10'
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'conda install conda-forge::ffmpeg --yes',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'python -m pip install --upgrade pip',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'script.start',
      params: {
        uri: 'torch.js',
        params: {
          path: '.',
          venv: null,
          conda: '.env',
          xformers: false,
          triton: true,
          sageattention: false,
          flashattention: false
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'uv pip install -r requirements_full.txt',
        conda: {
          path: '.env'
        }
      }
    },
    {
      // A failed clone can leave a non-empty mamba/ directory behind.
      // Remove it so the next step can clone cleanly.
      when: "{{exists('mamba') && !exists('mamba/setup.py')}}",
      method: 'fs.rm',
      params: {
        path: 'mamba',
        recursive: true,
        force: true
      }
    },
    {
      when: "{{!exists('mamba/setup.py')}}",
      method: 'shell.run',
      params: {
        message: 'git clone --branch v2.3.2.post1 --depth 1 https://github.com/state-spaces/mamba.git',
        conda: {
          path: '.env'
        }
      }
    },
    {
      // The pinned tag v2.3.2.post1 reads MAMBA_SKIP_CUDA_BUILD.
      // Do not add MAMBA_KEEP_CUDA_BUILD here: that flag does not exist in this tag.
      method: 'shell.run',
      params: {
        message: 'uv pip install --no-build-isolation --no-deps .',
        path: 'mamba',
        conda: {
          path: '.env'
        },
        env: {
          MAMBA_SKIP_CUDA_BUILD: 'TRUE'
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'python verify_gpu.py',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'input',
      params: {
        title: 'Installation verified',
        description: 'CUDA is available and the required GPU architectures are present. Return to the dashboard to start RE-USE.'
      }
    }
  ]
};
