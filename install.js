const path = require('path');

module.exports = {
  requires: {
    bundle: "ai"
  },
  run: [
    {
      method: 'shell.run',
      params: {
        message: 'conda install python=3.10 --yes',
        conda: {
          path: path.resolve(__dirname, '.env')
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'conda install conda-forge::ffmpeg --yes',
        conda: {
          path: path.resolve(__dirname, '.env')
        }
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'python -m pip install --upgrade pip',
        conda: {
          path: path.resolve(__dirname, '.env')
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
          path: path.resolve(__dirname, '.env')
        }
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: 'shell.run',
      params: {
        message: 'git clone https://github.com/Dao-AILab/causal-conv1d.git && cd causal-conv1d && python setup.py install && cd ..',
        conda: {
          path: path.resolve(__dirname, '.env')
        },
        env: {
          CAUSAL_CONV1D_FORCE_BUILD: 'TRUE',
          MAX_JOBS: '1'
        }
      }
    },
    {
      when: "{{platform === 'win32'}}",
      method: 'shell.run',
      params: {
        message: 'git clone https://github.com/state-spaces/mamba.git && cd mamba && python setup.py install && cd ..',
        conda: {
          path: path.resolve(__dirname, '.env')
        },
        env: {
          MAMBA_FORCE_BUILD: 'TRUE',
          MAX_JOBS: '1'
        }
      }
    },
    {
      when: "{{platform !== 'win32'}}",
      method: 'shell.run',
      params: {
        message: 'pip install --no-build-isolation --no-deps causal-conv1d mamba-ssm',
        conda: {
          path: path.resolve(__dirname, '.env')
        }
      }
    },
    {
      method: 'input',
      params: {
        title: 'Installation completed',
        description: 'Return to the dashboard to start RE-USE Speech Enhancement.'
      }
    }
  ]
};