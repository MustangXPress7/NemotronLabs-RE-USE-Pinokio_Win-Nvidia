module.exports = {
  run: [
    {
      method: 'shell.run',
      params: {
        message: 'python verify_gpu.py',
        path: '.',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'input',
      params: {
        title: 'GPU verification passed',
        description: 'CUDA is available and the required architectures are present. See the terminal output for the JSON report.'
      }
    }
  ]
};
