module.exports = {
  run: [
    {
      method: 'shell.run',
      params: {
        message: 'python verify_mac.py',
        path: '.',
        conda: {
          path: '.env'
        }
      }
    },
    {
      method: 'input',
      params: {
        title: 'Metal verification passed',
        description: 'Apple Silicon host, LiteRT graphs and dependencies are present. See the terminal output for the JSON report.'
      }
    }
  ]
};
