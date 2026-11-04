const path = require('path');

module.exports = {
  run: [
    {
      method: 'shell.run',
      params: {
        message: 'git pull',
        path: __dirname
      }
    },
    {
      method: 'shell.run',
      params: {
        message: 'git submodule update --init --recursive',
        path: __dirname
      }
    },
    {
      method: 'input',
      params: {
        title: 'Update completed',
        description: 'Repository updated. Run Install again if dependencies changed.'
      }
    }
  ]
};