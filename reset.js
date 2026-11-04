const path = require('path');

module.exports = {
  run: [
    {
      method: 'fs.rm',
      params: {
        path: path.resolve(__dirname, '.env'),
        recursive: true,
        force: true
      }
    },
    {
      method: 'fs.rm',
      params: {
        path: path.resolve(__dirname, 'cache'),
        recursive: true,
        force: true
      }
    },
    {
      method: 'fs.rm',
      params: {
        path: path.resolve(__dirname, 'logs'),
        recursive: true,
        force: true
      }
    },
    {
      method: 'fs.rm',
      params: {
        path: path.resolve(__dirname, 'output'),
        recursive: true,
        force: true
      }
    },
    {
      method: 'input',
      params: {
        title: 'Reset completed',
        description: 'Environment cleaned. You can now run Install again.'
      }
    }
  ]
};