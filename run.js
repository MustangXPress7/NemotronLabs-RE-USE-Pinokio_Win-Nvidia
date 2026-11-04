const path = require('path');

module.exports = () => {
	const config = {
		daemon: true,
		cmd: {
			'Default': 'python app.py'
		},
		run: [
			{
				method: 'local.set',
				params: {
					mode: '{{ input.mode }}'
				}
			},
			{
				method: 'shell.run',
				params: {
					message: '{{ self.cmd[local.mode] }}',
					path: '.',
					conda: {
						path: path.resolve(__dirname, '.env')
					},
					on: [
						{
							event: '/(http:\/\/[0-9.:]+)/',
							done: true
						}
					]
				}
			},
			{
				method: 'local.set',
				params: {
					url: '{{ input.event[1] }}'
				}
			}
		]
	};

	return config;
};