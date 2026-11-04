module.exports = async (kernel, info) => {
	const menu = [];

	if (!info.exists('.env')) {
		menu.push({
			icon: 'fa-solid fa-plug',
			text: 'Install',
			href: 'install.js',
			params: {
				run: true,
				fullscreen: true
			}
		});

		return menu;
	}

	if (!info.running('run.js') && !info.running('install.js')) {
		[
			['fa-solid fa-power-off', 'Run RE-USE', 'Default']
		].forEach(([icon, text, mode]) => {
			menu.push({
				icon,
				text,
				href: 'run.js',
				params: {
					run: true,
					fullscreen: true,
					mode
				}
			});
		});
	}
	if (info.running('run.js')) {
		const memory = info.local('run.js');

		if (memory && memory.url && memory.mode) {
			menu.push({
				icon: 'fa-solid fa-rocket',
				text: 'Open UI (' + memory.mode + ')',
				href: memory.url,
				params: {
					run: true,
					fullscreen: true
				}
			});
		}
		menu.push({
			icon: 'fa-solid fa-desktop',
			text: 'CLI',
			href: 'run.js',
			params: {
				run: true,
				fullscreen: true
			}
		});
	} else {
		[
			['fa-solid fa-plug', 'Install', 'install.js'],
			['fa-solid fa-rotate', 'Reinstall', 'install.js'],
			['fa-solid fa-arrow-rotate-right', 'Update', 'update.js'],
			['fa-solid fa-trash', 'Reset', 'reset.js']
		].forEach(([icon, text, href]) => {
			menu.push({
				icon,
				text,
				href,
				params: {
					run: true,
					fullscreen: true
				}
			});
		});
	}

	return menu;
};