        (function() {
            const root = document.documentElement;
            const storageKey = 'theme';
            const systemMedia = window.matchMedia('(prefers-color-scheme: dark)');

            const getStoredPreference = () => localStorage.getItem(storageKey) || 'auto';

            const resolveTheme = preference => {
                if (preference === 'light' || preference === 'dark') {
                    return preference;
                }
                return systemMedia.matches ? 'dark' : 'light';
            };

            const applyTheme = preference => {
                const effectiveTheme = resolveTheme(preference);
                root.setAttribute('data-bs-theme', effectiveTheme);
                root.dataset.themePreference = preference;
            };

            const handleSystemChange = () => {
                const preference = root.dataset.themePreference || getStoredPreference();
                if (preference === 'auto') {
                    applyTheme('auto');
                }
            };

            if (typeof systemMedia.addEventListener === 'function') {
                systemMedia.addEventListener('change', handleSystemChange);
            } else if (typeof systemMedia.addListener === 'function') {
                systemMedia.addListener(handleSystemChange);
            }

            window.toggleTheme = function() {
                const currentPreference = root.dataset.themePreference || getStoredPreference();
                let nextPreference = 'auto';
                if (currentPreference === 'auto') {
                    nextPreference = 'light';
                } else if (currentPreference === 'light') {
                    nextPreference = 'dark';
                }
                localStorage.setItem(storageKey, nextPreference);
                applyTheme(nextPreference);
            };

            applyTheme(getStoredPreference());
        })();
