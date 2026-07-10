/**
 * Classification Banner — fetches the platform default classification
 * level from /api/settings/public and updates any .classification-banner element.
 * Falls back to the existing text if API is unavailable.
 */
(function() {
    var banners = document.querySelectorAll('.classification-banner');
    if (!banners.length) return;

    fetch('/api/settings/public')
        .then(function(r) { return r.ok ? r.json() : null; })
        .then(function(data) {
            if (!data || !data.default_classification) return;
            var level = data.default_classification;
            banners.forEach(function(el) {
                el.textContent = level;
                // Update color based on level
                var colors = {
                    'UNCLASSIFIED': '#4caf50',
                    'CONFIDENTIAL': '#2196f3',
                    'SECRET': '#f44336',
                    'TOP SECRET': '#ff9800',
                    'TOP SECRET//SCI': '#ff5722'
                };
                if (colors[level]) {
                    el.style.backgroundColor = colors[level];
                    el.style.color = level === 'UNCLASSIFIED' ? '#fff' : '#fff';
                }
            });
        })
        .catch(function() { /* keep existing text */ });
})();
