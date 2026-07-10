/**
 * MapStyles — Shared map engine for Intel Globe apps
 * Extracted from map-viewer.html (source of truth)
 * Provides: tile URLs, style generation, color palettes, paint application, style switching
 */
(function() {
    'use strict';

    var HOSTNAME = window.location.host;
    var PROTOCOL = window.location.protocol;

    var TILESERVER = PROTOCOL + '//' + window.location.hostname + ':8090';
    var TILES_URL = TILESERVER + '/data/maplibre/{z}/{x}/{y}.pbf';
    var ASIA_TILES = TILESERVER + '/data/asia/{z}/{x}/{y}.pbf';
    var AFRICA_TILES = TILESERVER + '/data/africa/{z}/{x}/{y}.pbf';
    var EUROPE_TILES = TILESERVER + '/data/europe/{z}/{x}/{y}.pbf';
    var NA_TILES = TILESERVER + '/data/north-america/{z}/{x}/{y}.pbf';
    var SA_TILES = TILESERVER + '/data/south-america/{z}/{x}/{y}.pbf';
    var OCEANIA_TILES = TILESERVER + '/data/oceania/{z}/{x}/{y}.pbf';
    var ANTARCTICA_TILES = TILESERVER + '/data/antarctica/{z}/{x}/{y}.pbf';
    var GLYPHS_URL = TILESERVER + '/fonts/{fontstack}/{range}.pbf';

    var CONTINENTS = ['asia', 'africa', 'europe', 'na', 'sa', 'oceania', 'antarctica'];

    var _map = null;
    var _lang = 'en';
    var _onStyleChange = null;
    var _currentStyle = 'dark';
    var _osmRasterStyle = null;
    var _osmVectorStyle = null;

    // Styles that require a full setStyle swap (not just paint property changes)
    var FULL_SWAP_STYLES = { 'osm': true, 'osm-raster': true };

    function _nameExpr(lang) {
        if (lang === 'ar') {
            return ['coalesce', ['get', 'name:ar'], ['get', 'name:latin'], ['get', 'name']];
        }
        return ['coalesce', ['get', 'name:latin'], ['get', 'name']];
    }

    function getStyle(lang) {
        lang = lang || _lang;
        var nameField = _nameExpr(lang);
        var contSources = { asia: 'asia', africa: 'africa', europe: 'europe', na: 'north-america', sa: 'south-america', oceania: 'oceania', antarctica: 'antarctica' };
        var nobridge = ['any', ['!has', 'brunnel'], ['==', 'brunnel', 'tunnel']];
        var isbridge = ['==', 'brunnel', 'bridge'];
        var defaultRdColor = ['match', ['get', 'class'], 'motorway', '#00f5ff', 'trunk', '#d35400', 'primary', '#f39c12', 'secondary', '#7f8c8d', 'bridge', '#8a9aaa', '#4a5a6a'];
        var defaultRdWidth = ['interpolate', ['linear'], ['zoom'], 7, 0.5, 14, ['match', ['get', 'class'], 'motorway', 4, 'trunk', 3, 'primary', 2, 'bridge', 3, 1]];
        var defaultCasingWidth = ['interpolate', ['linear'], ['zoom'], 7, 1.5, 14, ['match', ['get', 'class'], 'motorway', 6, 'trunk', 5, 'primary', 4, 'bridge', 5, 3]];
        var placeLabelSize = ['interpolate', ['linear'], ['zoom'],
            3, ['match', ['get', 'class'], 'country', 10, 'state', 9, 0],
            5, ['match', ['get', 'class'], 'country', 14, 'state', 11, 'city', 10, 0],
            8, ['match', ['get', 'class'], 'country', 16, 'state', 13, 'city', 14, 'town', 11, 0],
            12, ['match', ['get', 'class'], 'city', 18, 'town', 14, 'village', 12, 'suburb', 11, 10],
            16, ['match', ['get', 'class'], 'city', 22, 'town', 18, 'village', 15, 'suburb', 13, 12]
        ];
        var layers = [
            { id: 'bg', type: 'background', paint: { 'background-color': ['interpolate', ['linear'], ['zoom'], 0, '#091428', 6, '#091428', 8, '#1a2540'] } },
            { id: 'countries-base', type: 'fill', source: 'maplibre', 'source-layer': 'countries', paint: { 'fill-color': '#1a2540', 'fill-opacity': ['interpolate', ['linear'], ['zoom'], 6, 1, 8, 0] } },
            { id: 'countries-highlight', type: 'fill', source: 'maplibre', 'source-layer': 'countries', maxzoom: 7, paint: { 'fill-color': '#00f5ff', 'fill-opacity': 0.8 }, filter: ['==', ['get', 'name'], '___NONE___'] },
            { id: 'border', type: 'line', source: 'maplibre', 'source-layer': 'countries', maxzoom: 14, paint: { 'line-color': '#4a5a8a', 'line-width': ['interpolate', ['linear'], ['zoom'], 1, 0.5, 4, 1, 7, 1.5, 10, 2] } },
            { id: 'border-highlight', type: 'line', source: 'maplibre', 'source-layer': 'countries', maxzoom: 14, paint: { 'line-color': '#00f5ff', 'line-width': 2 }, filter: ['==', ['get', 'name'], '___NONE___'] },
            { id: 'geo', type: 'line', source: 'maplibre', 'source-layer': 'geolines', maxzoom: 7, paint: { 'line-color': '#3a4a6a', 'line-width': 0.5, 'line-dasharray': [4, 4] } }
        ];
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-landuse', type: 'fill', source: contSources[c], 'source-layer': 'landuse', minzoom: 7, paint: { 'fill-color': 'rgba(0,0,0,0)', 'fill-opacity': 0.7 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-landcover', type: 'fill', source: contSources[c], 'source-layer': 'landcover', minzoom: 7, paint: { 'fill-color': ['match', ['get', 'class'], 'grass', '#1c2842', 'wood', '#162038', 'sand', '#1f2a42', '#1a2540'], 'fill-opacity': 0.7 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-park', type: 'fill', source: contSources[c], 'source-layer': 'park', minzoom: 10, paint: { 'fill-color': 'rgba(0,0,0,0)', 'fill-opacity': 0 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-water', type: 'fill', source: contSources[c], 'source-layer': 'water', minzoom: 4, paint: { 'fill-color': '#06101e' } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-waterway', type: 'line', source: contSources[c], 'source-layer': 'waterway', minzoom: 9, paint: { 'line-color': '#06101e', 'line-width': 1, 'line-opacity': 0 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-aeroway', type: 'fill', source: contSources[c], 'source-layer': 'aeroway', minzoom: 11, paint: { 'fill-color': 'rgba(0,0,0,0)', 'fill-opacity': 0 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-road-casing', type: 'line', source: contSources[c], 'source-layer': 'transportation', minzoom: 7, filter: nobridge, paint: { 'line-color': '#bbbbbb', 'line-width': defaultCasingWidth, 'line-opacity': 0 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-roads', type: 'line', source: contSources[c], 'source-layer': 'transportation', minzoom: 7, filter: nobridge, paint: { 'line-color': defaultRdColor, 'line-width': defaultRdWidth } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-bridge-casing', type: 'line', source: contSources[c], 'source-layer': 'transportation', minzoom: 10, filter: isbridge, paint: { 'line-color': '#bbbbbb', 'line-width': defaultCasingWidth, 'line-opacity': 0 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-bridge', type: 'line', source: contSources[c], 'source-layer': 'transportation', minzoom: 10, filter: isbridge, paint: { 'line-color': defaultRdColor, 'line-width': defaultRdWidth } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-buildings', type: 'fill', source: contSources[c], 'source-layer': 'building', minzoom: 12, paint: { 'fill-color': '#3a4a5a', 'fill-opacity': 0.8 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-boundary', type: 'line', source: contSources[c], 'source-layer': 'boundary', minzoom: 7, paint: { 'line-color': '#4a5a8a', 'line-width': ['interpolate', ['linear'], ['zoom'], 7, 1, 12, 2], 'line-dasharray': [2, 2] } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-water-labels', type: 'symbol', source: contSources[c], 'source-layer': 'water_name', minzoom: 10, layout: { 'text-field': nameField, 'text-font': ['Noto Sans Regular'], 'text-size': ['interpolate', ['linear'], ['zoom'], 10, 10, 16, 16], 'text-max-width': 8, visibility: 'none' }, paint: { 'text-color': '#4d80b3', 'text-halo-color': 'rgba(255,255,255,0.6)', 'text-halo-width': 2 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-aeroway-labels', type: 'symbol', source: contSources[c], 'source-layer': 'aerodrome_label', minzoom: 10, layout: { 'text-field': nameField, 'text-font': ['Noto Sans Regular'], 'text-size': ['interpolate', ['linear'], ['zoom'], 10, 10, 14, 13], 'text-max-width': 8, visibility: 'none' }, paint: { 'text-color': '#555555', 'text-halo-color': '#ffffff', 'text-halo-width': 2 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-road-labels', type: 'symbol', source: contSources[c], 'source-layer': 'transportation_name', minzoom: 12, layout: { 'text-field': nameField, 'text-font': ['Noto Sans Regular'], 'text-size': ['interpolate', ['linear'], ['zoom'], 12, 10, 16, 14], 'symbol-placement': 'line', 'text-rotation-alignment': 'map', 'text-max-angle': 30, visibility: 'none' }, paint: { 'text-color': '#333333', 'text-halo-color': '#ffffff', 'text-halo-width': 2 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-place-labels', type: 'symbol', source: contSources[c], 'source-layer': 'place', minzoom: 3, layout: { 'text-field': nameField, 'text-font': ['Noto Sans Regular'], 'text-size': placeLabelSize, 'text-max-width': 8, visibility: 'none' }, paint: { 'text-color': '#222222', 'text-halo-color': 'rgba(255,255,255,0.6)', 'text-halo-width': 2, 'text-halo-blur': 1 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-poi-labels', type: 'symbol', source: contSources[c], 'source-layer': 'poi', minzoom: 15, layout: { 'text-field': nameField, 'text-font': ['Noto Sans Regular'], 'text-size': 11, 'text-max-width': 8, 'text-anchor': 'top', 'text-offset': [0, 0.5], visibility: 'none' }, paint: { 'text-color': '#555555', 'text-halo-color': '#ffffff', 'text-halo-width': 1.5 } }); });
        CONTINENTS.forEach(function(c) { layers.push({ id: c + '-housenumber', type: 'symbol', source: contSources[c], 'source-layer': 'housenumber', minzoom: 17, layout: { 'text-field': ['get', 'housenumber'], 'text-font': ['Noto Sans Regular'], 'text-size': 10, visibility: 'none' }, paint: { 'text-color': '#666666', 'text-halo-color': '#ffffff', 'text-halo-width': 1 } }); });
        layers.push({
            id: 'country-labels', type: 'symbol', source: 'maplibre', 'source-layer': 'centroids', maxzoom: 7,
            layout: { 'text-field': ['get', 'name'], 'text-font': ['Noto Sans Regular'], 'text-size': ['interpolate', ['linear'], ['zoom'], 1, 10, 4, 14, 6, 16], 'text-transform': 'uppercase', 'text-letter-spacing': 0.1, 'text-max-width': 8, visibility: 'none' },
            paint: { 'text-color': '#e0e0e0', 'text-halo-color': '#0a0a0f', 'text-halo-width': 2, 'text-halo-blur': 1 }
        });
        // Abyss: basemap = DIRECT online CARTO raster tiles (the browser fetches
        // them straight from the CDN — no internal tileserver/proxy dependency).
        // dark_all matches the night theme; light mode swaps to light_all below.
        void TILES_URL; void GLYPHS_URL; void layers;
        return ABYSS_BASEMAP_STYLE(lang);
    }

    // Direct online basemap (CARTO). Picks dark/light from the active theme.
    function ABYSS_BASEMAP_STYLE(lang) {
        var theme = (document.documentElement.getAttribute('data-theme') === 'light') ? 'light_all' : 'dark_all';
        var bg = (theme === 'light_all') ? '#eef2f7' : '#0a0a0f';
        return {
            version: 8,
            sources: {
                'carto': {
                    type: 'raster',
                    tiles: [
                        'https://a.basemaps.cartocdn.com/' + theme + '/{z}/{x}/{y}.png',
                        'https://b.basemaps.cartocdn.com/' + theme + '/{z}/{x}/{y}.png',
                        'https://c.basemaps.cartocdn.com/' + theme + '/{z}/{x}/{y}.png'
                    ],
                    tileSize: 256,
                    maxzoom: 20,
                    attribution: '© OpenStreetMap contributors © CARTO'
                }
            },
            layers: [
                { id: 'background', type: 'background', paint: { 'background-color': bg } },
                { id: 'carto-layer', type: 'raster', source: 'carto' }
            ]
        };
    }

    function getStyleColors(preset) {
        var styles = {
            'dark': {
                bg: '#091428', land: '#1a2540', landOpacity: 1,
                border: '#3a5080', borderHighlight: '#00f5ff', geo: '#2a3a5a', water: '#06101e',
                landcover: { grass: '#1c2842', wood: '#162038', sand: '#1f2a42', default: '#1a2540' },
                roads: { motorway: '#00f5ff', trunk: '#d35400', primary: '#f39c12', secondary: '#7f8c8d', bridge: '#8a9aaa', default: '#4a5a6a' },
                buildings: '#3a4a5a', contBorder: '#4a5a8a',
                textColor: '#e0e0e0', textHalo: '#0a0a0f', labelColor: '#e0e0e0'
            },
            'light': {
                bg: '#d4e6f1', land: '#f5f0e1', landOpacity: 1,
                border: '#8a7a5a', borderHighlight: '#0066cc', geo: '#bbb', water: '#a3c4d9',
                landcover: { grass: '#c8ddb5', wood: '#a8c89a', sand: '#e8ddb5', default: '#f5f0e1' },
                roads: { motorway: '#0066cc', trunk: '#cc6600', primary: '#cc9900', secondary: '#999', bridge: '#999', default: '#bbb' },
                buildings: '#d5c8b0', contBorder: '#8a7a5a',
                textColor: '#333', textHalo: '#fff', labelColor: '#333'
            },
            'contrast': {
                bg: '#000020', land: '#0a0a2a', landOpacity: 1,
                border: '#00f5ff', borderHighlight: '#ff00ff', geo: '#003366', water: '#000044',
                landcover: { grass: '#0a0a35', wood: '#060628', sand: '#0f0f2a', default: '#0a0a2a' },
                roads: { motorway: '#00ffff', trunk: '#ff6600', primary: '#ffcc00', secondary: '#aaaaff', bridge: '#aaaaff', default: '#4444aa' },
                buildings: '#1a1a4a', contBorder: '#00f5ff',
                textColor: '#ffffff', textHalo: '#000020', labelColor: '#ffffff'
            },
            'terrain': {
                bg: '#0a1a15', land: '#1a2a1a', landOpacity: 0.95,
                border: '#4a6a4a', borderHighlight: '#88cc44', geo: '#2a3a2a', water: '#0f2a3a',
                landcover: { grass: '#2a4a2a', wood: '#1a3a1a', sand: '#4a3a1a', default: '#1a2a1a' },
                roads: { motorway: '#88cc44', trunk: '#c87533', primary: '#bba033', secondary: '#6a7a5a', bridge: '#8a9a6a', default: '#3a4a3a' },
                buildings: '#2a3a2a', contBorder: '#4a6a4a',
                textColor: '#c0d0c0', textHalo: '#0a1a15', labelColor: '#c0d0c0'
            },
            'midnight': {
                bg: '#0d1117', land: '#161b22', landOpacity: 1,
                border: '#30363d', borderHighlight: '#58a6ff', geo: '#21262d', water: '#040810',
                landcover: { grass: '#1a2332', wood: '#121a28', sand: '#1e2230', default: '#161b22' },
                roads: { motorway: '#58a6ff', trunk: '#d29922', primary: '#e3b341', secondary: '#484f58', bridge: '#6e7681', default: '#30363d' },
                buildings: '#21262d', contBorder: '#30363d',
                textColor: '#c9d1d9', textHalo: '#0d1117', labelColor: '#c9d1d9'
            },
            'ocean': {
                bg: '#03111e', land: '#0a192f', landOpacity: 1,
                border: '#1a3a5c', borderHighlight: '#64ffda', geo: '#112240', water: '#020c1b',
                landcover: { grass: '#0d2137', wood: '#091a2e', sand: '#142030', default: '#0a192f' },
                roads: { motorway: '#64ffda', trunk: '#f78166', primary: '#ffa657', secondary: '#1a3a5c', bridge: '#3a5a7c', default: '#112240' },
                buildings: '#112240', contBorder: '#1a3a5c',
                textColor: '#8892b0', textHalo: '#03111e', labelColor: '#8892b0'
            },
            'sand': {
                bg: '#1a150f', land: '#2a1f14', landOpacity: 1,
                border: '#5a4a3a', borderHighlight: '#e6a860', geo: '#3d2b1f', water: '#0f1520',
                landcover: { grass: '#2a2618', wood: '#22200f', sand: '#3a2e1a', default: '#2a1f14' },
                roads: { motorway: '#e6a860', trunk: '#c17838', primary: '#a89070', secondary: '#5a4a3a', bridge: '#7a6a5a', default: '#3d2b1f' },
                buildings: '#3d2b1f', contBorder: '#5a4a3a',
                textColor: '#c8b898', textHalo: '#1a150f', labelColor: '#c8b898'
            },
            'tactical': {
                bg: '#111111', land: '#1a1a1a', landOpacity: 1,
                border: '#444444', borderHighlight: '#00ff41', geo: '#2a2a2a', water: '#0a0a0a',
                landcover: { grass: '#1e1e1e', wood: '#161616', sand: '#222222', default: '#1a1a1a' },
                roads: { motorway: '#00ff41', trunk: '#ff6b35', primary: '#cccccc', secondary: '#555555', bridge: '#666666', default: '#333333' },
                buildings: '#2a2a2a', contBorder: '#444444',
                textColor: '#00ff41', textHalo: '#111111', labelColor: '#00ff41'
            },
            'osm': {
                bg: '#aad3df', land: '#f2efe9', landOpacity: 1,
                border: '#8d618b', borderHighlight: '#0066cc', geo: '#cccccc', water: '#aad3df',
                landcover: { grass: '#cdebb0', wood: '#add19e', sand: '#f5e9c6', farmland: '#eef0d5', scrub: '#c8d7ab', wetland: '#add19e', ice: '#ddecec', bare_rock: '#eee5dc', default: '#f2efe9' },
                landuse: { park: '#c8facc', forest: '#add19e', residential: '#e0dfdf', farmland: '#eef0d5', cemetery: '#aacbaf', industrial: '#ebdbe8', commercial: '#f2dad9', retail: '#ffd6d1', military: '#ff5555', school: '#ffffe5', university: '#ffffe5', kindergarten: '#ffffe5', hospital: '#ffffe5', quarry: '#c5c3c3', pitch: '#88e0be', playground: '#d0fcd4', railway: '#ebdbe8', parking: '#eeeeee', garages: '#dfddce', allotments: '#c9e1bf', construction: '#c7c7b4', orchard: '#aedfa3', vineyard: '#aedfa3', default: 'rgba(0,0,0,0)' },
                roads: { motorway: '#e892a2', trunk: '#f9b29c', primary: '#fcd6a4', secondary: '#f7fabf', tertiary: '#ffffff', residential: '#ffffff', service: '#ffffff', bridge: '#ffffff', default: '#ffffff' },
                roadCasing: { motorway: '#dc2a67', trunk: '#c84e2f', primary: '#a06b00', secondary: '#707d05', tertiary: '#8f8f8f', residential: '#bbbbbb', service: '#bbbbbb', bridge: '#000000', default: '#bbbbbb' },
                waterway: '#aad3df',
                buildings: '#d9d0c9', buildingOutline: '#b8a99d',
                contBorder: '#8d618b',
                aeroway: '#bbbbcc', apron: '#dadae0',
                textColor: '#222222', textHalo: 'rgba(255,255,255,0.6)', labelColor: '#222222',
                placeColor: '#222222', placeHalo: 'rgba(255,255,255,0.6)',
                roadLabelColor: '#000000', roadLabelHalo: '#ffffff',
                waterLabelColor: '#4d80b3',
                bridgeCasing: '#000000'
            },
            'positron': { bg: '#d4e6f1', land: '#f5f0e1', landOpacity: 1, border: '#8a7a5a', geo: '#bbb', water: '#a3c4d9', landcover: { grass: '#c8ddb5', wood: '#a8c89a', sand: '#e8ddb5', default: '#f5f0e1' }, roads: { motorway: '#0066cc', trunk: '#cc6600', primary: '#cc9900', secondary: '#999', default: '#bbb' }, buildings: '#d5c8b0', contBorder: '#8a7a5a', textColor: '#333', textHalo: '#fff' },
            'transport': { bg: '#b5d0d0', land: '#f2efe9', landOpacity: 1, border: '#8a8a8a', geo: '#bbb', water: '#b5d0d0', landcover: { grass: '#d3e7b6', wood: '#b5d29c', sand: '#f0e3c8', default: '#f2efe9' }, roads: { motorway: '#006cd9', trunk: '#0c9748', primary: '#d6002a', secondary: '#f57e00', default: '#999' }, buildings: '#dddcdb', contBorder: '#8a8a8a', textColor: '#333', textHalo: '#fff' },
            'cyclosm': { bg: '#aad3df', land: '#f5f5dc', landOpacity: 1, border: '#9e9cab', geo: '#bbb', water: '#aad3df', landcover: { grass: '#c8e6a0', wood: '#a0cf85', sand: '#f0e3c8', default: '#f5f5dc' }, roads: { motorway: '#e892a2', trunk: '#f9b29c', primary: '#fcd6a4', secondary: '#90c040', default: '#bbb' }, buildings: '#d9d0c9', contBorder: '#9e9cab', textColor: '#333', textHalo: '#fff' },
            'topo': { bg: '#b5d0d0', land: '#f2efe9', landOpacity: 1, border: '#7a7a7a', geo: '#aaa', water: '#aad3df', landcover: { grass: '#d4e6c3', wood: '#8bbc68', sand: '#f5e9c6', default: '#f2efe9' }, roads: { motorway: '#e66f00', trunk: '#e66f00', primary: '#dd0000', secondary: '#ee9900', default: '#999' }, buildings: '#d1c4b0', contBorder: '#7a7a7a', textColor: '#333', textHalo: '#fff' },
            'humanitarian': { bg: '#b5d0d0', land: '#f0e6e6', landOpacity: 1, border: '#8a8a8a', geo: '#bbb', water: '#aad3df', landcover: { grass: '#d4e6c3', wood: '#b8d5a7', sand: '#f0e3c8', default: '#f0e6e6' }, roads: { motorway: '#c84e4e', trunk: '#d35400', primary: '#c87533', secondary: '#999', default: '#bbb' }, buildings: '#dfb5b5', contBorder: '#8a8a8a', textColor: '#333', textHalo: '#fff' },
            'dark-matter': { bg: '#000020', land: '#0a0a2a', landOpacity: 1, border: '#00f5ff', geo: '#003366', water: '#000044', landcover: { grass: '#002200', wood: '#001a00', sand: '#1a1a00', default: '#0a0a2a' }, roads: { motorway: '#00ffff', trunk: '#ff6600', primary: '#ffcc00', secondary: '#aaaaff', default: '#4444aa' }, buildings: '#1a1a4a', contBorder: '#00f5ff', textColor: '#ffffff', textHalo: '#000020' },
            'omt': { bg: '#c8e8f0', land: '#f0ede4', landOpacity: 1, border: '#8888aa', geo: '#aab', water: '#b6dce8', landcover: { grass: '#c4e2a3', wood: '#a8c88a', sand: '#ede8c8', default: '#f0ede4' }, roads: { motorway: '#ff6633', trunk: '#e68a00', primary: '#ffd700', secondary: '#c8c800', default: '#bbb' }, buildings: '#d4cec0', contBorder: '#8888aa', textColor: '#444', textHalo: '#fff' }
        };
        return styles[preset] || styles['dark'];
    }

    function applyVectorPaint(c, preset) {
        if (!_map) return;
        var map = _map;
        var isOsm = (preset === 'osm');

        // Background
        map.setPaintProperty('bg', 'background-color', ['interpolate', ['linear'], ['zoom'], 0, c.bg, 6, c.bg, 8, c.land]);

        // Base world layers
        map.setPaintProperty('countries-base', 'fill-color', c.land);
        map.setPaintProperty('countries-base', 'fill-opacity', ['interpolate', ['linear'], ['zoom'], 6, c.landOpacity, 8, 0]);
        try { map.setPaintProperty('countries-highlight', 'fill-color', c.borderHighlight || c.border); } catch(e) {}
        map.setPaintProperty('border', 'line-color', c.border);
        try { map.setPaintProperty('border-highlight', 'line-color', c.borderHighlight || c.border); } catch(e) {}
        map.setPaintProperty('geo', 'line-color', c.geo);

        // Country labels
        try {
            map.setPaintProperty('country-labels', 'text-color', c.textColor);
            map.setPaintProperty('country-labels', 'text-halo-color', c.textHalo);
        } catch(e) {}

        // Build expressions
        var lcExpr = isOsm
            ? ['match', ['get', 'class'], 'grass', c.landcover.grass, 'wood', c.landcover.wood, 'sand', c.landcover.sand, 'farmland', c.landcover.farmland, 'scrub', c.landcover.scrub, 'wetland', c.landcover.wetland, 'ice', c.landcover.ice, 'bare_rock', c.landcover.bare_rock, c.landcover.default]
            : ['match', ['get', 'class'], 'grass', c.landcover.grass, 'wood', c.landcover.wood, 'sand', c.landcover.sand, c.landcover.default];
        var rdColorExpr = isOsm
            ? ['match', ['get', 'class'], 'motorway', c.roads.motorway, 'trunk', c.roads.trunk, 'primary', c.roads.primary, 'secondary', c.roads.secondary, 'tertiary', c.roads.tertiary, 'residential', c.roads.residential, 'service', c.roads.service, c.roads.default]
            : ['match', ['get', 'class'], 'motorway', c.roads.motorway, 'trunk', c.roads.trunk, 'primary', c.roads.primary, 'secondary', c.roads.secondary, 'bridge', c.roads.bridge || c.roads.secondary, c.roads.default];
        var luExpr = isOsm
            ? ['match', ['get', 'class'], 'park', c.landuse.park, 'forest', c.landuse.forest, 'residential', c.landuse.residential, 'farmland', c.landuse.farmland, 'cemetery', c.landuse.cemetery, 'industrial', c.landuse.industrial, 'commercial', c.landuse.commercial, 'retail', c.landuse.retail, 'military', c.landuse.military, 'school', c.landuse.school, 'university', c.landuse.university, 'kindergarten', c.landuse.kindergarten, 'hospital', c.landuse.hospital, 'quarry', c.landuse.quarry, 'pitch', c.landuse.pitch, 'playground', c.landuse.playground, 'railway', c.landuse.railway, 'parking', c.landuse.parking, 'garages', c.landuse.garages, 'allotments', c.landuse.allotments, 'construction', c.landuse.construction, 'orchard', c.landuse.orchard, 'vineyard', c.landuse.vineyard, c.landuse.default]
            : 'rgba(0,0,0,0)';
        var rcExpr = isOsm
            ? ['match', ['get', 'class'], 'motorway', c.roadCasing.motorway, 'trunk', c.roadCasing.trunk, 'primary', c.roadCasing.primary, 'secondary', c.roadCasing.secondary, 'tertiary', c.roadCasing.tertiary, 'residential', c.roadCasing.residential, 'service', c.roadCasing.service, c.roadCasing.default]
            : '#bbbbbb';
        var bcExpr = isOsm ? c.bridgeCasing || '#000000' : rcExpr;

        var rdWidthExpr = isOsm
            ? ['interpolate', ['exponential', 1.2], ['zoom'],
                5, 0,
                7, ['match', ['get', 'class'], 'motorway', 0.8, 'trunk', 0.6, 0],
                9, ['match', ['get', 'class'], 'motorway', 1.4, 'trunk', 1.4, 'primary', 1.4, 'secondary', 1, 0],
                12, ['match', ['get', 'class'], 'motorway', 3.5, 'trunk', 3.5, 'primary', 3.5, 'secondary', 3.5, 'tertiary', 2.5, 'residential', 0.5, 0.5],
                13, ['match', ['get', 'class'], 'motorway', 6, 'trunk', 6, 'primary', 5, 'secondary', 5, 'tertiary', 4, 'residential', 2.5, 'service', 2, 2],
                15, ['match', ['get', 'class'], 'motorway', 10, 'trunk', 10, 'primary', 10, 'secondary', 9, 'tertiary', 9, 'residential', 5, 'service', 3.5, 3],
                17, ['match', ['get', 'class'], 'motorway', 18, 'trunk', 18, 'primary', 18, 'secondary', 18, 'tertiary', 18, 'residential', 12, 'service', 7, 7]]
            : ['interpolate', ['linear'], ['zoom'], 7, 0.5, 14, ['match', ['get', 'class'], 'motorway', 3, 'trunk', 2.5, 'primary', 2, 'secondary', 1.5, 'bridge', 3, 1]];
        var rcWidthExpr = isOsm
            ? ['interpolate', ['exponential', 1.2], ['zoom'],
                5, 0.4,
                7, ['match', ['get', 'class'], 'motorway', 1.6, 'trunk', 1.4, 0.4],
                9, ['match', ['get', 'class'], 'motorway', 2.4, 'trunk', 2.4, 'primary', 2.4, 'secondary', 2, 0.4],
                12, ['match', ['get', 'class'], 'motorway', 5.5, 'trunk', 5.5, 'primary', 5.5, 'secondary', 5.5, 'tertiary', 4.5, 'residential', 2.5, 2.5],
                13, ['match', ['get', 'class'], 'motorway', 8, 'trunk', 8, 'primary', 7, 'secondary', 7, 'tertiary', 6, 'residential', 4.5, 'service', 4, 4],
                15, ['match', ['get', 'class'], 'motorway', 12, 'trunk', 12, 'primary', 12, 'secondary', 11, 'tertiary', 11, 'residential', 7, 'service', 5.5, 5],
                17, ['match', ['get', 'class'], 'motorway', 22, 'trunk', 22, 'primary', 22, 'secondary', 22, 'tertiary', 22, 'residential', 16, 'service', 11, 11]]
            : ['interpolate', ['linear'], ['zoom'], 7, 1.5, 14, ['match', ['get', 'class'], 'motorway', 6, 'trunk', 5, 'primary', 4, 'bridge', 5, 3]];

        CONTINENTS.forEach(function(cont) {
            try {
                map.setPaintProperty(cont + '-landcover', 'fill-color', lcExpr);
                map.setPaintProperty(cont + '-landuse', 'fill-color', luExpr);
                map.setPaintProperty(cont + '-landuse', 'fill-opacity', isOsm
                    ? ['match', ['get', 'class'], 'military', 0.4, 'residential', 0.5, 0.7]
                    : 0);
                map.setPaintProperty(cont + '-park', 'fill-color', isOsm ? c.landuse.park : 'rgba(0,0,0,0)');
                map.setPaintProperty(cont + '-park', 'fill-opacity', isOsm ? 0.6 : 0);
                map.setPaintProperty(cont + '-water', 'fill-color', c.water);
                map.setPaintProperty(cont + '-waterway', 'line-color', c.waterway || c.water);
                map.setPaintProperty(cont + '-waterway', 'line-opacity', isOsm ? 1 : 0);
                map.setPaintProperty(cont + '-waterway', 'line-width', isOsm
                    ? ['interpolate', ['exponential', 1.2], ['zoom'], 11, 0.5, 20, 6]
                    : 1);
                map.setPaintProperty(cont + '-aeroway', 'fill-color', isOsm ? (c.aeroway || '#bbbbcc') : 'rgba(0,0,0,0)');
                map.setPaintProperty(cont + '-aeroway', 'fill-opacity', isOsm ? 1 : 0);
                map.setPaintProperty(cont + '-roads', 'line-color', rdColorExpr);
                map.setPaintProperty(cont + '-roads', 'line-width', rdWidthExpr);
                map.setPaintProperty(cont + '-road-casing', 'line-color', rcExpr);
                map.setPaintProperty(cont + '-road-casing', 'line-width', rcWidthExpr);
                map.setPaintProperty(cont + '-road-casing', 'line-opacity', isOsm ? 1 : 0);
                map.setPaintProperty(cont + '-bridge', 'line-color', rdColorExpr);
                map.setPaintProperty(cont + '-bridge', 'line-width', rdWidthExpr);
                map.setPaintProperty(cont + '-bridge-casing', 'line-color', bcExpr);
                map.setPaintProperty(cont + '-bridge-casing', 'line-width', rcWidthExpr);
                map.setPaintProperty(cont + '-bridge-casing', 'line-opacity', isOsm ? 1 : 0);
                map.setPaintProperty(cont + '-buildings', 'fill-color', c.buildings);
                map.setPaintProperty(cont + '-buildings', 'fill-outline-color', isOsm ? (c.buildingOutline || c.buildings) : c.buildings);
                map.setPaintProperty(cont + '-boundary', 'line-color', c.contBorder);
                map.setPaintProperty(cont + '-boundary', 'line-width', isOsm
                    ? ['interpolate', ['linear'], ['zoom'], 3, 1, 5, 1.2, 12, 3]
                    : 1);
                map.setPaintProperty(cont + '-boundary', 'line-dasharray', isOsm ? [6, 2, 2, 2] : [2, 2]);
                map.setLayoutProperty(cont + '-water-labels', 'visibility', isOsm ? 'visible' : 'none');
                map.setPaintProperty(cont + '-water-labels', 'text-color', isOsm ? (c.waterLabelColor || '#4d80b3') : '#4d80b3');
                map.setLayoutProperty(cont + '-aeroway-labels', 'visibility', isOsm ? 'visible' : 'none');
                map.setLayoutProperty(cont + '-road-labels', 'visibility', isOsm ? 'visible' : 'none');
                map.setPaintProperty(cont + '-road-labels', 'text-color', isOsm ? '#000000' : '#333333');
                map.setPaintProperty(cont + '-road-labels', 'text-halo-color', isOsm
                    ? ['match', ['get', 'class'], 'motorway', '#e892a2', 'trunk', '#f9b29c', 'primary', '#fcd6a4', 'secondary', '#f7fabf', '#ffffff']
                    : '#ffffff');
                map.setLayoutProperty(cont + '-place-labels', 'visibility', isOsm ? 'visible' : 'none');
                map.setPaintProperty(cont + '-place-labels', 'text-color', isOsm ? (c.placeColor || '#222222') : '#222222');
                map.setPaintProperty(cont + '-place-labels', 'text-halo-color', isOsm ? (c.placeHalo || 'rgba(255,255,255,0.6)') : 'rgba(255,255,255,0.6)');
                map.setLayoutProperty(cont + '-poi-labels', 'visibility', isOsm ? 'visible' : 'none');
                map.setLayoutProperty(cont + '-housenumber', 'visibility', isOsm ? 'visible' : 'none');
            } catch(e) {}
        });
    }

    function loadOsmRasterStyle() {
        var map = _map;
        var center = map.getCenter();
        var zoom = map.getZoom();
        if (!_osmRasterStyle) {
            _osmRasterStyle = {
                version: 8,
                glyphs: GLYPHS_URL,
                sources: {
                    'osm-raster': {
                        type: 'raster',
                        tiles: [
                            PROTOCOL + '//' + HOSTNAME + '/raster-tiles/{z}/{x}/{y}.png'
                        ],
                        tileSize: 256,
                        maxzoom: 19
                    }
                },
                layers: [
                    { id: 'osm-raster-layer', type: 'raster', source: 'osm-raster' }
                ]
            };
        }
        map.setStyle(_osmRasterStyle, { diff: false });
        map.once('style.load', function() {
            map.setCenter(center);
            map.setZoom(zoom);
        });
    }

    function loadOsmVectorStyle() {
        var map = _map;
        var center = map.getCenter();
        var zoom = map.getZoom();
        var bearing = map.getBearing();
        var pitch = map.getPitch();

        if (!_osmVectorStyle) {
            if (typeof OsmStyleFull === 'undefined') {
                console.warn('OsmStyleFull not loaded — falling back to paint swap');
                var c = getStyleColors('osm');
                if (c) applyVectorPaint(c, 'osm');
                return;
            }
            _osmVectorStyle = OsmStyleFull.generate({
                glyphs: GLYPHS_URL,
                sources: {
                    maplibre: { type: 'vector', tiles: [TILES_URL], maxzoom: 6 },
                    asia: { type: 'vector', tiles: [ASIA_TILES], minzoom: 4, maxzoom: 14 },
                    africa: { type: 'vector', tiles: [AFRICA_TILES], minzoom: 4, maxzoom: 14 },
                    europe: { type: 'vector', tiles: [EUROPE_TILES], minzoom: 4, maxzoom: 14 },
                    'north-america': { type: 'vector', tiles: [NA_TILES], minzoom: 4, maxzoom: 14 },
                    'south-america': { type: 'vector', tiles: [SA_TILES], minzoom: 4, maxzoom: 14 },
                    oceania: { type: 'vector', tiles: [OCEANIA_TILES], minzoom: 4, maxzoom: 14 },
                    antarctica: { type: 'vector', tiles: [ANTARCTICA_TILES], minzoom: 4, maxzoom: 14 }
                },
                nameExpr: _nameExpr(_lang)
            });
        }

        map.setStyle(_osmVectorStyle, { diff: false });
        map.once('style.load', function() {
            map.setCenter(center);
            map.setZoom(zoom);
            map.setBearing(bearing);
            map.setPitch(pitch);
        });
    }

    function switchMapStyle(preset) {
        if (!_map) return;
        var map = _map;
        var prevStyle = _currentStyle;
        _currentStyle = preset;
        var wasFullSwap = FULL_SWAP_STYLES[prevStyle];
        var isFullSwap = FULL_SWAP_STYLES[preset];

        if (preset === 'osm') {
            // Full OSM vector style swap
            loadOsmVectorStyle();
            if (_onStyleChange) _onStyleChange(preset, prevStyle, true);
        } else if (preset === 'osm-raster') {
            // Legacy raster-only OSM (pre-rendered PNGs)
            loadOsmRasterStyle();
            if (_onStyleChange) _onStyleChange(preset, prevStyle, true);
        } else if (wasFullSwap) {
            // Switching away from a full-swap style back to the base vector style
            var center = map.getCenter();
            var zoom = map.getZoom();
            var bearing = map.getBearing();
            var pitch = map.getPitch();
            map.setStyle(getStyle(_lang), { diff: false });
            map.once('style.load', function() {
                map.setCenter(center);
                map.setZoom(zoom);
                map.setBearing(bearing);
                map.setPitch(pitch);
                var c = getStyleColors(preset);
                if (c) applyVectorPaint(c, preset);
                if (_onStyleChange) _onStyleChange(preset, prevStyle, true);
            });
        } else {
            // Normal paint-property swap on existing base style
            var c = getStyleColors(preset);
            if (c) applyVectorPaint(c, preset);
            if (_onStyleChange) _onStyleChange(preset, prevStyle, false);
        }
    }

    function init(options) {
        _map = options.map;
        _lang = options.lang || 'en';
        _onStyleChange = options.onStyleChange || null;
        if (options.initialStyle) _currentStyle = options.initialStyle;

        // Auto-load RTL text plugin
        if (typeof maplibregl !== 'undefined' && maplibregl.setRTLTextPlugin) {
            try { maplibregl.setRTLTextPlugin('/libs/mapbox-gl-rtl-text.min.js', true); } catch(e) {}
        }
    }

    window.MapStyles = {
        TILESERVER: TILESERVER,
        TILES_URL: TILES_URL,
        GLYPHS_URL: GLYPHS_URL,
        CONTINENTS: CONTINENTS,

        init: init,
        getStyle: getStyle,
        getStyleColors: getStyleColors,
        applyVectorPaint: applyVectorPaint,
        switchMapStyle: switchMapStyle,
        loadOsmRasterStyle: loadOsmRasterStyle,
        loadOsmVectorStyle: loadOsmVectorStyle,

        get currentStyle() { return _currentStyle; },
        set currentStyle(v) { _currentStyle = v; }
    };
})();
