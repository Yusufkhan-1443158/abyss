/**
 * OSM Bright Style Generator for Intel Globe
 * Produces a comprehensive MapLibre GL style that closely matches standard OpenStreetMap Carto.
 * Uses OpenMapTiles vector tile schema from continent-split mbtiles sources.
 */
(function() {
    'use strict';

    /**
     * Generate a full OSM Carto-like MapLibre GL style.
     * @param {object} opts
     * @param {string} opts.glyphs - Glyphs URL template
     * @param {object} opts.sources - MapLibre sources dict (maplibre, asia, europe, etc.)
     * @param {function} opts.nameExpr - Name expression for labels, e.g. ['coalesce', ['get','name:latin'], ['get','name']]
     * @returns {object} MapLibre GL style object
     */
    function generateOsmStyle(opts) {
        var glyphs = opts.glyphs;
        var sources = opts.sources;
        var nameExpr = opts.nameExpr || ['coalesce', ['get', 'name:latin'], ['get', 'name']];

        // All continent source keys (must match keys in sources dict)
        var CONT_SOURCES = ['asia', 'africa', 'europe', 'north-america', 'south-america', 'oceania', 'antarctica'];

        // Helpers
        function contLayer(idSuffix, def) {
            var result = [];
            CONT_SOURCES.forEach(function(src) {
                var layer = {};
                for (var k in def) { layer[k] = def[k]; }
                layer.id = src + '-' + idSuffix;
                layer.source = src;
                result.push(layer);
            });
            return result;
        }

        // Filters
        var isTunnel = ['all', ['==', ['get', 'brunnel'], 'tunnel']];
        var isBridge = ['==', ['get', 'brunnel'], 'bridge'];
        var notBridgeNotTunnel = ['all',
            ['any', ['!has', 'brunnel'], ['==', ['get', 'brunnel'], '']],
        ];
        var normalRoad = ['any', ['!has', 'brunnel'], ['==', ['get', 'brunnel'], ''], ['==', ['get', 'brunnel'], 'ford']];

        // Road width expressions (exponential, mimicking OSM Carto)
        var mwayFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0, 6, 0.5, 7, 0.8, 9, 1.4,
            11, 3, 13, 6, 15, 10, 17, 18, 20, 36];
        var trunkFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0, 7, 0.6, 9, 1.4,
            11, 2.5, 13, 5.5, 15, 9, 17, 17, 20, 34];
        var primaryFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0, 7, 0.3, 9, 1.2,
            11, 2, 13, 4.5, 15, 8, 17, 16, 20, 32];
        var secondaryFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            9, 0.5, 11, 1.8, 13, 4, 15, 7.5, 17, 15, 20, 30];
        var tertiaryFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            10, 0, 11, 1, 13, 3, 15, 6.5, 17, 13, 20, 26];
        var residentialFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            12, 0.5, 13, 2, 15, 5, 17, 10, 20, 20];
        var serviceFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            14, 0.5, 15, 2, 17, 4, 20, 10];

        // Casing = fill + 2px each side
        var mwayCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0.4, 6, 1, 7, 1.6, 9, 2.4,
            11, 5, 13, 8, 15, 12, 17, 22, 20, 40];
        var trunkCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0.4, 7, 1.4, 9, 2.4,
            11, 4.5, 13, 7.5, 15, 11, 17, 21, 20, 38];
        var primaryCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0.4, 7, 1, 9, 2,
            11, 4, 13, 6.5, 15, 10, 17, 20, 20, 36];
        var secondaryCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            9, 1.2, 11, 3.5, 13, 6, 15, 9.5, 17, 19, 20, 34];
        var tertiaryCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            10, 0.6, 11, 2.2, 13, 5, 15, 8.5, 17, 17, 20, 30];
        var residentialCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            12, 1.5, 13, 3.5, 15, 7, 17, 14, 20, 24];
        var serviceCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            14, 1.2, 15, 3.5, 17, 6, 20, 14];

        // Combined road fill width
        var roadFillWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0,
            7, ['match', ['get', 'class'], 'motorway', 0.8, 'trunk', 0.6, 'primary', 0.3, 0],
            9, ['match', ['get', 'class'], 'motorway', 1.4, 'trunk', 1.4, 'primary', 1.2, 'secondary', 0.5, 0],
            11, ['match', ['get', 'class'], 'motorway', 3, 'trunk', 2.5, 'primary', 2, 'secondary', 1.8, 'tertiary', 1, 0],
            13, ['match', ['get', 'class'], 'motorway', 6, 'trunk', 5.5, 'primary', 4.5, 'secondary', 4, 'tertiary', 3, 'minor', 2, 'service', 1, 2],
            15, ['match', ['get', 'class'], 'motorway', 10, 'trunk', 9, 'primary', 8, 'secondary', 7.5, 'tertiary', 6.5, 'minor', 5, 'service', 2, 'track', 1.5, 3],
            17, ['match', ['get', 'class'], 'motorway', 18, 'trunk', 17, 'primary', 16, 'secondary', 15, 'tertiary', 13, 'minor', 10, 'service', 4, 'track', 2, 7],
            20, ['match', ['get', 'class'], 'motorway', 36, 'trunk', 34, 'primary', 32, 'secondary', 30, 'tertiary', 26, 'minor', 20, 'service', 10, 'track', 4, 14]
        ];

        var roadCasingWidth = ['interpolate', ['exponential', 1.2], ['zoom'],
            5, 0.4,
            7, ['match', ['get', 'class'], 'motorway', 1.6, 'trunk', 1.4, 'primary', 1, 0.4],
            9, ['match', ['get', 'class'], 'motorway', 2.4, 'trunk', 2.4, 'primary', 2, 'secondary', 1.2, 0.4],
            11, ['match', ['get', 'class'], 'motorway', 5, 'trunk', 4.5, 'primary', 4, 'secondary', 3.5, 'tertiary', 2.2, 0.6],
            13, ['match', ['get', 'class'], 'motorway', 8, 'trunk', 7.5, 'primary', 6.5, 'secondary', 6, 'tertiary', 5, 'minor', 3.5, 'service', 2, 3.5],
            15, ['match', ['get', 'class'], 'motorway', 12, 'trunk', 11, 'primary', 10, 'secondary', 9.5, 'tertiary', 8.5, 'minor', 7, 'service', 3.5, 'track', 2.5, 5],
            17, ['match', ['get', 'class'], 'motorway', 22, 'trunk', 21, 'primary', 20, 'secondary', 19, 'tertiary', 17, 'minor', 14, 'service', 6, 'track', 3, 11],
            20, ['match', ['get', 'class'], 'motorway', 40, 'trunk', 38, 'primary', 36, 'secondary', 34, 'tertiary', 30, 'minor', 24, 'service', 14, 'track', 6, 18]
        ];

        // Road fill color
        var roadFillColor = ['match', ['get', 'class'],
            'motorway', '#e892a2',
            'trunk', '#f9b29c',
            'primary', '#fcd6a4',
            'secondary', '#f7fabf',
            'tertiary', '#ffffff',
            'minor', '#ffffff',
            'residential', '#ffffff',
            'service', '#ffffff',
            'track', '#f5f0e1',
            '#ffffff'];

        // Road casing color
        var roadCasingColor = ['match', ['get', 'class'],
            'motorway', '#dc2a67',
            'trunk', '#c84e2f',
            'primary', '#a06b00',
            'secondary', '#707d05',
            'tertiary', '#8f8f8f',
            'minor', '#999999',
            'residential', '#bbbbbb',
            'service', '#bbbbbb',
            'track', '#996600',
            '#bbbbbb'];

        // Tunnel fill lighter
        var tunnelFillColor = ['match', ['get', 'class'],
            'motorway', '#f2c0ca',
            'trunk', '#fcd4c8',
            'primary', '#fde8ca',
            'secondary', '#fbfdd7',
            '#e8e8e8'];

        var tunnelCasingColor = ['match', ['get', 'class'],
            'motorway', '#e892a2',
            'trunk', '#f9b29c',
            'primary', '#fcd6a4',
            'secondary', '#f7fabf',
            '#cccccc'];

        // Place label sizes (match OSM Carto)
        var placeLabelSize = ['interpolate', ['linear'], ['zoom'],
            2, ['match', ['get', 'class'], 'country', 10, 0],
            4, ['match', ['get', 'class'], 'country', 14, 'state', 9, 0],
            6, ['match', ['get', 'class'], 'country', 18, 'state', 12, 'city', 11, 0],
            8, ['match', ['get', 'class'], 'country', 20, 'state', 14, 'city', 14, 'town', 10, 0],
            10, ['match', ['get', 'class'], 'state', 14, 'city', 16, 'town', 12, 'village', 10, 0],
            12, ['match', ['get', 'class'], 'city', 20, 'town', 14, 'village', 12, 'suburb', 11, 'hamlet', 10, 10],
            14, ['match', ['get', 'class'], 'city', 22, 'town', 16, 'village', 14, 'suburb', 13, 'hamlet', 12, 'neighbourhood', 11, 11],
            16, ['match', ['get', 'class'], 'city', 24, 'town', 18, 'village', 16, 'suburb', 14, 'hamlet', 13, 'neighbourhood', 12, 12]
        ];

        // Road label sizes
        var roadLabelSize = ['interpolate', ['linear'], ['zoom'],
            10, ['match', ['get', 'class'], 'motorway', 8, 'trunk', 8, 'primary', 8, 0],
            12, ['match', ['get', 'class'], 'motorway', 10, 'trunk', 10, 'primary', 10, 'secondary', 9, 8],
            14, ['match', ['get', 'class'], 'motorway', 12, 'trunk', 12, 'primary', 11, 'secondary', 11, 'tertiary', 10, 9],
            16, ['match', ['get', 'class'], 'motorway', 14, 'trunk', 14, 'primary', 13, 'secondary', 12, 'tertiary', 12, 'minor', 11, 'residential', 11, 10],
            18, 14
        ];

        // Road label halo matches road fill
        var roadLabelHalo = ['match', ['get', 'class'],
            'motorway', '#e892a2',
            'trunk', '#f9b29c',
            'primary', '#fcd6a4',
            'secondary', '#f7fabf',
            '#ffffff'];

        // Waterway widths
        var waterwayWidth = ['interpolate', ['exponential', 1.3], ['zoom'],
            8, ['match', ['get', 'class'], 'river', 0.5, 0],
            10, ['match', ['get', 'class'], 'river', 1, 'canal', 0.5, 0],
            13, ['match', ['get', 'class'], 'river', 2, 'canal', 1.5, 'stream', 0.5, 'drain', 0.3, 0.3],
            16, ['match', ['get', 'class'], 'river', 5, 'canal', 4, 'stream', 2, 'drain', 1, 1],
            20, ['match', ['get', 'class'], 'river', 12, 'canal', 10, 'stream', 5, 'drain', 3, 3]
        ];

        // Boundary widths
        var adminWidth = ['interpolate', ['linear'], ['zoom'],
            2, ['match', ['get', 'admin_level'], 2, 0.5, 0],
            4, ['match', ['get', 'admin_level'], 2, 1, 4, 0.5, 0],
            6, ['match', ['get', 'admin_level'], 2, 1.5, 4, 1, 0.3],
            10, ['match', ['get', 'admin_level'], 2, 2.5, 4, 1.5, 6, 1, 0.5],
            14, ['match', ['get', 'admin_level'], 2, 3.5, 4, 2.5, 6, 1.5, 1]
        ];

        // --- BUILD LAYERS ---
        var layers = [];

        // 1. Background (land color)
        layers.push({
            id: 'background',
            type: 'background',
            paint: { 'background-color': '#f2efe9' }
        });

        // 2. Landcover (natural areas)
        contLayer('landcover', {
            type: 'fill',
            'source-layer': 'landcover',
            minzoom: 4,
            paint: {
                'fill-color': ['match', ['get', 'class'],
                    'grass', '#cdebb0',
                    'wood', '#add19e',
                    'sand', '#f5e9c6',
                    'farmland', '#eef0d5',
                    'scrub', '#c8d7ab',
                    'wetland', '#d5e6d5',
                    'ice', '#ddecec',
                    'bare_rock', '#eee5dc',
                    'rock', '#eee5dc',
                    '#f2efe9'
                ],
                'fill-opacity': ['interpolate', ['linear'], ['zoom'], 4, 0.3, 7, 0.6, 10, 0.8]
            }
        }).forEach(function(l) { layers.push(l); });

        // 3. Landuse
        contLayer('landuse', {
            type: 'fill',
            'source-layer': 'landuse',
            minzoom: 6,
            paint: {
                'fill-color': ['match', ['get', 'class'],
                    'park', '#c8facc',
                    'forest', '#add19e',
                    'residential', '#e0dfdf',
                    'farmland', '#eef0d5',
                    'cemetery', '#aacbaf',
                    'industrial', '#ebdbe8',
                    'commercial', '#f2dad9',
                    'retail', '#ffd6d1',
                    'military', '#f55a6e',
                    'school', '#ffffe5',
                    'university', '#ffffe5',
                    'kindergarten', '#ffffe5',
                    'hospital', '#ffffe5',
                    'quarry', '#c5c3c3',
                    'pitch', '#88e0be',
                    'playground', '#d0fcd4',
                    'railway', '#ebdbe8',
                    'parking', '#eeeeee',
                    'garages', '#dfddce',
                    'allotments', '#c9e1bf',
                    'construction', '#c7c7b4',
                    'orchard', '#aedfa3',
                    'vineyard', '#aedfa3',
                    'dam', '#aeaeb0',
                    'grass', '#cdebb0',
                    'meadow', '#cdebb0',
                    'village_green', '#cdebb0',
                    'recreation_ground', '#d0fcd4',
                    'heath', '#d6d99f',
                    'scrub', '#c8d7ab',
                    'sand', '#f5e9c6',
                    'beach', '#fff1ba',
                    'basin', '#aad3df',
                    'reservoir', '#aad3df',
                    'brownfield', '#c7c7b4',
                    'landfill', '#c7c7b4',
                    'plant_nursery', '#aedfa3',
                    'greenhouse_horticulture', '#aedfa3',
                    'aquaculture', '#b5d0d0',
                    'harbour', '#d5d5d5',
                    'bus_station', '#f0e0d0',
                    'marina', '#b5d0d0',
                    'pier', '#f2efe9',
                    'bridge', '#d5d5d5',
                    'track', '#ddddbb',
                    'stadium', '#d0fcd4',
                    'sports_centre', '#d0fcd4',
                    'swimming_pool', '#aad3df',
                    'water_park', '#aad3df',
                    'golf_course', '#b5e3b5',
                    'miniature_golf', '#b5e3b5',
                    'garden', '#cdebb0',
                    'nature_reserve', '#c8facc',
                    'national_park', '#c8facc',
                    'protected_area', '#c8facc',
                    'zoo', '#d0fcd4',
                    'theme_park', '#d0fcd4',
                    'dog_park', '#e0f8e0',
                    'camp_site', '#ccff99',
                    'caravan_site', '#ccff99',
                    'winter_sports', '#e0e0ff',
                    'common', '#cdebb0',
                    'place_of_worship', '#d0d0d0',
                    'wastewater_plant', '#d5d5d5',
                    'water_works', '#d5d5d5',
                    'power', '#d5d5d5',
                    'substation', '#d5d5d5',
                    'farmyard', '#ebd5c8',
                    'depot', '#d5d5d5',
                    'storage', '#d5d5d5',
                    'rest_area', '#efc8c8',
                    'fuel', '#f0e0d0',
                    'transportation', '#e8e0e0',
                    'taxi', '#e8e0e0',
                    'bicycle_parking', '#e8e0e0',
                    'motorcycle_parking', '#e8e0e0',
                    'ice_rink', '#ddecec',
                    'sports_hall', '#d0fcd4',
                    'fitness_centre', '#d0fcd4',
                    'college', '#ffffe5',
                    'library', '#ffffe5',
                    'arts_centre', '#ffffe5',
                    'museum', '#ffffe5',
                    'theatre', '#ffffe5',
                    'cinema', '#ffffe5',
                    'community_centre', '#ffffe5',
                    'social_facility', '#ffffe5',
                    'prison', '#f0dede',
                    'police', '#f0dede',
                    'fire_station', '#f0dede',
                    'courthouse', '#f0dede',
                    'embassy', '#f0dede',
                    'townhall', '#f0dede',
                    'post_office', '#f0dede',
                    'bank', '#f0dede',
                    'marketplace', '#ffd6d1',
                    'office', '#f0dede',
                    'shelter', '#f0dede',
                    '#f2efe9'
                ],
                'fill-opacity': ['match', ['get', 'class'],
                    'military', 0.35,
                    'residential', 0.5,
                    0.7
                ]
            }
        }).forEach(function(l) { layers.push(l); });

        // 4. Park fills (separate layer for parks from park source-layer)
        contLayer('park-fill', {
            type: 'fill',
            'source-layer': 'park',
            minzoom: 6,
            paint: {
                'fill-color': '#c8facc',
                'fill-opacity': ['interpolate', ['linear'], ['zoom'], 6, 0.2, 10, 0.5]
            }
        }).forEach(function(l) { layers.push(l); });

        // 4b. Park outlines
        contLayer('park-outline', {
            type: 'line',
            'source-layer': 'park',
            minzoom: 10,
            paint: {
                'line-color': '#6fbd6f',
                'line-width': 1,
                'line-opacity': 0.6,
                'line-dasharray': [4, 2]
            }
        }).forEach(function(l) { layers.push(l); });

        // 5. Water fills (ocean, lakes)
        contLayer('water', {
            type: 'fill',
            'source-layer': 'water',
            minzoom: 0,
            paint: {
                'fill-color': '#aad3df'
            }
        }).forEach(function(l) { layers.push(l); });

        // Also add maplibre overview water for z0-6
        // (maplibre source has 'countries' and 'geolines' layers but no water layer typically)

        // 6. Waterway lines (rivers, canals, streams)
        contLayer('waterway', {
            type: 'line',
            'source-layer': 'waterway',
            minzoom: 8,
            layout: { 'line-cap': 'round', 'line-join': 'round' },
            paint: {
                'line-color': '#aad3df',
                'line-width': waterwayWidth,
                'line-opacity': 1
            }
        }).forEach(function(l) { layers.push(l); });

        // 7. Aeroway fills (runways, taxiways, aprons)
        contLayer('aeroway-fill', {
            type: 'fill',
            'source-layer': 'aeroway',
            minzoom: 10,
            filter: ['any', ['==', ['geometry-type'], 'Polygon'], ['==', ['geometry-type'], 'MultiPolygon']],
            paint: {
                'fill-color': ['match', ['get', 'class'],
                    'runway', '#bbbbcc',
                    'taxiway', '#bbbbcc',
                    'apron', '#dadae0',
                    'helipad', '#bbbbcc',
                    '#dadae0'
                ],
                'fill-opacity': 0.9
            }
        }).forEach(function(l) { layers.push(l); });

        contLayer('aeroway-line', {
            type: 'line',
            'source-layer': 'aeroway',
            minzoom: 10,
            filter: ['any', ['==', ['geometry-type'], 'LineString'], ['==', ['geometry-type'], 'MultiLineString']],
            paint: {
                'line-color': '#bbbbcc',
                'line-width': ['interpolate', ['exponential', 1.2], ['zoom'],
                    10, ['match', ['get', 'class'], 'runway', 2, 'taxiway', 1, 0.5],
                    14, ['match', ['get', 'class'], 'runway', 8, 'taxiway', 4, 1],
                    18, ['match', ['get', 'class'], 'runway', 24, 'taxiway', 12, 3]
                ]
            }
        }).forEach(function(l) { layers.push(l); });

        // 8. Tunnel casings (drawn first, under normal roads)
        contLayer('tunnel-casing', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 10,
            filter: ['all', ['==', ['get', 'brunnel'], 'tunnel']],
            layout: { 'line-join': 'round' },
            paint: {
                'line-color': tunnelCasingColor,
                'line-width': roadCasingWidth,
                'line-dasharray': [0.5, 0.25],
                'line-opacity': 0.6
            }
        }).forEach(function(l) { layers.push(l); });

        // 9. Tunnel fills
        contLayer('tunnel-fill', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 10,
            filter: ['all', ['==', ['get', 'brunnel'], 'tunnel']],
            layout: { 'line-join': 'round' },
            paint: {
                'line-color': tunnelFillColor,
                'line-width': roadFillWidth,
                'line-opacity': 0.6
            }
        }).forEach(function(l) { layers.push(l); });

        // 10. Road casings (normal, not bridge/tunnel)
        contLayer('road-casing', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 5,
            filter: ['all',
                ['any', ['!has', 'brunnel'], ['==', ['get', 'brunnel'], ''], ['==', ['get', 'brunnel'], 'ford']],
                ['!=', ['get', 'class'], 'rail'],
                ['!=', ['get', 'class'], 'transit']
            ],
            layout: { 'line-cap': 'round', 'line-join': 'round' },
            paint: {
                'line-color': roadCasingColor,
                'line-width': roadCasingWidth,
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 5, 0, 7, 0.5, 9, 1]
            }
        }).forEach(function(l) { layers.push(l); });

        // 11. Road fills (normal)
        contLayer('road-fill', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 5,
            filter: ['all',
                ['any', ['!has', 'brunnel'], ['==', ['get', 'brunnel'], ''], ['==', ['get', 'brunnel'], 'ford']],
                ['!=', ['get', 'class'], 'rail'],
                ['!=', ['get', 'class'], 'transit']
            ],
            layout: { 'line-cap': 'round', 'line-join': 'round' },
            paint: {
                'line-color': roadFillColor,
                'line-width': roadFillWidth,
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 5, 0, 7, 0.8, 9, 1]
            }
        }).forEach(function(l) { layers.push(l); });

        // 12. Rail lines
        contLayer('rail', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 9,
            filter: ['any', ['==', ['get', 'class'], 'rail'], ['==', ['get', 'class'], 'transit']],
            paint: {
                'line-color': '#999999',
                'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.5, 14, 1.5, 18, 3],
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 9, 0.4, 12, 0.8]
            }
        }).forEach(function(l) { layers.push(l); });

        // Rail dash overlay
        contLayer('rail-dash', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 9,
            filter: ['any', ['==', ['get', 'class'], 'rail'], ['==', ['get', 'class'], 'transit']],
            paint: {
                'line-color': '#ffffff',
                'line-width': ['interpolate', ['linear'], ['zoom'], 9, 0.3, 14, 1, 18, 2],
                'line-dasharray': [5, 5],
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 9, 0.3, 12, 0.7]
            }
        }).forEach(function(l) { layers.push(l); });

        // 13. Bridge casings
        contLayer('bridge-casing', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 10,
            filter: ['all', ['==', ['get', 'brunnel'], 'bridge'],
                ['!=', ['get', 'class'], 'rail'], ['!=', ['get', 'class'], 'transit']
            ],
            layout: { 'line-join': 'miter' },
            paint: {
                'line-color': '#000000',
                'line-width': roadCasingWidth,
                'line-opacity': 0.6
            }
        }).forEach(function(l) { layers.push(l); });

        // 14. Bridge fills
        contLayer('bridge-fill', {
            type: 'line',
            'source-layer': 'transportation',
            minzoom: 10,
            filter: ['all', ['==', ['get', 'brunnel'], 'bridge'],
                ['!=', ['get', 'class'], 'rail'], ['!=', ['get', 'class'], 'transit']
            ],
            layout: { 'line-join': 'miter' },
            paint: {
                'line-color': roadFillColor,
                'line-width': roadFillWidth
            }
        }).forEach(function(l) { layers.push(l); });

        // 15. Buildings
        contLayer('building', {
            type: 'fill',
            'source-layer': 'building',
            minzoom: 13,
            paint: {
                'fill-color': '#d9d0c9',
                'fill-outline-color': '#b8a99d',
                'fill-opacity': ['interpolate', ['linear'], ['zoom'], 13, 0.3, 15, 0.8, 17, 0.9]
            }
        }).forEach(function(l) { layers.push(l); });

        // 16. Boundary lines (administrative borders)
        // Country borders (admin_level 2)
        contLayer('boundary-country', {
            type: 'line',
            'source-layer': 'boundary',
            minzoom: 2,
            filter: ['==', ['get', 'admin_level'], 2],
            layout: { 'line-join': 'round' },
            paint: {
                'line-color': '#8d618b',
                'line-width': ['interpolate', ['linear'], ['zoom'], 2, 0.5, 5, 1.2, 10, 2.5, 14, 3.5],
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 2, 0.5, 5, 0.8, 8, 1]
            }
        }).forEach(function(l) { layers.push(l); });

        // State/province borders (admin_level 4)
        contLayer('boundary-state', {
            type: 'line',
            'source-layer': 'boundary',
            minzoom: 3,
            filter: ['==', ['get', 'admin_level'], 4],
            layout: { 'line-join': 'round' },
            paint: {
                'line-color': '#8d618b',
                'line-width': ['interpolate', ['linear'], ['zoom'], 3, 0.3, 6, 0.5, 10, 1.5, 14, 2],
                'line-dasharray': [6, 2, 2, 2],
                'line-opacity': ['interpolate', ['linear'], ['zoom'], 3, 0.3, 6, 0.5, 8, 0.8]
            }
        }).forEach(function(l) { layers.push(l); });

        // Lower admin boundaries
        contLayer('boundary-lower', {
            type: 'line',
            'source-layer': 'boundary',
            minzoom: 8,
            filter: ['all', ['>=', ['get', 'admin_level'], 5], ['<=', ['get', 'admin_level'], 8]],
            layout: { 'line-join': 'round' },
            paint: {
                'line-color': '#ac78ab',
                'line-width': ['interpolate', ['linear'], ['zoom'], 8, 0.3, 14, 1],
                'line-dasharray': [3, 3],
                'line-opacity': 0.4
            }
        }).forEach(function(l) { layers.push(l); });

        // Also add maplibre overview borders for z0-6
        layers.push({
            id: 'maplibre-border',
            type: 'line',
            source: 'maplibre',
            'source-layer': 'countries',
            maxzoom: 7,
            paint: { 'line-color': '#8d618b', 'line-width': 1 }
        });

        // --- LABELS ---

        // 17. Water labels (blue italic)
        contLayer('water-label', {
            type: 'symbol',
            'source-layer': 'water_name',
            minzoom: 8,
            layout: {
                'text-field': nameExpr,
                'text-font': ['Noto Sans Italic'],
                'text-size': ['interpolate', ['linear'], ['zoom'], 8, 10, 14, 14, 18, 18],
                'text-max-width': 8,
                'text-letter-spacing': 0.1,
                'symbol-placement': 'point',
                'text-padding': 4,
                visibility: 'visible'
            },
            paint: {
                'text-color': '#4d80b3',
                'text-halo-color': 'rgba(255,255,255,0.7)',
                'text-halo-width': 1.5
            }
        }).forEach(function(l) { layers.push(l); });

        // 18. Road labels (along-line)
        contLayer('road-label', {
            type: 'symbol',
            'source-layer': 'transportation_name',
            minzoom: 10,
            layout: {
                'text-field': nameExpr,
                'text-font': ['Noto Sans Regular'],
                'text-size': roadLabelSize,
                'symbol-placement': 'line',
                'text-rotation-alignment': 'map',
                'text-pitch-alignment': 'viewport',
                'text-max-angle': 30,
                'text-padding': 2,
                visibility: 'visible'
            },
            paint: {
                'text-color': '#333333',
                'text-halo-color': roadLabelHalo,
                'text-halo-width': 2,
                'text-halo-blur': 0.5
            }
        }).forEach(function(l) { layers.push(l); });

        // 19. Aerodrome labels
        contLayer('aerodrome-label', {
            type: 'symbol',
            'source-layer': 'aerodrome_label',
            minzoom: 10,
            layout: {
                'text-field': nameExpr,
                'text-font': ['Noto Sans Regular'],
                'text-size': ['interpolate', ['linear'], ['zoom'], 10, 10, 14, 13],
                'text-max-width': 8,
                'text-anchor': 'top',
                'text-offset': [0, 0.5],
                visibility: 'visible'
            },
            paint: {
                'text-color': '#555555',
                'text-halo-color': '#ffffff',
                'text-halo-width': 2
            }
        }).forEach(function(l) { layers.push(l); });

        // 20. Place labels (country, state, city, town, village, etc.)
        contLayer('place-label', {
            type: 'symbol',
            'source-layer': 'place',
            minzoom: 2,
            layout: {
                'text-field': nameExpr,
                'text-font': ['step', ['zoom'],
                    ['match', ['get', 'class'], 'country', ['literal', ['Noto Sans Bold']], 'state', ['literal', ['Noto Sans Italic']], 'city', ['literal', ['Noto Sans Bold']], ['literal', ['Noto Sans Regular']]],
                    12,
                    ['match', ['get', 'class'], 'city', ['literal', ['Noto Sans Bold']], ['literal', ['Noto Sans Regular']]]
                ],
                'text-size': placeLabelSize,
                'text-max-width': 8,
                'text-transform': ['match', ['get', 'class'], 'country', 'uppercase', 'state', 'uppercase', 'none'],
                'text-letter-spacing': ['match', ['get', 'class'], 'country', 0.15, 'state', 0.1, 0.02],
                'text-padding': ['match', ['get', 'class'], 'country', 8, 'state', 6, 'city', 4, 2],
                'text-anchor': 'center',
                visibility: 'visible'
            },
            paint: {
                'text-color': ['match', ['get', 'class'],
                    'country', '#333333',
                    'state', '#555555',
                    'city', '#222222',
                    'town', '#333333',
                    'village', '#444444',
                    'suburb', '#555555',
                    'hamlet', '#666666',
                    'neighbourhood', '#666666',
                    '#444444'
                ],
                'text-halo-color': 'rgba(255,255,255,0.8)',
                'text-halo-width': ['match', ['get', 'class'],
                    'country', 2.5,
                    'state', 2,
                    'city', 2.5,
                    'town', 2,
                    'village', 2,
                    1.5
                ],
                'text-halo-blur': 0.5
            }
        }).forEach(function(l) { layers.push(l); });

        // Also: country labels from maplibre overview source (z0-6)
        layers.push({
            id: 'maplibre-country-label',
            type: 'symbol',
            source: 'maplibre',
            'source-layer': 'centroids',
            maxzoom: 7,
            layout: {
                'text-field': ['get', 'name'],
                'text-font': ['Noto Sans Bold'],
                'text-size': ['interpolate', ['linear'], ['zoom'], 1, 10, 4, 14, 6, 18],
                'text-transform': 'uppercase',
                'text-letter-spacing': 0.15,
                'text-max-width': 8,
                'text-padding': 8,
                visibility: 'visible'
            },
            paint: {
                'text-color': '#333333',
                'text-halo-color': 'rgba(255,255,255,0.8)',
                'text-halo-width': 2.5,
                'text-halo-blur': 0.5
            }
        });

        // 21. Mountain peak labels
        contLayer('mountain-peak', {
            type: 'symbol',
            'source-layer': 'mountain_peak',
            minzoom: 11,
            layout: {
                'text-field': ['concat',
                    nameExpr,
                    ['case',
                        ['has', 'ele'],
                        ['concat', '\n', ['to-string', ['get', 'ele']], ' m'],
                        ''
                    ]
                ],
                'text-font': ['Noto Sans Regular'],
                'text-size': ['interpolate', ['linear'], ['zoom'], 11, 9, 14, 11, 18, 13],
                'text-max-width': 8,
                'text-anchor': 'top',
                'text-offset': [0, 0.5],
                visibility: 'visible'
            },
            paint: {
                'text-color': '#7a4d13',
                'text-halo-color': 'rgba(255,255,255,0.7)',
                'text-halo-width': 1.5
            }
        }).forEach(function(l) { layers.push(l); });

        // 22. POI labels (high zoom)
        contLayer('poi-label', {
            type: 'symbol',
            'source-layer': 'poi',
            minzoom: 14,
            layout: {
                'text-field': nameExpr,
                'text-font': ['Noto Sans Regular'],
                'text-size': ['interpolate', ['linear'], ['zoom'], 14, 10, 18, 13],
                'text-max-width': 8,
                'text-anchor': 'top',
                'text-offset': [0, 0.5],
                visibility: 'visible'
            },
            paint: {
                'text-color': '#555555',
                'text-halo-color': '#ffffff',
                'text-halo-width': 1.5
            }
        }).forEach(function(l) { layers.push(l); });

        // 23. Housenumber labels (very high zoom)
        contLayer('housenumber-label', {
            type: 'symbol',
            'source-layer': 'housenumber',
            minzoom: 17,
            layout: {
                'text-field': ['get', 'housenumber'],
                'text-font': ['Noto Sans Regular'],
                'text-size': 10,
                visibility: 'visible'
            },
            paint: {
                'text-color': '#696969',
                'text-halo-color': '#ffffff',
                'text-halo-width': 1
            }
        }).forEach(function(l) { layers.push(l); });

        return {
            version: 8,
            name: 'OSM Bright',
            glyphs: glyphs,
            sources: sources,
            layers: layers
        };
    }

    // Export
    window.OsmStyleFull = {
        generate: generateOsmStyle
    };
})();
