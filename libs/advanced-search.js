/* Abyss — AdvancedSearch: lightweight client-side keyword query.
 *
 * Restores the keyword search reports-browse (and any future page) expects.
 * Supported syntax:
 *   gulf survey      → AND  (both terms must appear)
 *   "arabian gulf"   → exact phrase
 *   -classified      → negation (term must NOT appear)
 *   gulf OR red sea  → OR groups (any group may match)
 *
 * API (stable):
 *   AdvancedSearch.parse(queryString)          -> ast
 *   AdvancedSearch.evaluate(ast, record, fields) -> boolean
 *     record : an object; fields : array of field names to search across.
 */
(function (global) {
  'use strict';

  // Quoted phrases stay intact; bare words split on whitespace.
  function tokenize(q) {
    var tokens = [], re = /"([^"]*)"|(\S+)/g, m;
    while ((m = re.exec(q)) !== null) {
      tokens.push(m[1] !== undefined ? { text: m[1], quoted: true }
                                      : { text: m[2], quoted: false });
    }
    return tokens;
  }

  function parse(query) {
    var q = (query || '').trim();
    if (!q) return { groups: [] };
    // bare "OR" (any case) separates AND-groups; terms within a group are AND.
    var groups = [], current = [];
    tokenize(q).forEach(function (tok) {
      if (!tok.quoted && tok.text.toUpperCase() === 'OR') {
        if (current.length) { groups.push(current); current = []; }
        return;
      }
      var text = tok.text, negate = false;
      if (!tok.quoted && text.charAt(0) === '-' && text.length > 1) {
        negate = true; text = text.slice(1);
      }
      current.push({ text: text.toLowerCase(), negate: negate });
    });
    if (current.length) groups.push(current);
    return { groups: groups };
  }

  function haystack(record, fields) {
    var parts = [];
    (fields || []).forEach(function (f) {
      var v = record ? record[f] : null;
      if (v !== undefined && v !== null) parts.push(String(v));
    });
    //  separator keeps phrases from spanning two fields.
    return parts.join('  ').toLowerCase();
  }

  // Matches if ANY OR-group fully matches (every AND-term in the group satisfied).
  function evaluate(ast, record, fields) {
    if (!ast || !ast.groups || !ast.groups.length) return true;
    var hay = haystack(record, fields);
    return ast.groups.some(function (group) {
      return group.every(function (term) {
        var hit = hay.indexOf(term.text) !== -1;
        return term.negate ? !hit : hit;
      });
    });
  }

  global.AdvancedSearch = { parse: parse, evaluate: evaluate };
})(typeof window !== 'undefined' ? window : this);
