const assert = require('assert');
var cufoojs = require('cufoojs');

console.info('[js] version = ' + cufoojs.version());

assert.throws(function() { cufoojs.add(); }, 'number of arguments');
assert.throws(function() { cufoojs.add('', 2); }, 'types of arguments');

var c = cufoojs.add(5, 3);
assert.strictEqual(c, 8, '[js] add(..): expected 8, got ' + c);

console.info('[js] ' + c);