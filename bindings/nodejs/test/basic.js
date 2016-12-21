const assert = require('assert');
var libsrjs = require('libsrjs');

assert.throws(function() { libsrjs.add(); }, 'number of arguments');
assert.throws(function() { libsrjs.add('', 2); }, 'types of arguments');

var c = libsrjs.add(5, 3);
assert.strictEqual(c, 8, '[js] add(..): expected 8, got ' + c);

console.info('[js] ' + c);