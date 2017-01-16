const assert = require('assert');
var cufoo = require('cufoo');

console.info('[js] version=' + cufoo.version());

assert.throws(function() { cufoo.add(); }, 'number of arguments');
assert.throws(function() { cufoo.add('', 2); }, 'types of arguments');

var c = cufoo.add(5, 3);
assert.strictEqual(c, 8, '[js] add(..): expected 8, got ' + c);

console.info('[js] ' + c);