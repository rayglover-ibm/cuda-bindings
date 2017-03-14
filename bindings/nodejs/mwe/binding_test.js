const assert = require('assert');
var mwe = require('mwe');

console.info('[js] version=' + mwe.version());

function add_test()
{
    assert.throws(function() { mwe.add(); }, 'number of arguments');
    assert.throws(function() { mwe.add('', 2); }, 'types of arguments');

    var c = mwe.add(5, 3);
    assert.strictEqual(c, 8, '[js] add(..): expected 8, got ' + c);

    console.info('[js] ' + c);
    
    /* mwe.add() performs integer addition */
    var d = mwe.add(5.5, 3.5);
    assert.strictEqual(d, 8.0, '[js] add(..): expected 8, got ' + d);
};

function addAll_test()
{
    var arrA = new Int32Array([1,2,3,4]);
    var arrB = new Int32Array([5,6,7,8]);

    var arrC = mwe.addAll(arrA, arrB);
    assert.deepEqual(arrC, [6,8,10,12]);
};

add_test();
addAll_test();
