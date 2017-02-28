const assert = require('assert');
var cufoo = require('cufoo');

console.info('[js] version=' + cufoo.version());

function add_test()
{
    assert.throws(function() { cufoo.add(); }, 'number of arguments');
    assert.throws(function() { cufoo.add('', 2); }, 'types of arguments');

    var c = cufoo.add(5, 3);
    assert.strictEqual(c, 8, '[js] add(..): expected 8, got ' + c);

    console.info('[js] ' + c);
    
    /* cufoo.add() performs integer addition */
    var d = cufoo.add(5.5, 3.5);
    assert.strictEqual(d, 8.0, '[js] add(..): expected 8, got ' + d);
};

function addAll_test()
{
    var arrA = new Int32Array([1,2,3,4]);
    var arrB = new Int32Array([5,6,7,8]);

    var arrC = cufoo.addAll(arrA, arrB);
    assert.deepEqual(arrC, [6,8,10,12]);
};

add_test();
addAll_test();
