import os, sys, tempfile, shutil
sys.path.insert(0, os.path.dirname(__file__))
import pytest
from node_repl import js, js_add_node_module_dir, js_reset

async def _reset():
    await js_reset()

@pytest.mark.asyncio
async def test_arithmetic():
    await _reset()
    out = await js(code="1 + 2")
    assert "3" in out

@pytest.mark.asyncio
async def test_string():
    await _reset()
    out = await js(code="'hello world'")
    assert "hello world" in out

@pytest.mark.asyncio
async def test_console_log():
    await _reset()
    out = await js(code="console.log('foo')")
    assert "foo" in out

@pytest.mark.asyncio
async def test_json_stringify():
    await _reset()
    out = await js(code='JSON.stringify({a:1})')
    assert '{"a":1}' in out

@pytest.mark.asyncio
async def test_var_persistence():
    await _reset()
    await js(code="var x = 100")
    out = await js(code="x + 1")
    assert "101" in out

@pytest.mark.asyncio
async def test_function_call():
    await _reset()
    await js(code="function add(a,b){return a+b}")
    out = await js(code="add(10,20)")
    assert "30" in out

@pytest.mark.asyncio
async def test_let_variable():
    await _reset()
    await js(code="let count = 0")
    await js(code="count++")
    out = await js(code="count")
    assert "1" in out

@pytest.mark.asyncio
async def test_reset_clears_vars():
    await _reset()
    await js(code="var y = 999")
    await js_reset()
    out = await js(code="typeof y")
    assert "undefined" in out

@pytest.mark.asyncio
async def test_reset_returns_message():
    await _reset()
    assert "Kernel reset" in await js_reset()

@pytest.mark.asyncio
async def test_add_new_dir():
    d = tempfile.mkdtemp(prefix="nm_")
    nd = os.path.join(d, "node_modules")
    os.makedirs(nd, exist_ok=True)
    assert await js_add_node_module_dir(path=nd) is True
    shutil.rmtree(d, ignore_errors=True)

@pytest.mark.asyncio
async def test_duplicate_returns_false():
    d = tempfile.mkdtemp(prefix="nm_")
    nd = os.path.join(d, "node_modules")
    os.makedirs(nd, exist_ok=True)
    await js_add_node_module_dir(path=nd)
    assert await js_add_node_module_dir(path=nd) is False
    shutil.rmtree(d, ignore_errors=True)

@pytest.mark.asyncio
async def test_invalid_dir_raises():
    with pytest.raises(ValueError):
        await js_add_node_module_dir(path="/no_such")

@pytest.mark.asyncio
async def test_short_timeout():
    await _reset()
    out = await js(code="while(true){}", timeout_ms=500)
    assert "timed out" in out.lower()

@pytest.mark.asyncio
async def test_normal_no_timeout():
    await _reset()
    out = await js(code="'ok'", timeout_ms=5000)
    assert "ok" in out

@pytest.mark.asyncio
async def test_syntax_error():
    await _reset()
    out = await js(code="{")
    assert "SyntaxError" in out or "unexpected" in out.lower()

@pytest.mark.asyncio
async def test_reference_error():
    await _reset()
    out = await js(code="undefinedVar")
    assert "ReferenceError" in out or "not defined" in out

@pytest.mark.asyncio
async def test_empty_code():
    await _reset()
    out = await js(code="")
    assert out == "(no output)" or out == ""
