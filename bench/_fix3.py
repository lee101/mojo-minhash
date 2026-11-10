import pathlib

p = pathlib.Path('src/_probe_par.mojo')
s = p.read_text()
s = s.replace("def chunk(k: Int) -> None:", "def chunk(k: Int) {imm}:")
s = s.replace("var sh = alloc[UInt32](S)", "var sh_mem = alloc[UInt32](S)\n    var sh = Pointer[UInt32, AnyOrigin[mut=True]](unsafe_from_address=Int(sh_mem))")
s = s.replace("var co = alloc[Int64](2 * H)", "var co_mem = alloc[Int64](2 * H)\n    var co = Pointer[Int64, AnyOrigin[mut=True]](unsafe_from_address=Int(co_mem))")
s = s.replace("var out = alloc[UInt64](H)", "var out_mem = alloc[UInt64](H)\n    var out = Pointer[UInt64, AnyOrigin[mut=True]](unsafe_from_address=Int(out_mem))")
p.write_text(s)
