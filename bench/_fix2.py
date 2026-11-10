import pathlib

p = pathlib.Path('src/_probe_par.mojo')
s = p.read_text()
s = s.replace('from std.memory import alloc, free', 'from std.memory import alloc, dealloc')
s = s.replace('free(sh)\n    free(co)\n    free(out)', 'dealloc(sh)\n    dealloc(co)\n    dealloc(out)')
p.write_text(s)
