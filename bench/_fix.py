import pathlib

p = pathlib.Path('src/kernels.mojo')
lines = p.read_text().split('\n')
start = next(i for i, l in enumerate(lines) if l.startswith('def _signature_parallel'))
end = next(i for i, l in enumerate(lines) if l.startswith('def _agree'))
del lines[start:end]
p.write_text('\n'.join(lines))
print('removed', end - start, 'lines from', start + 1)
