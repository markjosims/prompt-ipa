uv pip compile --output-file requirements.txt --universal requirements.in
uv pip compile --output-file requirements-gui.txt --universal requirements.in requirements-gui.in
uv pip compile --output-file requirements-dev.txt --universal requirements.in requirements-dev.in requirements-gui.in