package = "spectralnet"
version = "scm-1"
source = {
   url = "..." --TODO
}
description = {
   summary = "Spectral network library",
   detailed = [[
   ]],
   homepage = "...",
   license = "None"
}
dependencies = {
   "torch >= 7.0"
}
build = {
   type = "command",
   build_command = [[
	 mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)" && $(MAKE) 
   ]],
   install_command = "cd build && $(MAKE) install"
}