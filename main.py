from pyboy import PyBoy
pyboy = PyBoy('pyboy/games/Super Mario Land (World).gb')
print("Position = " + str(pyboy.game_wrapper().position))
while not pyboy.tick():
    pass
pyboy.stop()




from pyboy import WindowEvent

