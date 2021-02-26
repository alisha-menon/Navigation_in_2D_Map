from modules.game_module import game_module

train_file = './data/game_dat.out'

live = int(input("Enter 1 to play the game live, 0 to autoplay using recall\n"))
gametype = int(input("Enter 1 to play with goals, 0 without\n"))

a = game_module()
if live:
    a.play_game(gametype)
else:
    a.train_from_file(train_file)
    a.autoplay_game(gametype)

